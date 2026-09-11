// Batched WMMA GEMM matching megakernel layout: C[batch,out] = H[batch,in] * W^T
// W is stored [out, in] bf16 row-major (megakernel layout). A = matrix_a(row_major, M=batch),
// B = matrix_b(col_major so B[:,k] walks W rows) or W^T. Compute H·W^T.
// nvcc -arch=sm_80 -O3 wmma_qkv.cu
#include <cuda_bf16.h>
#include <mma.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace nvcuda;
constexpr int MT=16, NT=16, KT=16;
constexpr int HID=1024, OUT=4096; // qkv output rows

// WMMA C[MT x NT] = A[MT x KT] * B[KT x NT]; A = H slice (row-major, in=HID),
// B = W^T slice => for given k-range, B[k, n] = W[n, k] (n=out row). Load B as
// matrix_b col_major with ptr to W + n*HID + k? Let's load B as a col_major
// fragment from a temp where B[k][n] is built. Simplify: build B^T in shared as
// [out][in]? Instead compute A(row_major)*W_T where W_T[in][out]=W[out][in].
__global__ void gemm_qkv_wmma(const __nv_bfloat16* H, const __nv_bfloat16* W,
                              float* C, int M, int N, int K) {
  // H[M x K] row-major; W[K x N] = transpose of W[out,in]; here caller passes W^T
  int tr=blockIdx.x, tc=blockIdx.y;
  wmma::fragment<wmma::accumulator, MT,NT,KT,float> acc;
  wmma::fill_fragment(acc,0.f);
  for(int k=0;k<K;k+=KT){
    wmma::fragment<wmma::matrix_a,MT,NT,KT,__nv_bfloat16,wmma::row_major> a;
    wmma::fragment<wmma::matrix_b,MT,NT,KT,__nv_bfloat16,wmma::row_major> b;
    wmma::load_matrix_sync(a, H + tr*MT*K + k, K);
    // B = W^T [K x N] stored row-major as WT[k*N+n]
    wmma::load_matrix_sync(b, W + k*N + tc*NT, N);
    wmma::mma_sync(acc,a,b,acc);
  }
  wmma::store_matrix_sync(C + tr*MT*N + tc*NT, acc, N, wmma::mem_row_major);
}

// Kernel-style vectorized scalar matvec (like megakernel): each warp does rows,
// 4-wide unroll, fp32 accum; approximates the kernel's performance.
__global__ void matvec_scalar_kernelstyle(const __nv_bfloat16* H, const __nv_bfloat16* W,
                                          float* C, int M, int N, int K) {
  // one thread per (row, n) with K loop, 4-unroll fused
  int tid = blockIdx.x*blockDim.x+threadIdx.x;
  int total = (int)M*N;
  for(int idx=tid; idx<total; idx+=gridDim.x*blockDim.x){
    int r=idx/N, n=idx%N;
    const __nv_bfloat16* h=H+r*K;
    const __nv_bfloat16* w=W+n*K;
    float s0=0,s1=0,s2=0,s3=0;
    int k=0;
    for(;k+4<=K;k+=4){
      s0+=__bfloat162float(h[k])*__bfloat162float(w[k]);
      s1+=__bfloat162float(h[k+1])*__bfloat162float(w[k+1]);
      s2+=__bfloat162float(h[k+2])*__bfloat162float(w[k+2]);
      s3+=__bfloat162float(h[k+3])*__bfloat162float(w[k+3]);
    }
    C[idx]=s0+s1+s2+s3;
  }
}

int main(){
  int M=64,N=OUT,K=HID;
  size_t hb=(size_t)M*K, wb=(size_t)K*N, cb=(size_t)M*N; // W passed as W^T [K x N]
  __nv_bfloat16 *H,*W; float *C,*Csc,*Cref;
  cudaMalloc(&H,hb*2); cudaMalloc(&W,wb*2); cudaMalloc(&C,cb*4); cudaMalloc(&Csc,cb*4); cudaMalloc(&Cref,cb*4);
  std::vector<__nv_bfloat16> Hh(hb), Wh(wb); std::vector<float> CrefH(cb);
  for(int i=0;i<hb;i++) Hh[i]=__float2bfloat16(((float)(rand()%2000)-1000)/1000);
  for(int i=0;i<wb;i++) Wh[i]=__float2bfloat16(((float)(rand()%2000)-1000)/1000); // W^T
  cudaMemcpy(H,Hh.data(),hb*2,cudaMemcpyHostToDevice);
  cudaMemcpy(W,Wh.data(),wb*2,cudaMemcpyHostToDevice);
  for(int r=0;r<M;r++)for(int n=0;n<N;n++){float s=0;for(int k=0;k<K;k++)s+=__bfloat162float(Hh[r*K+k])*__bfloat162float(Wh[k*N+n]);CrefH[r*N+n]=s;}
  cudaMemcpy(Cref,CrefH.data(),cb*4,cudaMemcpyHostToDevice);
  dim3 g(M/MT,N/NT);
  gemm_qkv_wmma<<<g,32>>>(H,W,C,M,N,K);
  matvec_scalar_kernelstyle<<<M*16,256>>>(H,W,Csc,M,N,K);
  cudaDeviceSynchronize();
  std::vector<float> Ch(cb),ChS(cb),ChR(cb);
  cudaMemcpy(Ch.data(),C,cb*4,cudaMemcpyDeviceToHost);
  cudaMemcpy(ChS.data(),Csc,cb*4,cudaMemcpyDeviceToHost);
  cudaMemcpy(ChR.data(),Cref,cb*4,cudaMemcpyDeviceToHost);
  float dw=0,ds=0;for(int i=0;i<cb;i++){dw=fmax(dw,fabs(Ch[i]-ChR[i]));ds=fmax(ds,fabs(ChS[i]-ChR[i]));}
  printf("wmma correct=%s (maxdiff %.2f), scalar maxdiff %.2f\n", dw<5?"YES":"NO",dw,ds);
  int it=200;cudaEvent_t e0,e1;cudaEventCreate(&e0);cudaEventCreate(&e1);
  for(int i=0;i<10;i++){gemm_qkv_wmma<<<g,32>>>(H,W,C,M,N,K);matvec_scalar_kernelstyle<<<M*16,256>>>(H,W,Csc,M,N,K);}
  cudaDeviceSynchronize();
  cudaEventRecord(e0);for(int i=0;i<it;i++)gemm_qkv_wmma<<<g,32>>>(H,W,C,M,N,K);cudaEventRecord(e1);cudaEventSynchronize(e1);float tmw;cudaEventElapsedTime(&tmw,e0,e1);
  cudaEventRecord(e0);for(int i=0;i<it;i++)matvec_scalar_kernelstyle<<<M*16,256>>>(H,W,Csc,M,N,K);cudaEventRecord(e1);cudaEventSynchronize(e1);float tms;cudaEventElapsedTime(&tms,e0,e1);
  printf("batch=%d QKV(1024->4096) wmma %.4f ms, kernel-style scalar %.4f ms, speedup %.1fx\n", M,tmw/it,tms/it,tms/tmw);
  return 0;
}
