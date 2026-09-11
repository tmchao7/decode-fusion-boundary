// Standalone bf16 WMMA (tensor-core) GEMM microbenchmark vs scalar FMA.
// nvcc -arch=sm_80 -O3 mma_gemm.cu -o mma_gemm
#include <cuda_bf16.h>
#include <mma.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace nvcuda;
constexpr int M_TILE=16, N_TILE=16, K_TILE=16;

__global__ void gemm_wmma(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C,
                          int M, int N, int K) {
  // block = 32 threads = 1 warp; grid over M/16 x N/16 tiles
  int tr = blockIdx.x, tc = blockIdx.y; // tile row/col (16x16)
  wmma::fragment<wmma::accumulator, M_TILE, N_TILE, K_TILE, float> acc;
  wmma::fill_fragment(acc, 0.0f);
  for (int k = 0; k < K; k += K_TILE) {
    wmma::fragment<wmma::matrix_a, M_TILE, N_TILE, K_TILE, __nv_bfloat16, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, M_TILE, N_TILE, K_TILE, __nv_bfloat16, wmma::row_major> b;
    wmma::load_matrix_sync(a, A + tr*M_TILE*K + k, K);
    wmma::load_matrix_sync(b, B + k*N + tc*N_TILE, N);
    wmma::mma_sync(acc, a, b, acc);
  }
  wmma::store_matrix_sync(C + tr*M_TILE*N + tc*N_TILE, acc, N, wmma::mem_row_major);
}

__global__ void gemm_scalar(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C,
                            int M, int N, int K) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= M) return;
  for (int c = 0; c < N; c++) {
    float sum = 0;
    for (int k = 0; k < K; k++)
      sum += __bfloat162float(A[r*K+k]) * __bfloat162float(B[k*N+c]);
    C[r*N+c] = sum;
  }
}

int main() {
  int M = 64, N = 4096, K = 1024;
  size_t ab = (size_t)M*K, bb=(size_t)K*N, cb=(size_t)M*N;
  __nv_bfloat16 *A, *B; float *C, *Cscalar, *Cref;
  cudaMalloc(&A, ab*2); cudaMalloc(&B, bb*2); cudaMalloc(&C, cb*4);
  cudaMalloc(&Cscalar, cb*4); cudaMalloc(&Cref, cb*4);
  std::vector<__nv_bfloat16> Ah(ab), Bh(bb);
  std::vector<float> CrefH(cb);
  for (int i=0;i<ab;i++) Ah[i] = __float2bfloat16(((float)(rand()%1000)-500)/100);
  for (int i=0;i<bb;i++) Bh[i] = __float2bfloat16(((float)(rand()%1000)-500)/100);
  cudaMemcpy(A, Ah.data(), ab*2, cudaMemcpyHostToDevice);
  cudaMemcpy(B, Bh.data(), bb*2, cudaMemcpyHostToDevice);
  for (int r=0;r<M;r++) for (int c=0;c<N;c++){
    float s=0; for (int k=0;k<K;k++) s += __bfloat162float(Ah[r*K+k])*__bfloat162float(Bh[k*N+c]);
    CrefH[r*N+c]=s; }
  cudaMemcpy(Cref, CrefH.data(), cb*4, cudaMemcpyHostToDevice);

  dim3 wmmaGrid(M/M_TILE, N/N_TILE);
  gemm_wmma<<<wmmaGrid, 32>>>(A,B,C,M,N,K);
  gemm_scalar<<<(M+255)/256, 256>>>(A,B,Cscalar,M,N,K);
  cudaDeviceSynchronize();
  std::vector<float> Ch(cb), ChS(cb), ChR(cb);
  cudaMemcpy(Ch.data(), C, cb*4, cudaMemcpyDeviceToHost);
  cudaMemcpy(ChS.data(), Cscalar, cb*4, cudaMemcpyDeviceToHost);
  cudaMemcpy(ChR.data(), Cref, cb*4, cudaMemcpyDeviceToHost);
  float maxDiff=0, maxDiffScalar=0;
  for (int i=0;i<cb;i++){ maxDiff=fmax(maxDiff, fabs(Ch[i]-ChR[i])); maxDiffScalar=fmax(maxDiffScalar, fabs(ChS[i]-ChR[i])); }
  printf("max abs diff vs fp32 ref: wmma=%.3f scalar=%.3f\n", maxDiff, maxDiffScalar);
  printf("wmma correct: %s\n", maxDiff < 1.0 ? "YES" : "NO");

  int iters=200;
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  // warmup
  for(int i=0;i<10;i++){ gemm_wmma<<<wmmaGrid,32>>>(A,B,C,M,N,K); gemm_scalar<<<(M+255)/256,256>>>(A,B,Cscalar,M,N,K);}
  cudaDeviceSynchronize();
  cudaEventRecord(e0);
  for(int i=0;i<iters;i++) gemm_wmma<<<wmmaGrid,32>>>(A,B,C,M,N,K);
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float wms; cudaEventElapsedTime(&wms, e0, e1);
  cudaEventRecord(e0);
  for(int i=0;i<iters;i++) gemm_scalar<<<(M+255)/256,256>>>(A,B,Cscalar,M,N,K);
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float sms; cudaEventElapsedTime(&sms, e0, e1);
  printf("M=%d N=%d K=%d: wmma %.3f ms, scalar %.3f ms, speedup %.2fx\n",
         M,N,K, wms/iters, sms/iters, sms/wms);
  // also test single-row (M=1) case
  M=1; N=4096; K=1024;
  printf("(note: single-token M=1 is a GEMV; wmma needs M>=16, so little benefit)\n");
  return 0;
}
