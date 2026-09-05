## Qwen3-0.6B Megakernel for A100

This megakernel fuses the Qwen3-0.6B (BF16) single-token decode path. A100 (`sm_80`) is the primary target; V100 is a best-effort build target and may fall back to the standard nano-vLLM backend.

More details on this blogpost: https://blog.alpindale.net/posts/5090_decode_optimization/


The following numbers are the original RTX 5090 reference result, not an
A100 claim. Re-run the benchmark on the target server before reporting them:

| Backend      | tok/s  | ms/tok | Speedup |
|--------------|--------|--------|---------|
| PyTorch (HF) | 123.3  | 8.11   | 1.00x   |
| Megakernel   | 1036.3 | 0.99   | 8.40x   |


To use this:

```bash
uv pip install -r requirements.txt
python -m qwen_megakernel.bench
```

The kernel is fixed-shape: Qwen3-0.6B, single-sequence greedy decode, and max sequence length 2048. Build on the target server with `NANOVLLM_CUDA_ARCH=sm_80` for A100 or `sm_70` for V100. CUDA correctness and performance tests must run on the GPU server, not on a development Mac.


### Credits
Based on Elliot Arledge's [MegaQwen](https://github.com/Infatoshi/MegaQwen) for the RTX 3090 GPU.
