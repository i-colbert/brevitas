# Pushing the Limits of Block Rotations in Post-Training Quantization

This branch is intended to ease the reproduction of the experiments from our paper: "[Pushing the Limits of Block Rotations in Post-Training Quantization](https://arxiv.org/pdf/2601.22347)"

🚨 This branch is not intended to be maintained, PeRQ support is added to mainline Brevitas: https://xilinx.github.io/brevitas/dev/papers/perq.html

## Citation

```
@article{sanjeet2026perq,
      title={Pushing the Limits of Block Rotations in Post-Training Quantization},
      author={Sai Sanjeet and Ian Colbert and Pablo Monteagudo-Lago and Giuseppe Franco and Yaman Umuroglu and Nicholas J. Fraser},
      year={2026},
      eprint={2601.22347},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.22347},
}
```

## Requirements for these experiments

See [README](https://github.com/i-colbert/brevitas?tab=readme-ov-file#requirements) for general Brevitas requirements. Below are versions used here.

- python==3.12
- torch==2.6.0+rocm6.1
- transformers==4.57.3
- lighteval==0.13.0
- fast_hadamard_transform==1.0.4.post1 (custom fork, see below)

### Installation

Install PyTorch with ROCm 6.1 support:
```shell
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1
```

Install the `fast_hadamard_transform` library with ROCm support:
```shell
git clone https://github.com/jeffdaily/fast-hadamard-transform -b rocm
cd fast-hadamard-transform
pip install -e . --no-build-isolation
```

## Reproducing Results

### Quantization Formats

The configuration files specify three quantization formats:
- **INT4**: Integer 4-bit weight-activation quantization
- **FP4**: Floating-point 4-bit (E2M1) weight-activation quantization
- **MXFP4**: Microscaling FP4 weight-activation quantization

### Pipeline Compositions

We provide configurations for the following pipeline compositions for all three quantization formats:

- **PeRQ\***: `{format}/llama3-perq_star-{format}.yml`
- **PeRQ†**: `{format}/llama3-perq_dag-{format}.yml`
- **MR-Qronos**: `{format}/llama3-mr-qronos-{format}.yml`
- **MR-GPTQ** [1]: `{format}/llama3-mr-gptq-{format}.yml`
- **BRQ-Spin** [2]: `{format}/llama3-brq-spin-{format}.yml`

Please see the PeRQ paper or the corresponding references for details on each composition. Fully online rotation configurations are also available in the `online/` subdirectories.

All config files specify Llama-3.2-1B-Instruct by default. You can choose a different model using the `--model` flag. For example:
```shell
brevitas_ptq_llm --config=int4/llama3-perq_star-int4.yml --model=meta-llama/Llama-3.2-3B-Instruct
```

Below, we summarize WikiText2 perplexity when quantizing Llama3.2-1B-Instruct to various formats (lower is better).

| Configuration | INT4 | FP4 | MXFP4 |
|--------------|------|-----|-------|
| **MR-GPTQ** [1]   |  2256.0    |  43.2   |  14.2     |
| **BRQ-Spin** [2]    |   1456.0   |   51.2  |   14.9    |
| **MR-Qronos**    |   41.8   |   23.9  |  14.0     |
| **PeRQ\***    |  16.9    |  21.0   |   14.2    |
| **PeRQ†**    |   **15.9**   |  **18.0**   |  **13.2**     |

The full-precision BF16 model gives a perplexity of 11.8.

## References

1. Egiazarian, Vage, et al. "Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization." arXiv preprint (2025).

2. Shao, Yuantian, et al. "Block Rotation is All You Need for MXFP4 Quantization." arXiv preprint (2025).
