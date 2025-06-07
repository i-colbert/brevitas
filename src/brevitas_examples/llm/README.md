# Few-Bit LLM Quantization with Qronos

This branch is intended to ease reproduction of the results in our paper: "[Qronos: Correcting the Past by Shaping the Future... in Post-Training Quantization](https://arxiv.org/abs/2505.11695)"

🚨 This branch is not intended to be maintained, Qronos support is added to mainline Brevitas here:
https://github.com/Xilinx/brevitas/pull/1311

## Citation

```
@article{zhang2025qronos,
      title={Qronos: Correcting the Past by Shaping the Future... in Post-Training Quantization}, 
      author={Shihao Zhang and Haoyu Zhang and Ian Colbert and Rayan Saab},
      year={2025},
      eprint={2505.11695},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.11695}, 
}
```

## Requirements for these experiments

See [README](https://github.com/i-colbert/brevitas?tab=readme-ov-file#requirements) for Brevitas requirements. Below are versions used for this work.

- python==3.12
- torch==2.4.0
- datasets==3.2.0
- optimum==1.24.0
- accelerate==1.3.0
- transformers==4.51.3 (custom fork, see below)
- fast_hadamard-transform==1.0.4 (custom fork, see below)
- lighteval==0.6.0 (custom fork, see below)


Experiments were run on an AMD MI210 with ROCm 6.1, which can be installed via:
```shell
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1
```

You can install a fork of the `fast_hadamard_transform` library with ROCm support via:
```shell
git clone https://github.com/jeffdaily/fast-hadamard-transform -b rocm
cd fast-hadamard-transform
pip install -e .
```

Known issue with `lighteval` v0.6.0 (see [here](https://github.com/huggingface/lighteval/issues/489)). To collect zero-shot results, please use the patched fork:
```shell
git clone https://github.com/Giuseppe5/lighteval
cd lighteval
pip install .
```
The specific configurations for these benchmarks are specified in the YML file.

Known issue with `transformers` v4.51.3 (see [here](https://github.com/huggingface/transformers/issues/38271)). To use QuaRot and SpinQuant here, please use the patched fork:
```shell
git clone https://github.com/i-colbert/transformers -b v4.51.3-patch
cd transformers
pip install -e .
```

## Reproducing results

The results in our paper use Llama3 models:
- [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)
- [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B)
- [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)

One can specify another model via the CLI, for example: `--model=meta-llama/Llama-3.2-1B-Instruct`

We provide base configurations in the `configs/` folder. One can also collect the baseline BF16 results by specifying `--no-quantize` in the CLI args, for example:
```shell
python main.py --config=config/llama3-w4-none.yml --model=meta-llama/Llama-3.2-1B --no-quantize
```

### 4-bit weight-only quantization

To collect weight-only results, we use `config/llama3-w4-none.yml` as our base configuration via:
```shell
python main.py --config=config/llama3-w4-none.yml --model=meta-llama/Llama-3.2-1B
```

The base config runs round-to-nearest (RTN). One can collect OPTQ (also known as GPTQ), GPFQ, and Qronos results by adding `--gptq`, `--gpfq`, and `--qronos`, respectively. For example,

```shell
python main.py --config=config/llama3-w4-none.yml --model=meta-llama/Llama-3.2-1B --qronos
```

We collect 3-bit weight-only (i.e., W3) results via:
```shell
python main.py --config=config/llama3-w4-none.yml --model=meta-llama/Llama-3.2-1B --weight-bit-width=3
```

SmoothQuant can be enabled by adding the following to the config:
```yaml
act_equalization: layerwise  # enables SmoothQuant (i.e., activation equalization)
act_equalization_alpha: 0.3  # we use alpha=0.3
```

MagR can be enabled by adding the following to the config:
```yaml
magr: true  # enables MagR
magr_alpha: 0.01  # we use alpha=0.01
```

Hadamard-based incoherence processing (HIP) can be enabled by adding the following to the config:
```yaml
rotation: layerwise  # enables layerwise rotation
rotation_mode: had  # specifies Hadamard rotations
```

Note that we enable both HIP and MagR for 2-bit and 1.58-bit weight-only quantization. We provide the joint config as `config/llama3-w2-hip-magr.yml`, which is run via:
```shell
python main.py --config=config/llama3-w2-hip-magr.yml --model=meta-llama/Llama-3.2-1B --weight-bit-width=2
```
Similarly, 1.58-bit results are collected via:
```shell
python main.py --config=config/llama3-w2-hip-magr.yml --model=meta-llama/Llama-3.2-1B --weight-bit-width=2 --weight-narrow-range
```
where `--weight-bit-width=2 --weight-narrow-range` restricts the quantization alphabet to $\mathcal{A}=\{-1, 0, 1\}$.

### 4-bit weight-activation quantization

To collect weight-activation QuaRot results, we use `config/llama3-w4a4-quarot.yml` as our base configuration via:
```shell
python main.py --config=config/llama3-w4a4-quarot.yml --model=meta-llama/Llama-3.2-1B
```

To collect weight-activation SpinQuant results, we use `config/llama3-w4a4-spinquant.yml` as our base configuration via:
```shell
python main.py --config=config/llama3-w4a4-quarot.yml --model=meta-llama/Llama-3.2-1B
```

Again, adding `--qronos`, `--gptq`, or `--gpfq` also runs Qronos, OPTQ (formely known as GPTQ), or GPFQ.
