# Dynamic Hessian-Based Precision Scaling for Post-Training Quantization


## Repository Structure

```
combined_project/
├── 1_baseline_GPTQ/          # Uniform GPTQ baseline (Frantar et al., 2022)
│   ├── gptq.py               # Core GPTQ quantization algorithm
│   ├── opt.py                # OPT model quantization entry point
│   ├── bloom.py              # BLOOM model quantization entry point
│   ├── quant.py              # Quantization primitives
│   ├── datautils.py          # Calibration data loading (WikiText2, PTB, C4)
│   ├── modelutils.py         # Model utility functions
│   ├── llama.py              # LLaMA model support
│   ├── quant_cuda.cpp        # CUDA kernel bindings
│   ├── quant_cuda_kernel.cu  # CUDA quantization kernel
│   ├── setup_cuda.py         # CUDA build script
│   └── zeroShot/             # Zero-shot evaluation harness
│       ├── main.py           # Evaluation entry point
│       ├── evaluator.py      # Task evaluator
│       ├── tasks/            # LAMBADA, ARC, PIQA, StoryCloze tasks
│       └── models/           # Model wrappers for zero-shot eval
│
├── 2_dynamic_GPTQ/           # Our proposed method: Dynamic Hessian-Based Precision Scaling
│   ├── main.py               # Entry point — runs Algorithm 1 end-to-end
│   ├── gptq.py               # Modified GPTQ with per-layer bit-width support
│   ├── opt.py                # OPT model with dynamic precision scaling
│   ├── bloom.py              # BLOOM model with dynamic precision scaling
│   ├── quant.py              # Quantization primitives (extended for mixed precision)
│   ├── utils.py              # FIM sensitivity profiling & bit allocation (Algorithm 1)
│   ├── datautils.py          # Calibration data loading
│   └── modelutils.py         # Model utility functions
│
└── 3_results/                # Experimental results (Tables 1–11 in the report)
    ├── baseline/
    │   ├── perplexity/       # WikiText2, PTB, C4 perplexity for FP16, RTN, GPTQ
    │   │   ├── OPT_family/   # OPT-125M, 350M, 1.3B, 2.7B, 6.7B
    │   │   └── BLOOM_family/ # BLOOM-560M, 1.1B, 1.7B, 3B, 7.1B
    │   └── zeroshot/         # LAMBADA, ARC, PIQA, StoryCloze accuracy
    │       ├── OPT_family/
    │       └── BLOOM_family/
    └── dynamic/
        ├── perplexity/       # FP16 (Dynamic) perplexity results
        │   ├── OPT_family/
        │   └── BLOOM_family/
        └── zeroshot/         # FP16 (Dynamic) zero-shot accuracy
            ├── OPT_family/
            └── BLOOM_family/
```

---

## Method Summary (Algorithm 1)

Our framework implements three stages:

1. **Sensitivity Profiling** (`utils.py` → `PROFILESENSITIVITY`): Computes per-layer
   Fisher Information Matrix (FIM) traces as a proxy for quantization sensitivity (Eq. 1).

2. **Bit Allocation** (`utils.py` → `ALLOCATEBITS`): Greedy sequential bit-reduction
   assigns discrete bit-widths {2, 3, 4, 8} subject to a global average bit-rate
   constraint `Btarget` (Eq. 2).

3. **Quantization with Error Compensation** (`gptq.py` → `QUANTIZEMODEL`): Applies
   GPTQ's second-order weight update per layer using its assigned bit-width (Eq. 3).

---

## Baselines

| Method        | Description                                      |
|---------------|--------------------------------------------------|
| FP16          | Uncompressed full-precision model                |
| RTN-4 / RTN-3 | Round-to-nearest at 4-bit / 3-bit               |
| GPTQ-4        | Uniform 4-bit GPTQ (Frantar et al., 2022)       |
| GPTQ-3        | Uniform 3-bit GPTQ (Frantar et al., 2022)       |
| FP16 (Dyn.)  | **Ours** — mixed-precision, Hessian-guided      |

---

## Evaluation

- **Perplexity**: WikiText2, PTB, C4 (Tables 1–3 in report)
- **Zero-Shot Accuracy**: LAMBADA, ARC-Easy, ARC-Challenge, PIQA, StoryCloze (Tables 4–8)
- **Hardware**: Single NVIDIA A100 80GB GPU

---

## Reference

Frantar et al. (2022). *GPTQ: Accurate Post-Training Quantization for Generative
Pre-Trained Transformers.* arXiv:2210.17323
