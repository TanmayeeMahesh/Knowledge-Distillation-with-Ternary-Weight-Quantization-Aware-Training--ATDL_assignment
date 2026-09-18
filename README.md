# ATDL Assignment 1 — Knowledge Distillation + Ternary Quantization-Aware Training

CIFAR-10, ResNet34 teacher → ternary ResNet18 student, trained with combined
Knowledge Distillation (Hinton et al., 2015) and Quantization-Aware Training
using Ternary Weight Networks (Li & Liu, 2016) with a clipped/saturated
straight-through estimator (Bengio et al., 2013; Yin et al., 2019).

**The implementation is notebook-based, not script-based.** All code lives in
two Jupyter notebooks, run cell-by-cell in order. There are no standalone
`.py` training scripts other than `verify.py`.

---

## 1. Repository Structure

```
DELIVERABLES-AS1/
├── AS1_ATDL (notebooks)/
│   ├── Main.ipynb                  # Everything: data pipeline, teacher, baseline,
│   │                                #   ternary layer, KD+QAT training, evaluation, plots
│   ├── Test_on_cifar_net.ipynb     # Out-of-distribution test on EleutherAI/cifarnet
│   └── verify.py                   # standalone ternary-constraint verification script
├── checkpoints/
│   ├── pth/                        # all saved model weights
│   │   ├── teacher_resnet34_best.pth
│   │   ├── teacher_resnet34.pth
│   │   ├── baseline_resnet18_best.pth
│   │   ├── baseline_r18_fp32.pth
│   │   ├── student_main_T4_lam0.9_best.pth
│   │   ├── student_ablation_T2_lam0.9_best.pth
│   │   └── student_r18_ternary_kd.pth
│   ├── teacher_history.json
│   ├── teacher_resnet34_summary.json
│   ├── baseline_r18_fp32_summary.json
│   ├── student_r18_ternary_kd_summary.json
│   ├── compression_summary.json
│   ├── global_sparsity_summary.json
│   ├── final_evaluation_full.json
│   └── ternary_verification_report.json   # output of verify.py
├── figures/                         # all training curves, confusion matrices, histograms
│   ├── teacher_training_curves.png
│   ├── teacher_train_val_curves.png
│   ├── baseline_training_curves.png
│   ├── baseline_train_val_curves.png
│   ├── student_main_curves.png
│   ├── student_ablation_curves.png
│   ├── all_models_val_acc_comparison.png
│   ├── T_ablation_comparison.png
│   ├── confusion_matrices_all_models.png
│   ├── per_class_accuracy_comparison.png
│   ├── latent_weight_histograms.png
│   ├── ternary_forward_weight_distribution.png
│   └── layerwise_sparsity_alpha.png
├── Results/
│   ├── Main_results/                # final per-model metrics (from Main.ipynb)
│   │   ├── teacher_final_results.json
│   │   ├── baseline_final_results.json
│   │   ├── student_main_T4_lam0.9_results.json
│   │   ├── student_ablation_T2_lam0.9_results.json
│   │   └── final_evaluation_summary.csv
│   └── cifarnet_test_results.json   # OOD test results (from Test_on_cifar_net.ipynb)
├── AS1_Tanmayee KM_134.pdf            # report
├── README.md
└── requirements.txt
```

## 2. Environment Setup

```bash
pip install -r requirements.txt
```

Python 3.x with a CUDA-capable GPU is expected for training (an RTX-class GPU
was used). `verify.py` runs fine on CPU.

## 3. How to Reproduce

This is **not** a `python train.py` pipeline. To reproduce:

1. Open `AS1_ATDL (notebooks)/Main.ipynb` in Jupyter or Colab.
2. Run every cell in order, top to bottom. The notebook is organized into
   sections matching Tasks 1–4 (data pipeline → teacher → baseline → ternary
   layer → KD+QAT training → evaluation → plots).
3. Open `AS1_ATDL (notebooks)/Test_on_cifar_net.ipynb` separately and run its
   cells in order to reproduce the out-of-distribution CIFARNet check
   (requires uploading a trained student `.pth` when prompted).
4. To independently verify a saved checkpoint's ternary constraint without
   the notebook (run from inside `AS1_ATDL (notebooks)/`, adjusting the
   relative path to `checkpoints/pth/` as needed):
   ```bash
   python verify.py --checkpoint ../checkpoints/pth/student_main_T4_lam0.9_best.pth
   ```

All random splits/initializations use a fixed seed of **134** for
reproducibility (set at the top of `Main.ipynb`).

## 4. Experiment Setup

### Data
- CIFAR-10, split **45,000 train / 5,000 validation / 10,000 test** (fixed
  seed 134). Test set touched only once per model, at final evaluation.
- Augmentation (train only): pad-4 random crop to 32×32, random horizontal
  flip, per-channel normalize.

### Architectures
CIFAR-adapted ResNet (3×3 stem, no initial max-pool), built from a shared
`BasicBlock`:
- **Teacher**: ResNet34 (`[3,4,6,3]` blocks), FP32, 21,282,122 params.
- **Baseline**: ResNet18 (`[2,2,2,2]` blocks), FP32, no KD, 11,173,962 params.
- **Student**: ResNet18 with the same block layout, all Conv/FC weights
  ternary-constrained (including the first conv and final FC — a deliberate
  deviation from TWN/TTQ's own convention of exempting those two layers,
  chosen to match this assignment's literal wording). BatchNorm and all
  biases stay FP32.

### Optimizer / Schedule (identical for teacher, baseline, and student)
| Setting | Value |
|---|---|
| Optimizer | SGD, momentum 0.9, weight decay 1e-4 |
| Initial LR | 0.1 |
| LR schedule | Step decay ÷10 at epochs 100 and 150 |
| Epochs | 200 |
| Batch size | 128 |

The ternary student's latent weights are initialized from the trained FP32
baseline checkpoint (progressive-quantization warm start), rather than from
random initialization.

## 5. Ternary Quantization (Task 3)

Forward pass follows **TWN** (Li & Liu, 2016), computed per-layer (not
per-channel):

| Symbol | Formula |
|---|---|
| Δ | `0.7 * mean(|W|)` |
| α | mean(|W_i|) over surviving weights where `|W_i| > Δ` |
| Forward weight | `+α` if `W > Δ`, `-α` if `W < -Δ`, else `0` |

Backward pass uses a **clipped/saturated straight-through estimator**:
gradient passes through unchanged where `|W| <= clip_value` (1.0), zeroed
otherwise. This is the convention introduced by Hubara et al. (2016) for
weight binarization, not TWN's own literal rule (which is plain identity) —
its use here is motivated by the general stability advantage that
bounded/clipped STE surrogates show in Yin et al. (2019)'s convergence
analysis, which is proven for *activation* quantization, not weight
quantization; the connection here is by analogy, not literal citation. See
the report's discussion section for the full citation trail.

## 6. Knowledge Distillation Loss (Task 4)

Hinton et al. (2015), exact formulation:

```
L = (1 - λ) * CE(y, p_s) + λ * T² * KL(p_t^T || p_s^T)
```

Teacher is loaded from the best teacher checkpoint, set to `.eval()`, and
its parameters are frozen (`requires_grad=False`); its forward pass runs
under `torch.no_grad()`. Verified: zero gradient reaches teacher parameters.

Two runs were trained (both 200 epochs, identical setup otherwise):

| Run | T | λ | Checkpoint |
|---|---|---|---|
| Main | 4 | 0.9 | `student_main_T4_lam0.9_best.pth` |
| Ablation | 2 | 0.9 | `student_ablation_T2_lam0.9_best.pth` |

## 7. Results

### Validation / Test (best-checkpoint-by-val-accuracy, test set touched once)

| Model | Best Val Epoch | Best Val Acc | Test Loss | Test Acc |
|---|---|---|---|---|
| Teacher (ResNet34, FP32) | 177 | 95.00% | 0.2726 | 94.39% |
| Baseline (ResNet18, FP32, no KD) | 155 | 94.48% | 0.2582 | 94.13% |
| Ternary Student + KD (T=4, λ=0.9) | 192 | 94.30% | 0.3455 | 94.29% |
| Ternary Student + KD (T=2, λ=0.9, ablation) | 124 | 94.48% | 0.2935 | 94.28% |

Full per-epoch curves: `figures/*_training_curves.png`,
`figures/*_train_val_curves.png` per model; side-by-side comparison in
`figures/all_models_val_acc_comparison.png` and
`figures/T_ablation_comparison.png`. Confusion matrices and per-class
accuracy in `figures/confusion_matrices_all_models.png` /
`figures/per_class_accuracy_comparison.png`. Per-model final metrics as
individual JSON/CSV files are in `Results/Main_results/`.

### Compression (main student run, T=4)

**Theoretical storage** (weights only, computed from the actual model
structure — ternary weights at 2 bits + one FP32 α per layer + FP32
BatchNorm/bias parameters):

| Model | Total Params | Storage |
|---|---|---|
| Teacher (ResNet34, FP32) | 21,282,122 | 81.185 MB |
| Baseline (ResNet18, FP32) | 11,173,962 | 42.625 MB |
| Ternary Student (theoretical) | 11,173,962 latent | **2.699 MB** |

**Compression ratio: 15.8× vs. FP32 baseline, 30.08× vs. FP32 teacher.**

**Important distinction:** this theoretical figure is *not* the size of the
saved `.pth` file. The `.pth` checkpoint stores the student's **latent FP32
weights** (needed to resume/re-verify training), so the file on disk is
close in size to the FP32 baseline's checkpoint, not 2.699 MB. The 2.699 MB
figure represents the deployment-time storage achievable if only the
ternary indices + per-layer α scales were serialized (i.e. discarding the
latent weights, as TTQ's own paper describes doing at inference time) —
this repository does not implement that separate export format, only the
theoretical calculation. Full breakdown: `checkpoints/compression_summary.json`.

Per-layer sparsity/α values (from `verify.py`, run on
`student_main_T4_lam0.9_best.pth`): sparsity ranges roughly 49–80% across
the 21 ternary layers; full table in `figures/layerwise_sparsity_alpha.png`
and `checkpoints/global_sparsity_summary.json`.

### Out-of-distribution check (CIFARNet, 10 images/class, main T=4 checkpoint)

Overall accuracy: **78.00%** (vs. 94.29% in-distribution CIFAR-10 test
accuracy). Per-class breakdown in `Results/cifarnet_test_results.json`.

## 8. Reproducibility Checklist

| Item | Value |
|---|---|
| Random seed | 134 |
| Data split | Fixed 45k/5k/10k, seed 134 |
| Optimizer/schedule | Identical across teacher/baseline/student |
| Test set usage | Touched exactly once per model |
| Teacher initialization for student | From trained FP32 baseline checkpoint |

## 9. Limitations

See the report for the full discussion, including: real hardware speedup
requires specialized low-bit inference kernels (none are implemented or
benchmarked here — all reported gains are theoretical storage/param
calculations, not measured latency); and the sensitivity of ternarizing the
first and last layers, which deviates from TWN/TTQ's own practice of
keeping those FP32.
