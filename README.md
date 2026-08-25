# Self-Supervised Methods for Computation of Optimal Regularisation Parameters in Variational Image Denoising

Final year (BSc) project code — Queen Mary University of London.

**Author:** Rayyan Hameed
**Supervisor:** Dr Kostas Papafitsoros

The full written report is included in this repository as a PDF and gives the complete derivations, proofs, and results; this README summarises what the code does and how to run it.

## Overview

Total Variation (TV) denoising reconstructs a clean image from a noisy observation by solving

```
u* = argmin_u  ½‖u − y‖² + α·TV(u)
```

where `α` is a regularisation parameter that trades off noise removal against preservation of image features (edges, texture). Choosing a good `α` is normally done in a **supervised** way, by comparing candidate reconstructions against a known clean image and picking the `α` that minimises the reconstruction error. That requires a ground truth image, which frequently isn't available in real world denoising problems.

This project investigates **self-supervised** alternatives that select `α` directly from the noisy image alone, with no clean reference required, and asks how close they get to the supervised, ground truth based baseline.

## Methods Implemented

- **Supervised baseline** : the standard approach: reconstruct at each candidate `α` and pick the one minimising `‖f_α(y) − u_true‖²` against the known clean image.
- **Noise2Noise (N2N)** : the foundational idea (Lehtinen et al.) that, given two independent noisy observations of the same image, a denoiser can be trained/selected by comparing its output on one observation against the other, without ever seeing the clean image.
- **Recorrupted2Recorrupted (R2R)** : since only a single noisy image is available in practice, this method (Pang et al.) synthetically constructs two conditionally independent re-corruptions of the one observation and applies the Noise2Noise idea to them. Proved (and re-derived in the report) to be an unbiased estimator of the supervised loss.
- **Neighbour2Neighbour (N2N-loss)** : constructs a pseudo pair of "independent" observations by sub sampling neighbouring pixels (Huang et al.), combining a reconstruction term and a regularisation term.
- **SURE (Stein's Unbiased Risk Estimate)** : corrects the naive noisy residual using a divergence term that measures the reconstruction's sensitivity to small perturbations of the input, approximated via a Monte Carlo finite-difference estimator since the TV solver isn't differentiable end-to-end.
- **Spatially varying regularisation** : rather than a single global `α`, the image is split into 256 blocks of 16×16 pixels and an optimal `α_k` is selected per block (for both the supervised and SURE losses), producing a spatially varying parameter map `Λ`.

All reconstructions are computed with a **Chambolle–Pock primal-dual algorithm** (1000 iterations) for the TV denoising problem, implemented in `Tutorial_Codes.py` (kindly provided by Dr Kostas Papafitsoros).

## Key Results

Evaluated across all 24 images of the Kodak dataset (resized to 256×256, greyscale, Gaussian noise σ = 0.05):

- **SURE loss** is by far the most effective self-supervised method: mean PSNR of **31.38 dB**, versus **31.40 dB** for the supervised baseline; a gap of just 0.02 dB, generally indistinguishable to the eye.
- **R2R and N2N losses** are theoretically (R2R) or approximately (N2N) unbiased, but in practice they tend to over-regularise, smoothing out finer image detail and underperforming both the supervised baseline and SURE.
- **Spatially varying regularisation** improves on the global approach: the supervised spatially-varying scheme gains ~0.40 dB PSNR over the global supervised baseline by applying more regularisation to smooth regions (sky, flat backgrounds) and less to detailed regions (textures, edges).
- Extending spatially varying selection to **SURE at the block level underperforms**, with only 16×16 = 256 pixels per block, the spatial averaging that makes SURE's divergence estimate reliable at full image scale is no longer sufficient, increasing the variance of the block-level `α` estimates. Reconstructions remain visually reasonable despite the PSNR gap.

Full per image results, plots, and proofs are in the report PDF.

## Repository Structure

```
ThirdYearProjectCode/
├── Clean_experiment_file.ipynb   # Main notebook — run this to reproduce all experiments
├── Project_code_clean_file.py    # Supervised/self-supervised losses (R2R, N2N, SURE), golden
│                                  # section search, spatially varying block-based alpha selection
├── Tutorial_Codes.py             # Chambolle-Pock TV/TGV/Huber-TV solvers (provided by
│                                  # Dr Kostas Papafitsoros) used to actually perform the denoising
└── Self-Supervised Methods ... .pdf   # Full written report
```

## Getting Started

### Prerequisites

- Python 3.x
- `numpy`, `scipy`, `matplotlib`, `imageio`, `scikit-image`, `jupyter`

```bash
pip install numpy scipy matplotlib imageio scikit-image jupyter
```

### Dataset

The experiments use the [Kodak lossless true colour image suite](http://r0k.us/graphics/kodak/) (24 images). Download it and place the images in an `archive/` folder in the repository root (the notebook loads images as e.g. `archive/kodim21.png`). the dataset itself is not included in this repository.

### Running

```bash
git clone https://github.com/RayyanHameed/ThirdYearProjectCode.git
cd ThirdYearProjectCode
jupyter notebook Clean_experiment_file.ipynb
```

Run the notebook top to bottom to reproduce the global parameter selection experiments (all four losses on all 24 images), the PSNR/SSIM comparison plots, and the spatially varying regularisation experiments.

### Notes on parameter search

Two search strategies are implemented in `Project_code_clean_file.py`:

- **Grid search** (350 points over `[0.001, 0.2]`) — slower, but used for all final results reported in the paper, since it's more robust to the non-smooth, noisy objective produced by the Monte Carlo estimates in the self supervised losses.
- **`golden_section_search`** — a much faster bracketing search that was tried during development but found to consistently underperform, as it assumes a smooth, unimodal objective that the self-supervised losses don't reliably have.

## Acknowledgements

Thanks to Dr Kostas Papafitsoros for supervision throughout the project and for providing `Tutorial_Codes.py`, the Chambolle-Pock primal-dual solver implementation used for all TV (and TGV) denoising in this project.

## References

- Rudin, Osher, Fatemi. *Nonlinear total variation based noise removal algorithms.* Physica D, 1992.
- Chambolle, Pock. *A first-order primal-dual algorithm for convex problems with applications to imaging.* JMIV, 2011.
- Lehtinen et al. *Noise2Noise: Learning image restoration without clean data.* arXiv:1803.04189, 2018.
- Pang, Zheng, Quan, Ji. *Recorrupted-to-recorrupted: Unsupervised deep learning for image denoising.* CVPR, 2021.
- Huang, Li, Jia, Lu, Liu. *Neighbor2Neighbor: Self-supervised denoising from single noisy images.* CVPR, 2021.
- Ramani, Blu, Unser. *Monte-Carlo SURE: A black-box optimization of regularization parameters for general denoising algorithms.* IEEE TIP, 2008.
- Tachella, Davies. *Self-supervised learning from noisy and incomplete data.* arXiv:2601.03244, 2026.
- Kofler et al. *Learning regularization parameter-maps for variational image reconstruction using deep neural networks and algorithm unrolling.* SIAM J. Imaging Sciences, 2023.
- Vu, Kofler, Papafitsoros. *Deep unrolling for learning optimal spatially varying regularisation parameters for total generalised variation.* SSVM, 2025.

Full bibliography in the report PDF.
