# VIPER: Variational Inference for Pattern Extraction and Recognition in Genomic Sequences

This repository contains the **official reference implementation** for the paper:

**Variational Inference for Pattern Extraction and Recognition in Genome Sequences using State Space Models for Cancer Detection**

VIPER is a hybrid deep learning framework for **cancer-causing mutation detection from genomic sequences**, combining **Conv1D layers** for local motif extraction with **Mamba-based State Space Models (SSMs)** for efficient long-range dependency modeling. The architecture is designed for scalability, interpretability, and clinical relevance in genome-wide mutation analysis.

---

## Repository Overview

This codebase supports the full experimental pipeline described in the paper, including:

- VCF-based genomic data preprocessing (COSMIC, GRCh37)
- Fixed-length genomic window extraction (101 bp)
- One-hot nucleotide encoding (101 × 4)
- VIPER hybrid architecture (Conv1D + Mamba blocks)
- Binary classification of driver vs non-driver mutations
- Training, evaluation, and ablation-ready components

The repository is intended for **reproducibility, inspection, and extension**, not redistribution of proprietary genomic datasets.

---

## Model Architecture

VIPER integrates local and global sequence modeling through a stacked hybrid design:

- **Conv1D layers** capture short-range nucleotide motifs (6–10 bp)
- **Group Normalization + ReLU** ensure stability across heterogeneous genomic samples
- **Mamba blocks (State Space Models)** model long-range genomic interactions with linear time complexity
- **Multiplicative gating** selectively amplifies biologically relevant signals
- **Dense classification head** performs binary mutation prediction

<p align="center">
  <img src="figures/model_architecture.pdf" width="800">
</p>

*Figure: Model Architecture showcasing Variational Inference integrated with State Space Models for pattern extraction and recognition in genome sequences.*

---

## Dataset and Preprocessing

The model is evaluated on **coding point mutations from COSMIC**, aligned to the **GRCh37 reference genome** and stored in **VCF v4.1** format.

Preprocessing steps include:

- Filtering for high-confidence variants (`FILTER = PASS`)
- Extraction of a **101 bp genomic context window** centered on each mutation
- One-hot encoding into a binary nucleotide matrix
- Removal of low-frequency and low-variance mutations
- Stratified train / validation / test split (70 / 15 / 15)

Driver mutations are labeled using **Cancer Gene Census (CGC) Tier 1 annotations**.

---

## Training Configuration

All experiments follow a standardized configuration:

- Optimizer: Adam  
- Learning rate: 1e-4  
- Batch size: 64  
- Epochs: 30  
- Loss function: Binary Cross-Entropy  
- Hardware: NVIDIA A100 (80GB)

Random seeds are varied across runs to ensure statistical robustness.

---

## Performance Metrics

<p align="center">
  <img src="figures/metrics_grid.pdf" width="900">
</p>

*Figure: Loss, Accuracy, Validation Loss, Validation Accuracy, Precision, Recall, F1 Score, AUC-ROC, and AUC-PR.*

---

## Ablation Study

To assess architectural contributions, we evaluate:

- Conv1D-only (local modeling)
- Mamba-only (global modeling)
- Full VIPER (hybrid)

<p align="center">
  <img src="figures/ablation_f1.png" width="600">
</p>

*Figure: F1-score comparison showing the performance gain from hybrid integration.*

---

## Comparison with State-of-the-Art Models

VIPER is compared against BERT, HyenaDNA, and S4 across accuracy, precision, recall, and runtime.

<p align="center">
  <img src="figures/runtime_comparison.png" width="600">
</p>

<p align="center">
  <img src="figures/recall_comparison.png" width="600">
</p>

<p align="center">
  <img src="figures/precision_comparison.png" width="600">
</p>

<p align="center">
  <img src="figures/accuracy_comparison.png" width="600">
</p>

These results demonstrate that VIPER achieves **higher accuracy and recall** while maintaining **lower runtime per epoch**, making it suitable for genome-wide analysis.

---

## Scaling Behavior

<p align="center">
  <img src="figures/length_sensitivity.png" width="650">
</p>

*Figure: Model accuracy across increasing genomic context windows, highlighting VIPER’s stability for long sequences.*

---

## Clinical Reliability

<p align="center">
  <img src="figures/confusion_matrix.png" width="450">
</p>

*Figure: Normalized confusion matrix showing a low false-negative rate, critical for clinical screening.*

---

## Reproducibility Notes

- COSMIC data is not included due to licensing restrictions
- Example scripts use placeholder tensors by default
- Preprocessing utilities are fully implemented and dataset-agnostic
- All architectural components follow the paper specification

---

## Citation

If you use this code or build upon it, please cite the corresponding paper as described in the manuscript.

---

## License and Usage

This repository is intended for **academic research and reproducibility purposes only**.
