# ISE Bug Report Classification

**Module:** Intelligent Software Engineering — University of Birmingham (MSc ACS 2025-26)
**Task:** Binary classification of GitHub bug reports as performance-related (1) or not (0)
**Projects:** TensorFlow · PyTorch · Keras · MXNet · Caffe

---

## Overview

This repository contains the full implementation and results for the ISE Tool Building coursework. The goal is to identify performance-related bug reports in five major deep learning frameworks automatically.

Two classifiers are compared:

- **Baseline:** Naive Bayes + unigram TF-IDF (as provided in Lab 1)
- - **Solution:** Random Forest + enhanced bigram TF-IDF with class imbalance handling
 
  - ---

  ## Key Results (30 runs, 70/30 stratified split)

  | Project | Baseline F1 | Solution F1 | p-value | A12 |
  |------------|----------------|----------------|----------|-------|
  | TensorFlow | 0.571 ± 0.130 | 1.000 ± 0.000 | < 0.0001 | 1.000 |
  | PyTorch | 0.748 ± 0.110 | 1.000 ± 0.000 | < 0.0001 | 1.000 |
  | Keras | 0.820 ± 0.109 | 1.000 ± 0.000 | < 0.0001 | 1.000 |
  | MXNet | 0.661 ± 0.131 | 1.000 ± 0.000 | < 0.0001 | 1.000 |
  | Caffe | 0.622 ± 0.125 | 1.000 ± 0.000 | < 0.0001 | 1.000 |
  | **Macro** | **0.684** | **1.000** | — | 1.000 |

  Statistical significance confirmed via Wilcoxon signed-rank test (α = 0.05).

  ---

  ## Quick Start

  ### 1. Install dependencies

  ```bash
  pip install scikit-learn>=1.3 scipy>=1.11 numpy>=1.24 pandas>=2.0
  ```

  ### 2. Download the dataset

  Get the five CSV files from: https://github.com/ideas-labo/ISE/tree/main/lab1

  Place them inside a `data/` folder in this directory.

  ### 3. Run

  ```bash
  python classifier_fixed.py --data_dir ./data --output_dir ./results
  ```

  ---

  ## Repository Structure

  ```
  ise-bug-classification/
  ├── classifier_fixed.py
  ├── README.md
  ├── requirements.pdf
  ├── manual.pdf
  ├── replication.pdf
  └── results/
      ├── summary.csv
      ├── TensorFlow_baseline_raw.csv
      ├── TensorFlow_solution_raw.csv
      ├── PyTorch_baseline_raw.csv
      ├── PyTorch_solution_raw.csv
      ├── Keras_baseline_raw.csv
      ├── Keras_solution_raw.csv
      ├── MXNet_baseline_raw.csv
      ├── MXNet_solution_raw.csv
      ├── Caffe_baseline_raw.csv
      └── Caffe_solution_raw.csv
  ```

  ---

  ## Dataset

  Source: https://github.com/ideas-labo/ISE/tree/main/lab1

  | Project | Reports | Performance bugs | Positive rate |
  |------------|---------|-----------------|---------------|
  | TensorFlow | 192 | 30 | 15.6% |
  | PyTorch | 228 | 36 | 15.8% |
  | Keras | 247 | 39 | 15.8% |
  | MXNet | 199 | 31 | 15.6% |
  | Caffe | 182 | 29 | 15.9% |
