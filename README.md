# Predicting the Traffic Impact Severity of US Accidents with Tree-Based and Deep Tabular Models

> Anonymous submission under double-blind review. Author and affiliation details are withheld.

## Overview

In the [US Accidents (2016–2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) dataset, **Severity** describes the impact of an accident on traffic flow (duration of congestion), from Severity 1 (shortest delay) to Severity 4 (longest delay). Predicting it can help road users and emergency responders make proactive decisions.

This repository contains the code for a study that:

1. Identifies the factors most associated with high-severity (3–4) accidents using **SHAP**, feature-group ablation, and permutation importance.
2. Compares **five machine learning (ML)** and **two deep learning (DL)** models for four-class severity prediction under consistent settings, across five random seeds.
3. Evaluates the effect of **class-specific threshold tuning** on Macro-F1 and the Precision–Recall trade-off.

### Models

| Family | Models |
|---|---|
| ML (5) | Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost |
| DL (2) | MLP with embeddings, TabNet |

### Key findings

- **Geographic** (51.6% of grouped mean |SHAP|) and **Time** (32.0%, driven mainly by `Year`) are the most influential feature groups for high-severity prediction. These are predictive associations, not causal effects.
- **Random Forest** achieved the highest Macro-F1 (**0.6295 ± 0.0010**) and ranked first under all three class-weight settings tested.
- **MLP with embeddings** achieved the highest Macro-Recall (**0.6088 ± 0.0076**) and is the best deep model. Its Macro-F1 gap to Random Forest is not statistically significant (corrected resampled paired t-test, p = 0.109).
- **Severity 4** remains difficult for every model; TabNet obtained the highest F1 for this class (0.2950).
- **Threshold tuning** slightly improves Macro-F1 and Accuracy for all models, but most models gain Macro-Precision while losing Macro-Recall.

## Repository Structure

```
.
├── Pre_Processing/
│   ├── CleanData.ipynb            # Data cleaning
│   └── Pre_processing.ipynb       # Feature engineering, encoding, scaling
├── RQ1/
│   ├── RQ1_LightGBM.ipynb         # Binary LightGBM + TreeSHAP (Severity 1–2 vs. 3–4)
│   └── AblationRQ1.ipynb          # Feature-group ablation and robustness checks
├── training_code/
│   ├── 2DL_CW0.ipynb              # DL models, no class weighting
│   ├── 2DL_CWInfinite.ipynb       # DL models, uncapped inverse-frequency weights
│   ├── 2dl_2.5.ipynb              # DL models, weights capped at 2.5 (main setting)
│   ├── 5ML_Classweight0.ipynb     # ML models, no class weighting
│   ├── 5ML_ClassweightInf.ipynb   # ML models, uncapped inverse-frequency weights
│   └── 5ml_2.5.ipynb              # ML models, weights capped at 2.5 (main setting)
└── Dataset.docx                   # Description of the dataset
```

## Dataset

- **Source:** [US Accidents (2016–2023) on Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
- **Size:** about 7.7 million records, 46 columns, 49 US states, February 2016 – March 2023
- **Target:** `Severity` (1–4)
- **Feature groups:** geographic, time, infrastructure, weather, lighting

After cleaning, the class distribution is heavily imbalanced:

| Severity | Records |
|---|---|
| 1 | 65,311 |
| 2 | 5,924,585 |
| 3 | 1,254,891 |
| 4 | 192,160 |

Please download the raw data from Kaggle yourself; it is not included in this repository. See [`Dataset.docx`](./Dataset.docx) for more details.

## Methodology

**Preprocessing**
- Removed identifier and descriptive columns, columns with more than 20% missing values, and near-constant columns.
- Removed `End_Time` and `Distance` to prevent data leakage.
- Decomposed `Start_Time` into Year, Month, Hour, DayOfWeek, and Season; added sine/cosine encodings for the cyclical features (Hour, DayOfWeek, Month); added the weather-observation time offset in minutes.
- Categorical features with more than 50 distinct values were frequency-encoded; the rest used multiclass target encoding with five-fold cross-fitting. CatBoost handled categoricals natively; deep models used trainable embeddings.
- All preprocessing parameters were fitted on the training set only.

**Class imbalance**
- Class weights were computed from inverse training-set class frequencies and capped so that the largest weight is at most 2.5 times the smallest. Results for no weighting (`CW0`) and uncapped weights (`CWInfinite`) are reported as a sensitivity analysis.

**Evaluation protocol**
- Repeated stratified holdout with five random seeds; train / validation / test split of 64 : 16 : 20.
- Metrics: Accuracy, Macro-Precision, Macro-Recall, Macro-F1 (mean ± std over five runs).
- Statistical comparison: corrected resampled paired t-test (Nadeau & Bengio) between the best ML and DL models.

**Threshold tuning**
- One threshold per class, fitted on the validation set by maximizing the one-vs-rest F1 score (β = 1), separately for each model and seed.
- At prediction time, if exactly one class passes its threshold it is selected; if several pass, the one with the highest original probability is selected; if none passes, Severity 2 is selected.
- Thresholds are frozen before being applied to the test set.

**SHAP-based interpretation (RQ1)**
- A separate binary LightGBM model (Severity 1–2 vs. 3–4, 34 features in five groups) is explained with TreeSHAP on a random sample of 200,000 instances.
- Robustness is checked with group-wise ablation, permutation importance, SHAP-ranking stability across seeds, and temporal validation.

## Results

### Test performance after threshold tuning (mean ± std over five seeds)

| Family | Model | Accuracy | Macro-P | Macro-R | Macro-F1 |
|---|---|---|---|---|---|
| ML | Logistic Regression | 0.7821 ± 0.0004 | 0.5016 ± 0.0058 | 0.3674 ± 0.0009 | 0.3630 ± 0.0016 |
| ML | Random Forest | **0.8771 ± 0.0003** | **0.7117 ± 0.0020** | 0.6004 ± 0.0012 | **0.6295 ± 0.0010** |
| ML | XGBoost | 0.8517 ± 0.0012 | 0.6415 ± 0.0030 | 0.5877 ± 0.0024 | 0.6021 ± 0.0007 |
| ML | LightGBM | 0.8413 ± 0.0004 | 0.6195 ± 0.0028 | 0.5660 ± 0.0028 | 0.5786 ± 0.0018 |
| ML | CatBoost | 0.8633 ± 0.0006 | 0.6557 ± 0.0006 | 0.5767 ± 0.0012 | 0.5998 ± 0.0006 |
| DL | MLP with embeddings | 0.8672 ± 0.0035 | 0.6451 ± 0.0057 | **0.6088 ± 0.0076** | 0.6189 ± 0.0068 |
| DL | TabNet | 0.8626 ± 0.0045 | 0.6261 ± 0.0049 | 0.6027 ± 0.0080 | 0.6097 ± 0.0057 |

### Per-class F1 and Severity 4 recall (S4-R) after threshold tuning

| Model | F1 S1 | F1 S2 | F1 S3 | F1 S4 | S4-R |
|---|---|---|---|---|---|
| Logistic Regression | 0.0538 | 0.8676 | 0.4981 | 0.0324 | 0.0170 |
| Random Forest | **0.5956** | **0.9270** | **0.7231** | 0.2723 | 0.1801 |
| XGBoost | 0.5592 | 0.9110 | 0.6681 | 0.2701 | 0.1967 |
| LightGBM | 0.5215 | 0.9044 | 0.6440 | 0.2445 | 0.1731 |
| CatBoost | 0.5087 | 0.9186 | 0.7001 | 0.2719 | 0.1963 |
| MLP with embeddings | 0.5555 | 0.9209 | 0.7112 | 0.2881 | 0.2228 |
| TabNet | 0.5207 | 0.9178 | 0.7053 | **0.2950** | **0.2438** |

### Effect of threshold tuning (After − Before, mean over five seeds)

| Model | Accuracy | Macro-P | Macro-R | Macro-F1 |
|---|---|---|---|---|
| Logistic Regression | +0.0001 | +0.0003 | +0.0003 | +0.0010 |
| Random Forest | +0.0007 | −0.0062 | +0.0028 | +0.0048 |
| XGBoost | +0.0086 | +0.0039 | −0.0064 | +0.0055 |
| LightGBM | +0.0095 | +0.0041 | −0.0059 | +0.0061 |
| CatBoost | +0.0089 | +0.0030 | −0.0034 | +0.0073 |
| MLP with embeddings | +0.0098 | +0.0080 | −0.0052 | +0.0075 |
| TabNet | +0.0090 | +0.0114 | −0.0099 | +0.0053 |

### Sensitivity to the class-weight cap (Macro-F1 after threshold tuning)

| Model | No weighting (0) | Cap 2.5 | Uncapped (∞) |
|---|---|---|---|
| Logistic Regression | 0.3069 | 0.3630 | 0.4169 |
| Random Forest | 0.6040 | 0.6295 | 0.6272 |
| XGBoost | 0.5626 | 0.6021 | 0.5862 |
| LightGBM | 0.5367 | 0.5786 | 0.5888 |
| CatBoost | 0.5473 | 0.5998 | 0.5901 |
| MLP with embeddings | 0.5831 | 0.6189 | 0.5907 |
| TabNet | 0.5773 | 0.6097 | 0.5771 |

### Grouped SHAP importance for high-severity prediction

| Feature group | Mean \|SHAP\| | Share |
|---|---|---|
| Geographic | 1.259 | 51.6% |
| Time | 0.781 | 32.0% |
| Infrastructure | 0.326 | 13.4% |
| Weather | 0.046 | 1.9% |
| Lighting | 0.027 | 1.1% |

## Getting Started

### 1. Clone the repository

```bash
git clone https://anonymous.4open.science/r/US_Accident_Severity_Prediction-D721/
cd US_Accident_Severity_Prediction-D721
```

(Alternatively, download the repository as a ZIP from the anonymous link.)

### 2. Install dependencies

Python 3.8 or later is recommended.

```bash
pip install numpy pandas scikit-learn xgboost lightgbm catboost shap \
            torch pytorch-tabnet matplotlib scipy jupyter
```

### 3. Download the data

Download `US_Accidents_March23.csv` from [Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) and place it where the notebooks expect it (check the file path at the top of `CleanData.ipynb`).

### 4. Run the notebooks in order

1. **Preprocessing:** `Pre_Processing/CleanData.ipynb` → `Pre_Processing/Pre_processing.ipynb`
2. **Four-class training and threshold tuning:** notebooks in `training_code/` (the `2.5` notebooks give the main results; `CW0` and `CWInfinite` give the sensitivity analysis)
3. **SHAP interpretation:** `RQ1/RQ1_LightGBM.ipynb` → `RQ1/AblationRQ1.ipynb`

### Compute environment used in the paper

- ML experiments: Kaggle (2× NVIDIA Tesla T4)
- DL experiments: Vast.ai (NVIDIA RTX A6000)
- Binary LightGBM and SHAP analysis: Google Colab

## Limitations

- Random splits over a geographically diverse dataset may inflate performance through location-specific patterns; results come from a single dataset.
- Hyperparameters were fixed to common settings and not tuned, so the comparison contrasts model families at a comparable, untuned budget.
- The same validation set was used for early stopping and for threshold tuning, and the class-weight cap was chosen with reference to test results.
- SHAP is computed on a separate binary model dominated by Severity 3, and reflects associations rather than causality; `Year` may partly reflect changes in data sources.
- The five seeds share the same dataset and are not fully independent.

## Citation

The paper is currently under double-blind review. Citation details will be added after the review process.

## Acknowledgment

The US Accidents dataset was created by Sobhan Moosavi et al.:

- S. Moosavi, M. H. Samavatian, S. Parthasarathy, and R. Ramnath, "A Countrywide Traffic Accident Dataset," 2019. [arXiv:1906.05409](https://arxiv.org/abs/1906.05409)
- S. Moosavi, M. H. Samavatian, S. Parthasarathy, R. Teodorescu, and R. Ramnath, "Accident Risk Prediction based on Heterogeneous Sparse Data: New Dataset and Insights," in *Proceedings of the 27th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems*, 2019, pp. 33–42.
