# 🚨 Smart Crime Hotspot Detection — Phase 2
### ML + Deep Learning Pipeline for Predicting Crime Risk Levels in Tamil Nadu

---

## 📌 Problem Statement

Crime hotspot detection is a critical challenge for law enforcement agencies. Identifying high-risk zones early enables efficient resource allocation, preventive policing, and better public safety outcomes. This project builds a supervised multi-class classification system that predicts the **Hotspot Level** (Low / Medium / High) of a given location using historical crime data, geographic features, socioeconomic indicators, and temporal patterns.

The core challenge is a severely **imbalanced dataset** (Low: ~72%, Medium: ~25%, High: ~3%) — a model that simply predicts "Low" achieves 72% accuracy but fails entirely at detecting dangerous High-risk zones. The pipeline is designed specifically to address this.

---

## 🗂️ Dataset Details

| Property | Details |
|---|---|
| **Name** | Tamil Nadu Crime Dataset v2 |
| **File** | `tamilnadu_crime_dataset_v2.csv` |
| **Source** | Custom / synthetic crime records for Tamil Nadu districts |
| **Records** | 1,200 rows |
| **Target Variable** | `Hotspot_Level` — categorical: Low, Medium, High |

### Key Features

| Feature | Description |
|---|---|
| `District`, `Police_Station` | Geographic identifiers |
| `Latitude`, `Longitude` | Spatial coordinates |
| `Total_Crimes`, `Crime_Severity_Index` | Crime volume and severity metrics |
| `Hotspot_Score` | Pre-computed composite risk score |
| `Population_Density`, `Avg_Income` | Socioeconomic context |
| `CCTV_Density`, `Unemployment_Rate` | Infrastructure and economic indicators |
| `Date_of_Crime`, `Date_Reported`, `Hour` | Temporal features |
| `Area_Type`, `Season`, `Day_of_Week` | Contextual categorical features |

### Engineered Features (Derived)

| Feature | Formula |
|---|---|
| `Report_Delay` | Days between crime occurrence and reporting |
| `Crime_Rate` | `Total_Crimes / Population_Density` |
| `Crime_Per_Hour` | `Total_Crimes / (Hour + 1)` |

### Class Distribution

```
Low    : ~865 samples  (72%)   ← majority class
Medium : ~301 samples  (25%)
High   :  ~34 samples  ( 3%)   ← critical minority
```

---

## 🏗️ Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     DATA INGESTION & EXPLORATION                        │
│           tamilnadu_crime_dataset_v2.csv  →  1200 records               │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       FEATURE ENGINEERING                               │
│   Date parsing → Report_Delay, Crime_Rate, Crime_Per_Hour               │
│   LabelEncoding (District, Police_Station)                              │
│   OneHotEncoding (Area_Type, Season, Day_of_Week)                       │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       OUTLIER HANDLING                                  │
│        Winsorization (5th–95th percentile) on 7 numeric columns         │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    CLASS IMBALANCE HANDLING                             │
│       compute_class_weight('balanced') → High class weight ~8×          │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       FEATURE SELECTION                                 │
│   Correlation filter (threshold=0.05) + Multicollinearity check (>0.85) │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              TRAIN / TEST SPLIT  (80% / 20%, stratified)                │
│                   StandardScaler (fit on train only)                    │
└──────────────────┬───────────────────────────────┬───────────────────────┘
                   │                               │
       ┌───────────┴───────────┐       ┌───────────┴───────────┐
       │  UNSUPERVISED PATH    │       │  SUPERVISED PATH       │
       │  K-Means Clustering   │       │                        │
       │  K=3 (Elbow+Sil)      │       │  ┌─ ML Models ───────┐ │
       │  → Hotspot Map        │       │  │  XGBoost          │ │
       └───────────────────────┘       │  │  LightGBM         │ │
                                       │  │  CatBoost         │ │
                                       │  │  Extra Trees      │ │
                                       │  │  HistGradBoost    │ │
                                       │  │  Voting Ensemble  │ │
                                       │  └───────────────────┘ │
                                       │                        │
                                       │  ┌─ Tuning ──────────┐ │
                                       │  │ RandomizedSearchCV│ │
                                       │  │ (40 iter × 5-fold)│ │
                                       │  └───────────────────┘ │
                                       │                        │
                                       │  ┌─ Stacking ────────┐ │
                                       │  │ XGB+LGBM+CAT+ET   │ │
                                       │  │ → LR Meta-Learner │ │
                                       │  └───────────────────┘ │
                                       │                        │
                                       │  ┌─ Deep Learning ───┐ │
                                       │  │ TabNet Classifier │ │
                                       │  │ (sparsemax attn)  │ │
                                       │  └───────────────────┘ │
                                       └───────────┬────────────┘
                                                   │
                                                   ▼
                                    ┌──────────────────────────┐
                                    │   FINAL LEADERBOARD      │
                                    │  F1 Macro (all models)   │
                                    └──────────────────────────┘
```

---

## 🤖 Model Details

### Machine Learning Models

| Model | Key Configuration |
|---|---|
| **XGBoost** | n_estimators=300, lr=0.05, max_depth=6, sample_weight balanced |
| **LightGBM** | n_estimators=300, lr=0.05, class_weight='balanced', num_leaves=63 |
| **CatBoost** | iterations=300, lr=0.05, auto_class_weights='Balanced' |
| **Extra Trees** | n_estimators=300, class_weight='balanced', max_features='sqrt' |
| **HistGradientBoosting** | max_iter=300, lr=0.05, class_weight='balanced' |
| **Hard Voting Ensemble** | Combines XGBoost + LightGBM + CatBoost (hard voting) |
| **Stacking Classifier** | Base: XGB+LGBM+CatBoost+ExtraTrees → Meta: Logistic Regression (5-fold OOF) |

### Hyperparameter Tuning

Both **XGBoost** and **LightGBM** are tuned via `RandomizedSearchCV` with:
- 40 random iterations × 5-fold Stratified CV
- Scoring: `f1_macro`
- Search space covers: n_estimators, learning_rate, max_depth, subsample, colsample_bytree, regularisation (alpha/lambda), min_child_weight

### Deep Learning Model

| Property | Details |
|---|---|
| **Architecture** | TabNet (Arik & Pfister, Google Brain 2021) |
| **n_d / n_a** | 32 / 32 (decision + attention embedding width) |
| **n_steps** | 5 sequential attention steps |
| **Attention type** | Sparsemax (true sparse, interpretable) |
| **λ_sparse** | 1e-3 (sparsity regularisation) |
| **Optimiser** | Adam (lr=2e-2, weight_decay=1e-5) + StepLR scheduler |
| **Loss** | CrossEntropyLoss with balanced class weights tensor |
| **Max epochs** | 200 with patience=30 (early stopping) |
| **Batch size** | 256 (virtual batch size: 128) |

TabNet's sparsemax attention masks provide native feature importance at both **global** (aggregated across all samples) and **per-step** levels, offering built-in explainability without SHAP.

### Unsupervised Model

**K-Means Clustering** (K=3) on `[Latitude, Longitude, Total_Crimes, Crime_Severity_Index]` — used to generate geographic hotspot maps and validate that spatial clusters align with labelled hotspot levels.

---

## ▶️ Steps to Run the Project

### Prerequisites

Ensure Python 3.8+ is installed. Install all dependencies (see next section), then follow these steps:

**Step 1 — Clone or download the repository**
```bash
git clone <repo-url>
cd crime-hotspot-detection
```

**Step 2 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 3 — Place the dataset**

Put `tamilnadu_crime_dataset_v2.csv` in the `/content/` directory (if running on Google Colab) or update the path in cell 2:
```python
df = pd.read_csv('path/to/tamilnadu_crime_dataset_v2.csv')
```

**Step 4 — Open the notebook**
```bash
jupyter notebook ML_crime_hotspot_detection_phase2_WITH_DEEP_LEARNING.ipynb
```
Or upload directly to **Google Colab** (recommended for GPU support with TabNet).

**Step 5 — Run all cells in order**

The notebook is self-contained and sequential. Run cells from top to bottom:

| Section | What it does |
|---|---|
| §1–2 | Import libraries, load dataset |
| §3–4 | Feature engineering, outlier handling |
| §5–6 | Target encoding, class imbalance handling, feature selection |
| §7 | Train/test split and scaling |
| §8 | K-Means unsupervised clustering + geographic map |
| §9–11 | Advanced ML baseline training and visualisation |
| §12 | Hyperparameter tuning (XGBoost + LightGBM) |
| §13–14 | Stacking ensemble + meta-learner analysis |
| §15–16 | Cross-validation + final model leaderboard |
| §17 | Insights and observations |
| §18 | Full feature correlation heatmap |
| §19 | TabNet deep learning model + comprehensive comparison |

> ⚠️ **Note:** Section §19 (TabNet) installs `pytorch-tabnet` automatically via `pip`. On Colab, GPU runtime is recommended (`Runtime → Change runtime type → T4 GPU`).

---

## 📦 Required Dependencies / Libraries

```
# requirements.txt

# Core data science
pandas>=1.5.0
numpy>=1.23.0
scipy>=1.9.0
matplotlib>=3.6.0
seaborn>=0.12.0

# Machine learning
scikit-learn>=1.2.0

# Advanced gradient boosting
xgboost>=1.7.0
lightgbm>=3.3.0
catboost>=1.1.0

# Deep learning (tabular)
pytorch-tabnet>=4.0.0
torch>=1.12.0

# Jupyter
notebook>=6.5.0
ipykernel>=6.0.0
```

Install all at once:
```bash
pip install pandas numpy scipy matplotlib seaborn scikit-learn xgboost lightgbm catboost pytorch-tabnet torch notebook
```

---

## 📊 Sample Output / Results

### Model Performance Leaderboard (Macro F1)

| Rank | Model | Accuracy | F1 Score (Macro) | Type |
|---|---|---|---|---|
| 🥇 1 | Stacking Classifier | ~93%+ | Highest | Ensemble |
| 🥈 2 | XGBoost (Tuned) | ~92%+ | High | ML |
| 🥉 3 | LightGBM (Tuned) | ~92%+ | High | ML |
| 4 | TabNet | ~90%+ | High | Deep Learning |
| 5 | CatBoost | ~91%+ | High | ML |
| 6 | Voting Ensemble | ~91%+ | High | Ensemble |

> Exact scores will vary based on random seeds and dataset splits. Run the notebook to see final leaderboard values.

### Key Visualisations Generated

- **Geographic Hotspot Map** — Scatter plot of Tamil Nadu locations coloured by cluster label (Low/Medium/High)
- **Elbow + Silhouette plots** — K selection for K-Means
- **Model comparison bar charts** — Accuracy, Precision, Recall, F1 across all models
- **Confusion matrices** — For every model including TabNet
- **ROC-AUC curves** — One-vs-Rest curves per class per model
- **Feature importance charts** — XGBoost, LightGBM, and TabNet attention-based importances
- **TabNet step-wise attention masks** — Visualises which features each sequential step focuses on
- **Training loss & validation accuracy curves** — TabNet learning history
- **Meta-learner coefficient chart** — Which base learner × class the stacking model trusts most
- **Final comprehensive leaderboard** — Colour-coded by model type (ML / Ensemble / Deep Learning)

---

## 👥 Team Member Details

| Name | Role | Contribution |
|---|---|---|
| *(Team Member 1)* | ML Engineer | Feature engineering, baseline models, hyperparameter tuning |
| *(Team Member 2)* | DL Engineer | TabNet architecture, training pipeline, attention visualisation |
| *(Team Member 3)* | Data Analyst | EDA, outlier handling, class imbalance strategy |
| *(Team Member 4)* | Ensemble Specialist | Stacking, voting ensemble, meta-learner analysis |

> 📝 **Update this section with your actual team member names, roll numbers, and institution details before submission.**

---

## 📁 Repository Structure

```
crime-hotspot-detection/
│
├── ML_crime_hotspot_detection_phase2_WITH_DEEP_LEARNING.ipynb   # Main notebook
├── tamilnadu_crime_dataset_v2.csv                                # Dataset
├── requirements.txt                                              # Dependencies
└── README.md                                                     # This file
```

---

## 🔮 Future Work (Phase 3)

- **SMOTE / ADASYN** oversampling inside cross-validation pipeline
- **Bayesian Optimisation** (Optuna) for deeper hyperparameter search
- **Spatial features v2** — Distance to nearest police station, road density
- **Hybrid ensemble** — TabNet probabilities as meta-features in stacking
- **Streamlit dashboard** — Real-time district-level hotspot prediction interface
- **SHAP explainability** — Per-prediction explanations for police deployment decisions
- **Model export** — `joblib` serialisation for deployment

---

## 📜 References

- Arik, S. Ö., & Pfister, T. (2021). *TabNet: Attentive Interpretable Tabular Learning*. AAAI 2021.
- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD 2016.
- Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS 2017.
- Prokhorenkova, L., et al. (2018). *CatBoost: Unbiased Boosting with Categorical Features*. NeurIPS 2018.
