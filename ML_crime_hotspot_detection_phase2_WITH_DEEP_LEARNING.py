import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from scipy.stats import mstats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

print("All libraries imported successfully.")

df = pd.read_csv('/content/tamilnadu_crime_dataset_v2.csv')
print("Shape:", df.shape)
df.head()

df.info()

print("Missing values per column:")
print(df.isnull().sum())
print("\nTotal missing:", df.isnull().sum().sum())

df_raw = df.copy()

df['Date_of_Crime']  = pd.to_datetime(df['Date_of_Crime'])
df['Date_Reported']  = pd.to_datetime(df['Date_Reported'])
print("Date columns converted.")

# Report_Delay: how many days before the crime was reported
df['Report_Delay'] = (df['Date_Reported'] - df['Date_of_Crime']).dt.days

# Crime_Rate: crimes relative to population density
df['Crime_Rate'] = df['Total_Crimes'] / df['Population_Density']

# Crime_Per_Hour: intensity of crime in that hour slot
df['Crime_Per_Hour'] = df['Total_Crimes'] / (df['Hour'] + 1)

print("Derived features created:")
print(df[['Report_Delay', 'Crime_Rate', 'Crime_Per_Hour']].describe().round(3))

df = df.drop(['Record_ID', 'Date_of_Crime', 'Date_Reported'], axis=1)
print("Dropped: Record_ID, Date_of_Crime, Date_Reported")
print("Remaining columns:", df.shape[1])

print("Before Encoding:")
print(df_raw[['District','Area_Type','Season','Day_of_Week']].head(3))

le = LabelEncoder()
df['District']       = le.fit_transform(df['District'])
df['Police_Station'] = le.fit_transform(df['Police_Station'])

df = pd.get_dummies(df, columns=['Area_Type', 'Season', 'Day_of_Week'], drop_first=True)

print("\nAfter Encoding — shape:", df.shape)
print("Sample District encoding:", df['District'].unique()[:5])

cols = ['Total_Crimes','Crime_Severity_Index','Hotspot_Score',
        'Avg_Income','Population_Density','CCTV_Density','Unemployment_Rate']

plt.figure(figsize=(13, 5))
sns.boxplot(data=df_raw[cols], palette='Set2')
plt.title("Before Outlier Handling — Boxplots")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()

cols_to_fix = ['Total_Crimes', 'Crime_Severity_Index', 'Hotspot_Score',
               'Avg_Income', 'Population_Density', 'CCTV_Density', 'Unemployment_Rate']

for col in cols_to_fix:
    df[col] = mstats.winsorize(df[col], limits=[0.05, 0.05])

print("Winsorization applied to", len(cols_to_fix), "columns")
print("\nRange after cleaning:")
for col in cols_to_fix:
    print(f"  {col}: min={df[col].min():.2f}, max={df[col].max():.2f}")

plt.figure(figsize=(13, 5))
sns.boxplot(data=df[cols_to_fix], palette='Set2')
plt.title("After Outlier Handling — Winsorized")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()

le_target = LabelEncoder()
y = le_target.fit_transform(df['Hotspot_Level'])

print("Target classes:", le_target.classes_)
dist = dict(zip(le_target.classes_,
                [list(y).count(i) for i in range(len(le_target.classes_))]))
print("\nClass distribution:", dist)

# Plot
plt.figure(figsize=(7, 4))
bars = plt.bar(dist.keys(), dist.values(), color=['#e74c3c','#3498db','#2ecc71'], edgecolor='gray', linewidth=0.5)
for bar, val in zip(bars, dist.values()):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             str(val), ha='center', fontsize=11, fontweight='bold')
plt.title("Class Distribution — Hotspot Level (IMBALANCED)", fontsize=13)
plt.ylabel("Count")
plt.tight_layout()
plt.show()

from sklearn.utils.class_weight import compute_class_weight

class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y),
    y=y
)
class_weight_dict = dict(enumerate(class_weights_array))
sample_weights = np.array([class_weight_dict[yi] for yi in y])

print("Computed class weights:")
for cls, w in zip(le_target.classes_, class_weights_array):
    print(f"  {cls}: {w:.3f}")
print("\nHigher weight = penalize misclassification of this class more heavily.")

# Exclude the target and any direct encodings of the target
cols_to_exclude = ['Hotspot_Level']
if 'Hotspot_Level_Encoded' in df.columns:
    cols_to_exclude.append('Hotspot_Level_Encoded')

X_all = df.drop(cols_to_exclude, axis=1)
print("Feature matrix shape:", X_all.shape)
print("Columns:", list(X_all.columns))

# Temporary encoding for correlation calculation
df['_target_enc'] = y

numeric_cols = df.drop(['Hotspot_Level', '_target_enc'], axis=1).select_dtypes(include='number')

# Exclude any residual target-related columns from correlation
for bad_col in ['Hotspot_Level_Encoded', 'Hotspot_Level_enc']:
    if bad_col in numeric_cols.columns:
        numeric_cols = numeric_cols.drop(bad_col, axis=1)

target_corr = (numeric_cols.assign(_target_enc=df['_target_enc'])
               .corr()['_target_enc']
               .drop('_target_enc')
               .abs()
               .sort_values(ascending=False))

THRESHOLD = 0.05
print("=" * 55)
print("FEATURE vs TARGET CORRELATION (absolute value)")
print("=" * 55)
for feat, val in target_corr.items():
    status = "KEEP" if val >= THRESHOLD else "drop"
    print(f"  {status:4s}  {feat:35s}  {val:.4f}")

df = df.drop('_target_enc', axis=1)

# Visualize
plt.figure(figsize=(11, 7))
colors = ['#1565C0' if v >= THRESHOLD else '#CFD8DC' for v in target_corr.values]
plt.barh(target_corr.index, target_corr.values, color=colors)
plt.axvline(x=THRESHOLD, color='red', linestyle='--', linewidth=1.5, label=f'Threshold = {THRESHOLD}')
plt.xlabel("Absolute Correlation with Hotspot_Level")
plt.title("Feature vs Target Correlation — Feature Selection")
plt.legend()
plt.tight_layout()
plt.show()

top_features = target_corr[target_corr >= THRESHOLD].index.tolist()

# Multicollinearity check
MULTI_THRESHOLD = 0.85
feature_corr = numeric_cols[top_features].corr().abs()
upper = feature_corr.where(np.triu(np.ones(feature_corr.shape), k=1).astype(bool))

high_pairs = [(col, row, upper.loc[row, col])
              for col in upper.columns
              for row in upper.index
              if pd.notna(upper.loc[row, col]) and upper.loc[row, col] > MULTI_THRESHOLD]

print("=" * 55)
print(f"MULTICOLLINEARITY CHECK (pairs with corr > {MULTI_THRESHOLD})")
print("=" * 55)
if high_pairs:
    to_drop_multi = []
    for a, b, v in sorted(high_pairs, key=lambda x: -x[2]):
        print(f"  {a}  <->  {b}  :  {v:.4f}  -> drop '{b}'")
        if b not in to_drop_multi:
            to_drop_multi.append(b)
    top_features = [f for f in top_features if f not in to_drop_multi]
    print(f"\n  Dropped due to multicollinearity: {to_drop_multi}")
else:
    print("  No high-correlation pairs found. All features retained.")

print("\n" + "=" * 55)
print("FINAL SELECTED FEATURES")
print("=" * 55)
final_features = [f for f in top_features if f in X_all.columns]
for f in final_features:
    print(f"  +  {f}  (corr={target_corr[f]:.4f})")
print(f"\n  Total: {len(final_features)} features selected")

# Heatmap of selected features
plt.figure(figsize=(9, 6))
sel_cols = numeric_cols[final_features].copy()
sel_cols['Hotspot_Level_enc'] = y
sns.heatmap(sel_cols.corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix — Selected Features vs Target")
plt.tight_layout()
plt.show()

X_selected = df[final_features]

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_selected, y,
    test_size=0.2,
    random_state=42,
    stratify=y   # preserve class proportions in both splits
)

# Fit scaler ONLY on training data, then transform both splits
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)   # fit + transform on train
X_test  = scaler.transform(X_test_raw)        # transform only on test

# Also compute sample weights for train split (for models that use it)
train_sample_weights = np.array([class_weight_dict[yi] for yi in y_train])

print(f"Training set : {X_train.shape[0]} rows, {X_train.shape[1]} features")
print(f"Testing  set : {X_test.shape[0]} rows,  {X_test.shape[1]} features")
print()
print("Class distribution in y_train:")
for cls, cnt in zip(le_target.classes_, np.bincount(y_train)):
    print(f"  {cls}: {cnt}")
print("\nClass distribution in y_test:")
for cls, cnt in zip(le_target.classes_, np.bincount(y_test)):
    print(f"  {cls}: {cnt}")

cluster_features = df[['Latitude', 'Longitude', 'Total_Crimes', 'Crime_Severity_Index']]
print("Clustering features shape:", cluster_features.shape)
print(cluster_features.describe().round(2))

scaler_cluster = StandardScaler()
cluster_scaled = scaler_cluster.fit_transform(cluster_features)
print("Clustering features scaled.")

inertia = []
silhouette_scores = []
K_range = range(2, 10)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(cluster_scaled)
    inertia.append(km.inertia_)
    silhouette_scores.append(silhouette_score(cluster_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].plot(K_range, inertia, marker='o', color='steelblue', linewidth=2)
axes[0].set_title("Elbow Method — Inertia per K")
axes[0].set_xlabel("K (number of clusters)")
axes[0].set_ylabel("Inertia")
axes[0].grid(True, alpha=0.3)

axes[1].plot(K_range, silhouette_scores, marker='s', color='darkorange', linewidth=2)
axes[1].set_title("Silhouette Score per K")
axes[1].set_xlabel("K")
axes[1].set_ylabel("Silhouette Score")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("K  | Inertia   | Silhouette")
print("-" * 38)
for k, inn, sil in zip(K_range, inertia, silhouette_scores):
    print(f"K={k} | {inn:9.1f} | {sil:.4f}")

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(cluster_scaled)
final_score = silhouette_score(cluster_scaled, df['Cluster'])
print(f"K-Means: K=3, Silhouette Score = {final_score:.3f}")
print("\nCluster counts:")
print(df['Cluster'].value_counts())

avg_crimes = df.groupby('Cluster')['Total_Crimes'].mean().sort_values()
print("Average Total_Crimes per cluster:")
print(avg_crimes)
cluster_order = avg_crimes.index.tolist()

df['Cluster_Label'] = df['Cluster'].map({
    cluster_order[0]: 'Low',
    cluster_order[1]: 'Medium',
    cluster_order[2]: 'High'
})
print("\nCluster labels assigned:")
print(df['Cluster_Label'].value_counts())

color_map = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}

plt.figure(figsize=(10, 8))
for label, color in color_map.items():
    subset = df[df['Cluster_Label'] == label]
    plt.scatter(subset['Longitude'], subset['Latitude'],
                c=color, label=f"{label} ({len(subset)})", alpha=0.6, s=40)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Tamil Nadu Crime Hotspot Clusters (K-Means, K=3)")
plt.legend(title="Hotspot Level")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Average Total Crimes per cluster label:")
print(df.groupby('Cluster_Label')['Total_Crimes'].mean().round(2))

print("\nCross-tabulation: Ground-truth Hotspot_Level vs K-Means Cluster_Label")
ct = pd.crosstab(df['Hotspot_Level'], df['Cluster_Label'])
print(ct)

print("\nInterpretation:")
print("  - 'High' ground-truth records are cleanly captured by the High cluster.")
print("  - 'Low' and 'Medium' overlap considerably in geographic/crime space,")
print("    suggesting these labels are based on factors beyond just location and")
print("    total crimes — e.g., crime type mix, socioeconomic context.")
print("  - This supports using supervised classification (which leverages all")
print("    features) rather than relying solely on spatial clustering.")

# ── Advanced model imports ────────────────────────────────────────────────────
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import (ExtraTreesClassifier,
                               HistGradientBoostingClassifier,
                               VotingClassifier)
from sklearn.utils.class_weight import compute_sample_weight

print("Advanced model libraries imported successfully.")
print("XGBoost, LightGBM, CatBoost, ExtraTrees, HistGradientBoosting, VotingClassifier — ready.")

# ── Compute class weights for XGBoost sample_weight approach ─────────────────
n_classes = len(np.unique(y_train))
class_weights_adv = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict_adv = dict(enumerate(class_weights_adv))
train_sw_adv = compute_sample_weight('balanced', y_train)

# XGBoost scale_pos_weight works only for binary; for multiclass use sample_weight
print("Class weights (balanced):")
for cls, w in zip(le_target.classes_, class_weights_adv):
    print(f"  {cls}: {w:.4f}")

# ── Define advanced baseline models ──────────────────────────────────────────
adv_models = {
    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        verbosity=0
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=63,
        class_weight='balanced',
        random_state=42,
        verbose=-1
    ),
    "CatBoost": CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        auto_class_weights='Balanced',
        random_seed=42,
        verbose=0
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=300,
        class_weight='balanced',
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=6,
        class_weight='balanced',
        random_state=42
    ),
}

print("Advanced models defined:")
for name in adv_models:
    print(f"  ✓  {name}")

# ── Train advanced models & collect metrics ───────────────────────────────────
adv_results = {}

for name, model in adv_models.items():
    # XGBoost does not accept class_weight parameter — use sample_weight at fit
    if name == "XGBoost":
        model.fit(X_train, y_train, sample_weight=train_sw_adv)
    else:
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='macro', zero_division=0)

    adv_results[name] = {
        'Accuracy':          round(acc,  4),
        'Precision (Macro)': round(prec, 4),
        'Recall (Macro)':    round(rec,  4),
        'F1 (Macro)':        round(f1,   4),
        'y_pred':            y_pred,
        'model':             model,
    }

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(classification_report(y_test, y_pred,
                                 target_names=le_target.classes_, zero_division=0))

print("All advanced baseline models trained.")

# ── Hard Voting Ensemble from top-3 advanced models ──────────────────────────
voting_clf = VotingClassifier(
    estimators=[
        ('xgb', adv_models['XGBoost']),
        ('lgbm', adv_models['LightGBM']),
        ('cat',  adv_models['CatBoost']),
    ],
    voting='hard',
    n_jobs=-1
)

# Refit on training data (VotingClassifier clones internally)
voting_clf.fit(X_train, y_train, xgb__sample_weight=train_sw_adv)
y_pred_vote = voting_clf.predict(X_test)

adv_results['Voting Ensemble'] = {
    'Accuracy':          round(accuracy_score(y_test, y_pred_vote), 4),
    'Precision (Macro)': round(precision_score(y_test, y_pred_vote, average='macro', zero_division=0), 4),
    'Recall (Macro)':    round(recall_score(y_test, y_pred_vote, average='macro', zero_division=0), 4),
    'F1 (Macro)':        round(f1_score(y_test, y_pred_vote, average='macro', zero_division=0), 4),
    'y_pred':            y_pred_vote,
    'model':             voting_clf,
}

print("Hard Voting Ensemble (XGB + LGBM + CatBoost):")
print(classification_report(y_test, y_pred_vote,
                             target_names=le_target.classes_, zero_division=0))

adv_results_df = pd.DataFrame([
    {
        'Model':             name,
        'Accuracy':          f"{v['Accuracy']*100:.2f}%",
        'Precision (Macro)': f"{v['Precision (Macro)']*100:.2f}%",
        'Recall (Macro)':    f"{v['Recall (Macro)']*100:.2f}%",
        'F1 Score (Macro)':  f"{v['F1 (Macro)']*100:.2f}%",
    }
    for name, v in adv_results.items()
])

print("\n" + "="*80)
print("ADVANCED MODEL PERFORMANCE COMPARISON TABLE")
print("="*80)
print(adv_results_df.to_string(index=False))
print("="*80)
print("* All metrics macro-averaged  |  class_weight='balanced' applied")

metrics_list = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1 (Macro)']
model_names_adv = list(adv_results.keys())
x = np.arange(len(model_names_adv))
width = 0.18

fig, ax = plt.subplots(figsize=(16, 6))
colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

for i, (metric, color) in enumerate(zip(metrics_list, colors)):
    vals = [adv_results[m][metric] for m in model_names_adv]
    ax.bar(x + i*width, vals, width, label=metric, color=color, alpha=0.85)

ax.set_xlabel("Model", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Advanced Baseline Model Comparison — All Metrics (Macro-Averaged)", fontsize=13)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(model_names_adv, rotation=15, ha='right')
ax.set_ylim(0, 1.08)
ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
ax.legend(loc='lower right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

n_models = len(adv_results)
ncols = 3
nrows = (n_models + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(17, 5*nrows))
axes = axes.flatten()

for idx, (name, res) in enumerate(adv_results.items()):
    y_pred = res['y_pred']
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le_target.classes_,
                yticklabels=le_target.classes_,
                ax=axes[idx], linewidths=0.5)
    f1_val = res['F1 (Macro)']
    axes[idx].set_title(f"{name}\nF1={f1_val:.3f}", fontsize=10)
    axes[idx].set_xlabel("Predicted")
    axes[idx].set_ylabel("True")

for j in range(idx+1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Confusion Matrices — Advanced Baseline Models", fontsize=14, y=1.01)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

for ax, model_name in zip(axes, ['XGBoost', 'LightGBM']):
    model = adv_results[model_name]['model']
    importances = model.feature_importances_
    fi_df = pd.DataFrame({'Feature': final_features, 'Importance': importances})
    fi_df = fi_df.sort_values('Importance', ascending=True).tail(15)
    ax.barh(fi_df['Feature'], fi_df['Importance'], color='steelblue', edgecolor='gray', lw=0.5)
    ax.set_title(f"Feature Importances — {model_name}")
    ax.set_xlabel("Importance")

plt.suptitle("Feature Importances: XGBoost vs LightGBM (Top 15)", fontsize=13)
plt.tight_layout()
plt.show()

from sklearn.preprocessing import label_binarize

y_bin = label_binarize(y_test, classes=[0, 1, 2])
class_names = le_target.classes_
colors_roc = ['#e74c3c', '#3498db', '#2ecc71']

prob_adv = {n: r['model'] for n, r in adv_results.items()
            if hasattr(r['model'], 'predict_proba') and n != 'Voting Ensemble'}

ncols = 3
nrows = (len(prob_adv) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
axes = axes.flatten()

for ax, (name, model) in zip(axes, prob_adv.items()):
    y_proba = model.predict_proba(X_test)
    for i, (cls, col) in enumerate(zip(class_names, colors_roc)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        auc = roc_auc_score(y_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, color=col, linewidth=2, label=f"{cls} (AUC={auc:.2f})")
    ax.plot([0,1],[0,1],'k--', linewidth=0.8, alpha=0.5)
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

for j in range(len(prob_adv), len(axes)):
    axes[j].set_visible(False)

plt.suptitle("ROC-AUC Curves — Advanced Models (One-vs-Rest)", fontsize=13, y=1.02)
plt.tight_layout()
plt.show()

from sklearn.model_selection import RandomizedSearchCV

cv_tune = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── XGBoost Hyperparameter Space ─────────────────────────────────────────────
xgb_param_dist = {
    'n_estimators':      [200, 300, 400, 500],
    'max_depth':         [3, 4, 5, 6, 7],
    'learning_rate':     [0.01, 0.03, 0.05, 0.1],
    'subsample':         [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree':  [0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_weight':  [1, 3, 5, 7],
    'gamma':             [0, 0.1, 0.2, 0.3],
    'reg_alpha':         [0, 0.01, 0.1, 1],
    'reg_lambda':        [0.5, 1, 1.5, 2],
}

xgb_base = XGBClassifier(
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42,
    verbosity=0
)

xgb_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=xgb_param_dist,
    n_iter=40,
    scoring='f1_macro',
    cv=cv_tune,
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("Starting XGBoost RandomizedSearchCV (40 iterations × 5-fold)...")
xgb_search.fit(X_train, y_train, sample_weight=train_sw_adv)
print(f"\nBest XGBoost params: {xgb_search.best_params_}")
print(f"Best CV F1 Macro   : {xgb_search.best_score_:.4f}")

# ── LightGBM Hyperparameter Space ────────────────────────────────────────────
lgbm_param_dist = {
    'n_estimators':    [200, 300, 400, 500],
    'max_depth':       [3, 4, 5, 6, 7, -1],
    'learning_rate':   [0.01, 0.03, 0.05, 0.1],
    'num_leaves':      [31, 63, 127, 255],
    'subsample':       [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree':[0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_samples':[10, 20, 30, 50],
    'reg_alpha':       [0, 0.01, 0.1, 1],
    'reg_lambda':      [0, 0.01, 0.1, 1],
}

lgbm_base = LGBMClassifier(class_weight='balanced', random_state=42, verbose=-1)

lgbm_search = RandomizedSearchCV(
    estimator=lgbm_base,
    param_distributions=lgbm_param_dist,
    n_iter=40,
    scoring='f1_macro',
    cv=cv_tune,
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("Starting LightGBM RandomizedSearchCV (40 iterations × 5-fold)...")
lgbm_search.fit(X_train, y_train)
print(f"\nBest LightGBM params: {lgbm_search.best_params_}")
print(f"Best CV F1 Macro    : {lgbm_search.best_score_:.4f}")

# ── Evaluate tuned models on test set ────────────────────────────────────────
tuned_models = {
    'XGBoost (Tuned)':  xgb_search.best_estimator_,
    'LightGBM (Tuned)': lgbm_search.best_estimator_,
}

tuned_results = {}
for name, model in tuned_models.items():
    y_pred = model.predict(X_test)
    tuned_results[name] = {
        'Accuracy':          round(accuracy_score(y_test, y_pred), 4),
        'Precision (Macro)': round(precision_score(y_test, y_pred, average='macro', zero_division=0), 4),
        'Recall (Macro)':    round(recall_score(y_test, y_pred, average='macro', zero_division=0), 4),
        'F1 (Macro)':        round(f1_score(y_test, y_pred, average='macro', zero_division=0), 4),
        'y_pred':            y_pred,
        'model':             model,
    }
    print(f"\n{'='*55}\n  {name}\n{'='*55}")
    print(classification_report(y_test, y_pred, target_names=le_target.classes_, zero_division=0))

# ── Compare base vs tuned ─────────────────────────────────────────────────────
print("\n" + "="*65)
print("BASE  vs  TUNED — F1 Macro Comparison")
print("="*65)
pairs = [('XGBoost', 'XGBoost (Tuned)'), ('LightGBM', 'LightGBM (Tuned)')]
for base_name, tuned_name in pairs:
    base_f1  = adv_results[base_name]['F1 (Macro)']
    tuned_f1 = tuned_results[tuned_name]['F1 (Macro)']
    delta    = tuned_f1 - base_f1
    print(f"  {base_name:12s}: base={base_f1:.4f}  tuned={tuned_f1:.4f}  Δ={delta:+.4f}")

from sklearn.ensemble import StackingClassifier

# ── Define stacking base learners ─────────────────────────────────────────────
stacking_base_learners = [
    ('xgb_tuned',  xgb_search.best_estimator_),
    ('lgbm_tuned', lgbm_search.best_estimator_),
    ('catboost',   adv_models['CatBoost']),
    ('extratrees', adv_models['Extra Trees']),
]

# ── Meta-learner ──────────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression as LR_meta

meta_learner = LR_meta(
    max_iter=1000,
    class_weight='balanced',
    C=1.0,
    random_state=42
)

# ── Stacking classifier ───────────────────────────────────────────────────────
stacking_clf = StackingClassifier(
    estimators=stacking_base_learners,
    final_estimator=meta_learner,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    stack_method='predict_proba',   # use probability outputs as meta-features
    passthrough=False,              # only meta-features go to meta-learner
    n_jobs=-1,
    verbose=0
)

print("Stacking Classifier defined:")
print("  Base learners: XGBoost(tuned), LightGBM(tuned), CatBoost, ExtraTrees")
print("  Meta-learner : Logistic Regression (balanced, C=1.0)")
print("  CV strategy  : 5-fold Stratified (OOF predictions, leak-free)")
print("  Stack method : predict_proba → 12 meta-features (4 models × 3 classes)")

# ── Train stacking classifier ─────────────────────────────────────────────────
print("Training Stacking Classifier (this runs 5-fold CV × 4 base learners)...")
stacking_clf.fit(X_train, y_train)
print("Stacking training complete.\n")

y_pred_stack = stacking_clf.predict(X_test)

stack_acc  = accuracy_score(y_test, y_pred_stack)
stack_prec = precision_score(y_test, y_pred_stack, average='macro', zero_division=0)
stack_rec  = recall_score(y_test, y_pred_stack, average='macro', zero_division=0)
stack_f1   = f1_score(y_test, y_pred_stack, average='macro', zero_division=0)

print("="*60)
print("  STACKING CLASSIFIER — TEST SET PERFORMANCE")
print("="*60)
print(f"  Accuracy          : {stack_acc:.4f} ({stack_acc*100:.2f}%)")
print(f"  Precision (Macro) : {stack_prec:.4f} ({stack_prec*100:.2f}%)")
print(f"  Recall (Macro)    : {stack_rec:.4f} ({stack_rec*100:.2f}%)")
print(f"  F1 Score (Macro)  : {stack_f1:.4f} ({stack_f1*100:.2f}%)")
print()
print(classification_report(y_test, y_pred_stack,
                             target_names=le_target.classes_, zero_division=0))

# ── Extract meta-learner coefficients ────────────────────────────────────────
meta_model = stacking_clf.final_estimator_

# Meta-feature names: model_name × class_name
meta_feature_names = []
for est_name, _ in stacking_base_learners:
    for cls in le_target.classes_:
        meta_feature_names.append(f"{est_name}_{cls}")

coef_df = pd.DataFrame(
    meta_model.coef_,
    index=[f"Predicts_{c}" for c in le_target.classes_],
    columns=meta_feature_names
).T

print("Meta-Learner Coefficients (Logistic Regression):")
print("Positive = strong signal for that output class | Negative = counter-signal")
print()
print(coef_df.round(4).to_string())

# ── Visualize meta-learner coefficients ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
coef_df.plot(kind='barh', ax=ax, colormap='RdYlGn', edgecolor='gray', linewidth=0.4)
ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_title("Meta-Learner Coefficients: Which Base Learner × Class Gets Trusted Most",
             fontsize=12)
ax.set_xlabel("Logistic Regression Coefficient")
ax.legend(title="Output Class", bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()
plt.show()

# ── OOF prediction confidence analysis ────────────────────────────────────────
# Examine stacking probabilities on test set
y_proba_stack = stacking_clf.predict_proba(X_test)
confidence     = y_proba_stack.max(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].hist(confidence, bins=30, color='steelblue', edgecolor='white', alpha=0.85)
axes[0].set_title("Stacking — Prediction Confidence Distribution")
axes[0].set_xlabel("Max Probability (Confidence)")
axes[0].set_ylabel("Count")
axes[0].axvline(confidence.mean(), color='red', linestyle='--',
                label=f"Mean={confidence.mean():.2f}")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Per-class probability distributions
for i, (cls, color) in enumerate(zip(le_target.classes_, ['#e74c3c','#3498db','#2ecc71'])):
    axes[1].hist(y_proba_stack[:, i], bins=25, alpha=0.6,
                 label=cls, color=color, edgecolor='white')
axes[1].set_title("Per-Class Probability Distributions (Stacking)")
axes[1].set_xlabel("Predicted Probability")
axes[1].set_ylabel("Count")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Mean prediction confidence : {confidence.mean():.3f}")
print(f"Low-confidence preds (<0.6): {(confidence < 0.6).sum()} / {len(confidence)}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Confusion matrix
cm_stack = confusion_matrix(y_test, y_pred_stack)
sns.heatmap(cm_stack, annot=True, fmt='d', cmap='Purples',
            xticklabels=le_target.classes_,
            yticklabels=le_target.classes_,
            ax=axes[0], linewidths=0.5)
axes[0].set_title(f"Stacking Classifier — Confusion Matrix\n(F1 Macro={stack_f1:.3f})")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")

# ROC-AUC
y_bin_full = label_binarize(y_test, classes=[0, 1, 2])
colors_roc  = ['#e74c3c', '#3498db', '#2ecc71']

for i, (cls, col) in enumerate(zip(le_target.classes_, colors_roc)):
    fpr, tpr, _ = roc_curve(y_bin_full[:, i], y_proba_stack[:, i])
    auc_val = roc_auc_score(y_bin_full[:, i], y_proba_stack[:, i])
    axes[1].plot(fpr, tpr, color=col, lw=2, label=f"{cls} (AUC={auc_val:.3f})")

axes[1].plot([0,1],[0,1],'k--', lw=0.8, alpha=0.5)
axes[1].set_title("Stacking Classifier — ROC-AUC (One-vs-Rest)")
axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

plt.suptitle("Stacking Generalization — Final Evaluation", fontsize=13, y=1.02)
plt.tight_layout()
plt.show()

cv_adv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

all_cv_models = {
    **{k: v['model'] for k, v in adv_results.items() if k != 'Voting Ensemble'},
    'XGBoost (Tuned)':   xgb_search.best_estimator_,
    'LightGBM (Tuned)':  lgbm_search.best_estimator_,
    'Stacking':          stacking_clf,
}

cv_adv_results = {}
print("="*65)
print("5-FOLD STRATIFIED CV — F1 MACRO (ADVANCED + STACKING MODELS)")
print("="*65)

for name, model in all_cv_models.items():
    scores = cross_val_score(model, X_train, y_train,
                             cv=cv_adv, scoring='f1_macro', n_jobs=-1)
    cv_adv_results[name] = scores
    print(f"  {name:25s}  Mean={scores.mean():.4f}  Std={scores.std():.4f}  "
          f"[{scores.min():.3f} – {scores.max():.3f}]")

# ── CV Boxplot ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
data_cv = [cv_adv_results[m] for m in all_cv_models.keys()]
bp = ax.boxplot(data_cv, patch_artist=True, notch=False)

palette = ['#2196F3','#4CAF50','#FF9800','#9C27B0','#E91E63','#00BCD4','#FF5722','#8BC34A']
for patch, color in zip(bp['boxes'], palette):
    patch.set_facecolor(color); patch.set_alpha(0.75)

ax.set_xticklabels(list(all_cv_models.keys()), rotation=20, ha='right')
ax.set_ylabel("F1 Macro Score")
ax.set_title("5-Fold CV Distribution — F1 Macro (Advanced + Stacking Models)")
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# ── Compile complete leaderboard ─────────────────────────────────────────────
all_results = {}

# Advanced baselines
for name, res in adv_results.items():
    all_results[name] = {k: v for k, v in res.items() if k not in ('y_pred', 'model')}

# Tuned models
for name, res in tuned_results.items():
    all_results[name] = {k: v for k, v in res.items() if k not in ('y_pred', 'model')}

# Stacking
all_results['Stacking'] = {
    'Accuracy':          round(stack_acc,  4),
    'Precision (Macro)': round(stack_prec, 4),
    'Recall (Macro)':    round(stack_rec,  4),
    'F1 (Macro)':        round(stack_f1,   4),
}

leaderboard_df = pd.DataFrame([
    {
        'Model':             name,
        'Accuracy':          f"{v['Accuracy']*100:.2f}%",
        'Precision (Macro)': f"{v['Precision (Macro)']*100:.2f}%",
        'Recall (Macro)':    f"{v['Recall (Macro)']*100:.2f}%",
        'F1 Score (Macro)':  f"{v['F1 (Macro)']*100:.2f}%",
    }
    for name, v in all_results.items()
])

# Sort by F1
leaderboard_df['_f1_sort'] = [v['F1 (Macro)'] for v in all_results.values()]
leaderboard_df = leaderboard_df.sort_values('_f1_sort', ascending=False).drop('_f1_sort', axis=1)

print("\n" + "="*80)
print("FINAL MODEL LEADERBOARD (sorted by F1 Macro)")
print("="*80)
print(leaderboard_df.to_string(index=False))
print("="*80)
print("★ Best model highlighted at top of leaderboard")

# ── Final comparison plot ─────────────────────────────────────────────────────
sorted_names = leaderboard_df['Model'].tolist()
f1_vals = [all_results[n]['F1 (Macro)'] for n in sorted_names]
colors_bar = ['#c0392b' if n == sorted_names[0] else '#2980b9' for n in sorted_names]

plt.figure(figsize=(12, 5))
bars = plt.barh(sorted_names, f1_vals, color=colors_bar, edgecolor='gray', lw=0.5)
for bar, val in zip(bars, f1_vals):
    plt.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
             f"{val:.3f}", va='center', fontsize=9)
plt.axvline(x=max(f1_vals), color='red', linestyle='--', alpha=0.4)
plt.xlabel("F1 Score (Macro)")
plt.title("Final Model Leaderboard — F1 Macro Score (Higher = Better)", fontsize=12)
plt.xlim(0, 1.05)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

numeric_df = df.select_dtypes(include=['int64', 'float64'])

# Drop any residual target-encoded columns
for c in ['Hotspot_Level_Encoded', 'Hotspot_Level_enc', '_target_enc']:
    if c in numeric_df.columns:
        numeric_df = numeric_df.drop(c, axis=1)

plt.figure(figsize=(14, 10))
sns.heatmap(numeric_df.corr(), cmap='coolwarm', annot=False, linewidths=0.3)
plt.title("Full Feature Correlation Heatmap (Post-Engineering)")
plt.tight_layout()
plt.show()

# Install pytorch-tabnet (lightweight, no extra GPU setup needed)
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pytorch-tabnet', '-q'])
print("pytorch-tabnet installed successfully.")


# ── Deep Learning Imports ─────────────────────────────────────────────────────
import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device    : {device}")


# ── Class weight computation for imbalanced dataset ──────────────────────────
# Recall: Low=72%, Medium=25%, High=3% — without weighting, model ignores High class

cw_array = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_tabnet = dict(enumerate(cw_array))

print("TabNet class weights (balanced):")
for cls_name, w in zip(le_target.classes_, cw_array):
    print(f"  {cls_name:8s}: {w:.4f}  {'← critical minority' if cls_name == 'High' else ''}")

# TabNet expects weights as a tensor
weights_tensor = torch.FloatTensor(list(class_weight_tabnet.values()))
print(f"\nWeight tensor: {weights_tensor}")


# ── TabNet Model Definition ───────────────────────────────────────────────────
#
# Key hyperparameters explained:
#   n_d / n_a   : width of decision step embedding and attention (32/32 = balanced)
#   n_steps     : number of sequential attention steps (5 = good for ~15 features)
#   gamma       : attention relaxation — how much overlap between steps (1.3 standard)
#   n_independent/n_shared : FC layers per step (2/2 standard)
#   momentum    : batch-norm momentum (0.02 keeps running stats stable on small data)
#   lambda_sparse: sparsity regularisation — pushes attention masks toward hard selection
#   mask_type   : 'sparsemax' gives truly sparse (interpretable) feature selection

tabnet = TabNetClassifier(
    n_d            = 32,           # decision step embedding width
    n_a            = 32,           # attention embedding width
    n_steps        = 5,            # number of sequential attention steps
    gamma          = 1.3,          # attention relaxation factor
    n_independent  = 2,            # independent FC layers per step
    n_shared       = 2,            # shared FC layers per step
    momentum       = 0.02,         # BatchNorm momentum (stable on ~1000-row dataset)
    lambda_sparse  = 1e-3,         # sparsity regularisation
    optimizer_fn   = torch.optim.Adam,
    optimizer_params = dict(lr=2e-2, weight_decay=1e-5),
    scheduler_fn   = torch.optim.lr_scheduler.StepLR,
    scheduler_params = dict(step_size=50, gamma=0.9),
    mask_type      = 'sparsemax',  # true sparse attention (vs entmax for softer)
    cat_idxs       = [],           # all features already encoded numerically
    cat_dims       = [],
    cat_emb_dim    = 1,
    device_name    = device,
    verbose        = 0,
    seed           = 42,
)

print("TabNet model defined with architecture:")
print(f"  n_d={tabnet.n_d}, n_a={tabnet.n_a}, n_steps={tabnet.n_steps}, gamma={tabnet.gamma}")
print(f"  FC layers: {tabnet.n_independent} independent + {tabnet.n_shared} shared per step")
print(f"  Attention mask: {tabnet.mask_type} (sparse, interpretable)")
print(f"  λ_sparse = {tabnet.lambda_sparse} (sparsity regularisation)")
print(f"  Device: {device}")


# ── Train TabNet ──────────────────────────────────────────────────────────────
# TabNet expects numpy float32 arrays
X_train_tab = X_train.astype(np.float32)
X_test_tab  = X_test.astype(np.float32)
y_train_tab = y_train.astype(int)
y_test_tab  = y_test.astype(int)

# Build per-sample weights from class weights (passed to fit as weights param)
train_sample_w_tab = np.array([class_weight_tabnet[yi] for yi in y_train_tab],
                               dtype=np.float32)

print("Starting TabNet training...")
print(f"  Training set : {X_train_tab.shape}")
print(f"  Max epochs   : 200  |  Patience : 30  |  Batch size : 256")
print(f"  Eval metric  : accuracy  |  Class weights: balanced\n")

tabnet.fit(
    X_train      = X_train_tab,
    y_train      = y_train_tab,
    eval_set     = [(X_test_tab, y_test_tab)],
    eval_name    = ['val'],
    eval_metric  = ['accuracy'],
    max_epochs   = 200,
    patience     = 30,           # early stopping: stop if no val improvement for 30 epochs
    batch_size   = 256,
    virtual_batch_size = 128,    # ghost batch normalisation size (must divide batch_size)
    weights      = 1,            # use balanced class weights (1 = auto from class dist)
    drop_last    = False,
    loss_fn      = torch.nn.CrossEntropyLoss(weight=weights_tensor.to(device)),
)

print("\nTabNet training complete.")
print(f"Best epoch: {tabnet.best_epoch}")


# ── TabNet Evaluation on Test Set ─────────────────────────────────────────────
y_pred_tab   = tabnet.predict(X_test_tab)
y_proba_tab  = tabnet.predict_proba(X_test_tab)

tab_acc  = accuracy_score(y_test_tab, y_pred_tab)
tab_prec = precision_score(y_test_tab, y_pred_tab, average='macro', zero_division=0)
tab_rec  = recall_score(y_test_tab, y_pred_tab, average='macro', zero_division=0)
tab_f1   = f1_score(y_test_tab, y_pred_tab, average='macro', zero_division=0)

print("=" * 62)
print("  TABNET DEEP LEARNING MODEL — TEST SET PERFORMANCE")
print("=" * 62)
print(f"  Accuracy          : {tab_acc:.4f}  ({tab_acc*100:.2f}%)")
print(f"  Precision (Macro) : {tab_prec:.4f}  ({tab_prec*100:.2f}%)")
print(f"  Recall (Macro)    : {tab_rec:.4f}  ({tab_rec*100:.2f}%)")
print(f"  F1 Score (Macro)  : {tab_f1:.4f}  ({tab_f1*100:.2f}%)")
print()
print(classification_report(y_test_tab, y_pred_tab,
                             target_names=le_target.classes_, zero_division=0))


# ── Training History — Loss & Accuracy Curves ─────────────────────────────────
train_loss = tabnet.history['loss']
val_acc    = tabnet.history['val_accuracy']

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Loss curve
axes[0].plot(train_loss, color='steelblue', linewidth=1.8, label='Train Loss')
axes[0].set_title("TabNet — Training Loss (per epoch)", fontsize=11)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("CrossEntropy Loss")
axes[0].axvline(x=tabnet.best_epoch, color='red', linestyle='--',
                alpha=0.7, label=f'Best epoch: {tabnet.best_epoch}')
axes[0].legend(); axes[0].grid(alpha=0.3)

# Validation accuracy curve
axes[1].plot(val_acc, color='darkorange', linewidth=1.8, label='Val Accuracy')
axes[1].set_title("TabNet — Validation Accuracy (per epoch)", fontsize=11)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].axvline(x=tabnet.best_epoch, color='red', linestyle='--',
                alpha=0.7, label=f'Best epoch: {tabnet.best_epoch}')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.suptitle("TabNet Deep Learning — Training History", fontsize=13, y=1.02)
plt.tight_layout()
plt.show()


# ── Confusion Matrix & ROC-AUC Curves — TabNet ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Confusion matrix
cm_tab = confusion_matrix(y_test_tab, y_pred_tab)
sns.heatmap(cm_tab, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=le_target.classes_,
            yticklabels=le_target.classes_,
            ax=axes[0], linewidths=0.5)
axes[0].set_title(f"TabNet — Confusion Matrix\n(F1 Macro = {tab_f1:.3f})")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")

# ROC-AUC (one-vs-rest)
y_bin_tab  = label_binarize(y_test_tab, classes=[0, 1, 2])
colors_roc = ['#e74c3c', '#3498db', '#2ecc71']

for i, (cls, col) in enumerate(zip(le_target.classes_, colors_roc)):
    fpr, tpr, _ = roc_curve(y_bin_tab[:, i], y_proba_tab[:, i])
    auc_val = roc_auc_score(y_bin_tab[:, i], y_proba_tab[:, i])
    axes[1].plot(fpr, tpr, color=col, lw=2, label=f"{cls} (AUC = {auc_val:.3f})")

axes[1].plot([0,1], [0,1], 'k--', lw=0.8, alpha=0.5)
axes[1].set_title("TabNet — ROC-AUC Curves (One-vs-Rest)")
axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

plt.suptitle("TabNet Deep Learning — Final Evaluation", fontsize=13, y=1.02)
plt.tight_layout()
plt.show()


# ── TabNet Feature Importances (from Attention Masks) ─────────────────────────
# TabNet's feature_importances_ aggregates sparsemax attention weights
# across all steps and all training samples → meaningful, not post-hoc SHAP

importances_tab = tabnet.feature_importances_  # shape: (n_features,)

fi_tab_df = pd.DataFrame({
    'Feature':    final_features,
    'Importance': importances_tab
}).sort_values('Importance', ascending=False)

print("TabNet Global Feature Importances (attention-based):")
print(fi_tab_df.to_string(index=False))


# ── Feature Importance Bar Chart ─────────────────────────────────────────────
plt.figure(figsize=(10, 5))
fi_plot = fi_tab_df.sort_values('Importance', ascending=True)
colors_fi = ['#c0392b' if v > fi_tab_df['Importance'].quantile(0.75) else '#2980b9'
             for v in fi_plot['Importance']]
plt.barh(fi_plot['Feature'], fi_plot['Importance'], color=colors_fi, edgecolor='gray', lw=0.4)
plt.xlabel("Attention-Based Importance Score")
plt.title("TabNet Global Feature Importances\n(Red = Top Quartile, attention mask aggregation)")
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()


# ── Step-Wise Attention Masks — What TabNet 'Looks At' Per Step ───────────────
# explain() returns per-step attention masks for test samples.
# This shows how the model's focus shifts across sequential decision steps.

explain_matrix, masks = tabnet.explain(X_test_tab)

n_steps_actual = len(masks)
fig, axes = plt.subplots(1, n_steps_actual, figsize=(4 * n_steps_actual, 4), sharey=True)
if n_steps_actual == 1:
    axes = [axes]

for step_idx, (ax, mask) in enumerate(zip(axes, masks)):
    # Average mask across test samples
    avg_mask = mask.mean(axis=0)
    ax.barh(final_features, avg_mask,
            color=plt.cm.RdYlBu_r(avg_mask / (avg_mask.max() + 1e-9)),
            edgecolor='gray', lw=0.3)
    ax.set_title(f"Step {step_idx+1}", fontsize=10)
    ax.set_xlabel("Avg Attention", fontsize=8)
    ax.tick_params(axis='y', labelsize=7)
    ax.grid(axis='x', alpha=0.3)

plt.suptitle("TabNet Step-Wise Feature Attention Masks\n(Each step focuses on different feature subsets)",
             fontsize=11, y=1.03)
plt.tight_layout()
plt.show()


# ── TabNet vs All Previous Models — Final Comprehensive Comparison ────────────
all_results_with_dl = dict(all_results)  # copy from section 16

all_results_with_dl['TabNet (DL)'] = {
    'Accuracy':          round(tab_acc,  4),
    'Precision (Macro)': round(tab_prec, 4),
    'Recall (Macro)':    round(tab_rec,  4),
    'F1 (Macro)':        round(tab_f1,   4),
}

final_lb = pd.DataFrame([
    {
        'Model':             name,
        'Accuracy':          f"{v['Accuracy']*100:.2f}%",
        'Precision (Macro)': f"{v['Precision (Macro)']*100:.2f}%",
        'Recall (Macro)':    f"{v['Recall (Macro)']*100:.2f}%",
        'F1 Score (Macro)':  f"{v['F1 (Macro)']*100:.2f}%",
        'Type':              'Deep Learning' if name == 'TabNet (DL)'
                             else ('Ensemble/Stack' if name in ('Stacking','Voting Ensemble')
                                   else 'ML')
    }
    for name, v in all_results_with_dl.items()
])

final_lb['_f1'] = [v['F1 (Macro)'] for v in all_results_with_dl.values()]
final_lb = final_lb.sort_values('_f1', ascending=False).drop('_f1', axis=1)

print("\n" + "="*85)
print("  COMPREHENSIVE FINAL LEADERBOARD — ALL MODELS (Including TabNet Deep Learning)")
print("="*85)
print(final_lb.drop('Type', axis=1).to_string(index=False))
print("="*85)


# ── Visual Leaderboard with Model Type Colouring ─────────────────────────────
sorted_names_dl = final_lb['Model'].tolist()
f1_vals_dl      = [all_results_with_dl[n]['F1 (Macro)'] for n in sorted_names_dl]
types_dl        = final_lb['Type'].tolist()

color_map_type = {'Deep Learning': '#e74c3c', 'Ensemble/Stack': '#9b59b6', 'ML': '#2980b9'}
colors_final   = [color_map_type[t] for t in types_dl]

fig, ax = plt.subplots(figsize=(13, 6))
bars = ax.barh(sorted_names_dl, f1_vals_dl, color=colors_final, edgecolor='gray', lw=0.5)

for bar, val in zip(bars, f1_vals_dl):
    ax.text(bar.get_width() + 0.004, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va='center', fontsize=9)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=l) for l, c in color_map_type.items()]
ax.legend(handles=legend_elements, loc='lower right', title='Model Type')

ax.axvline(x=max(f1_vals_dl), color='red', linestyle='--', alpha=0.4)
ax.set_xlabel("F1 Score (Macro)", fontsize=11)
ax.set_title("Complete Model Leaderboard: ML vs Ensemble vs Deep Learning (TabNet)\n"
             "Sorted by Macro F1 Score", fontsize=12)
ax.set_xlim(0, 1.08)
ax.gca() if False else ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

