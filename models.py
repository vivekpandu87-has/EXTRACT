import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib

from mlxtend.frequent_patterns import apriori, association_rules

from utils import preprocess, encode_for_model

RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION  →  Predict Interest_in_MoodCart
# ─────────────────────────────────────────────────────────────────────────────
def train_classification(df, target_col="Interest_in_MoodCart"):
    df_p = preprocess(df)
    X, y = encode_for_model(df_p, target_col=target_col)

    if y is None:
        raise ValueError(f"Target column '{target_col}' not found.")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=RANDOM_STATE, stratify=y_enc
    )

    models_dict = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree":       DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, random_state=RANDOM_STATE),
        "KNN":                 KNeighborsClassifier(n_neighbors=5),
    }

    results = []
    trained = {}
    conf_matrices = {}

    for name, model in models_dict.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_test, y_pred,    average="weighted", zero_division=0)
        f1   = f1_score(y_test, y_pred,        average="weighted", zero_division=0)

        roc = None
        if hasattr(model, "predict_proba") and len(le.classes_) == 2:
            y_prob = model.predict_proba(X_test)[:, 1]
            roc = round(roc_auc_score(y_test, y_prob), 4)

        results.append({
            "Model":     name,
            "Accuracy":  round(acc, 4),
            "Precision": round(prec, 4),
            "Recall":    round(rec, 4),
            "F1 Score":  round(f1, 4),
            "ROC-AUC":   roc,
        })
        trained[name] = model
        conf_matrices[name] = confusion_matrix(y_test, y_pred)

    best_name  = sorted(results, key=lambda x: x["F1 Score"], reverse=True)[0]["Model"]
    best_model = trained[best_name]

    # Feature importances from best tree-based model
    feat_imp = None
    for preferred in ["Random Forest", "Gradient Boosting", "Decision Tree"]:
        m = trained.get(preferred)
        if m and hasattr(m, "feature_importances_"):
            fi = pd.Series(m.feature_importances_, index=X.columns)
            feat_imp = fi.sort_values(ascending=False).head(20).reset_index()
            feat_imp.columns = ["Feature", "Importance"]
            break

    return pd.DataFrame(results), best_model, le, X.columns, conf_matrices, best_name, feat_imp


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION  →  Predict Monthly_Spend
# ─────────────────────────────────────────────────────────────────────────────
def train_regression(df, target_col="Monthly_Spend"):
    df_p = preprocess(df)
    df_p[target_col] = pd.to_numeric(df_p[target_col], errors="coerce")
    df_p = df_p.dropna(subset=[target_col])

    X, y = encode_for_model(df_p, target_col=target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    models_dict = {
        "Linear Regression":     LinearRegression(),
        "Random Forest Reg.":    RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
        "Gradient Boosting Reg.":GradientBoostingClassifier.__new__(GradientBoostingClassifier),
    }
    # rebuild properly (GB is a classifier, use RF for regression variant)
    from sklearn.ensemble import GradientBoostingRegressor
    models_dict["Gradient Boosting Reg."] = GradientBoostingRegressor(
        n_estimators=150, random_state=RANDOM_STATE
    )

    scores  = {}
    trained = {}
    for name, model in models_dict.items():
        model.fit(X_train, y_train)
        r2 = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        mae = np.mean(np.abs(y_test - y_pred))
        scores[name] = {"R²": round(r2, 4), "MAE": round(mae, 0)}
        trained[name] = model

    best_name  = max(scores, key=lambda k: scores[k]["R²"])

    # feature importances
    best_model = trained[best_name]
    feat_imp = None
    if hasattr(best_model, "feature_importances_"):
        fi = pd.Series(best_model.feature_importances_, index=X.columns)
        feat_imp = fi.sort_values(ascending=False).head(20).reset_index()
        feat_imp.columns = ["Feature", "Importance"]

    # actual vs predicted (sample 300)
    idx = np.random.choice(len(y_test), min(300, len(y_test)), replace=False)
    avp = pd.DataFrame({
        "Actual":    np.array(y_test)[idx],
        "Predicted": best_model.predict(X_test.iloc[idx]),
    })
    return scores, best_model, feat_imp, avp


# ─────────────────────────────────────────────────────────────────────────────
# CLUSTERING  →  KMeans Segments
# ─────────────────────────────────────────────────────────────────────────────
def train_clustering(df, k=4):
    df_p = preprocess(df)
    if "Interest_in_MoodCart" in df_p.columns:
        df_p = df_p.drop(columns=["Interest_in_MoodCart"])

    X, _ = encode_for_model(df_p, target_col=None)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km     = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)

    # PCA for 2D viz
    pca  = PCA(n_components=2, random_state=RANDOM_STATE)
    comp = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(comp, columns=["PC1", "PC2"])
    pca_df["Cluster"] = [f"Cluster {l}" for l in labels]

    # elbow data (k=2..9)
    inertias = []
    for ki in range(2, 10):
        inertias.append(KMeans(n_clusters=ki, random_state=RANDOM_STATE, n_init=10)
                        .fit(X_scaled).inertia_)

    return labels, pca_df, inertias


# ─────────────────────────────────────────────────────────────────────────────
# ASSOCIATION RULES  →  Product Combinations
# ─────────────────────────────────────────────────────────────────────────────
def association_mining(df, min_support=0.05):
    if "Product_Combinations" not in df.columns:
        return pd.DataFrame()

    s = df["Product_Combinations"].fillna("").apply(
        lambda x: [t.strip() for t in str(x).split("|") if t.strip()]
    )
    items = sorted({i for sub in s for i in sub})
    if not items:
        return pd.DataFrame()

    onehot = pd.DataFrame(False, index=df.index, columns=items)
    for idx, lst in s.items():
        for it in lst:
            if it in onehot.columns:
                onehot.at[idx, it] = True

    freq = apriori(onehot, min_support=min_support, use_colnames=True)
    if freq.empty:
        return pd.DataFrame()

    try:
        rules = association_rules(freq, metric="confidence",
                                  min_threshold=0.3,
                                  num_itemsets=len(freq))
    except TypeError:
        rules = association_rules(freq, metric="confidence", min_threshold=0.3)

    rules = rules.sort_values(by=["lift", "confidence"], ascending=False)
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
    return rules[["antecedents", "consequents", "support", "confidence", "lift"]].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# SEGMENT PROFILING
# ─────────────────────────────────────────────────────────────────────────────
def segment_profile(df, segment_col):
    df2 = df.copy()
    df2["Monthly_Spend"] = pd.to_numeric(df2["Monthly_Spend"], errors="coerce")

    agg = (df2.groupby(segment_col)
           .agg(
               Count=("Monthly_Spend", "count"),
               Avg_Spend=("Monthly_Spend", "mean"),
               Median_Spend=("Monthly_Spend", "median"),
           )
           .round(0)
           .reset_index())
    agg["Share_%"] = (agg["Count"] / agg["Count"].sum() * 100).round(1)

    if "Interest_in_MoodCart" in df2.columns:
        yes_rate = (df2[df2["Interest_in_MoodCart"] == "Yes"]
                    .groupby(segment_col).size() /
                    df2.groupby(segment_col).size() * 100).round(1)
        agg["Interest_Yes_%"] = agg[segment_col].map(yes_rate).fillna(0)

    return agg


# ─────────────────────────────────────────────────────────────────────────────
# MODEL PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────
def save_model(model, le, feature_cols, prefix="moodcart_model"):
    joblib.dump(model,              f"{prefix}.joblib")
    joblib.dump(le,                 f"{prefix}_le.joblib")
    joblib.dump(list(feature_cols), f"{prefix}_cols.joblib")


def load_model(prefix="moodcart_model"):
    model = joblib.load(f"{prefix}.joblib")
    le    = joblib.load(f"{prefix}_le.joblib")
    cols  = joblib.load(f"{prefix}_cols.joblib")
    return model, le, cols


def predict_new(df_new, model, le, cols):
    df_p = preprocess(df_new)
    X, _  = encode_for_model(df_p, target_col=None)
    for c in cols:
        if c not in X.columns:
            X[c] = 0
    X = X[cols]
    preds = model.predict(X)
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
    return le.inverse_transform(preds), proba, le.classes_
