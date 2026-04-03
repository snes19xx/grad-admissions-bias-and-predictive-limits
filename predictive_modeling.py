"""
predictive_modeling.py
Handles all statistical modeling (Logistic regression)
and corresponding visualizations
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import HTML, display
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (BG, BODY, GRN, NAVY, OFFICIAL_RATES_PM, SLATE, STONE, UOFT,
                    UPENN)
from plotting import _style_ax

# Applicant A and B case study profiles:
cases = [
    {
        "GPA": 3.92, "GRE_Reported": 0, "key": "uoft",
        "school": "University of Toronto",
        "Profile": "A: GPA 3.92, no GRE -- UofT",
        "Label": "Applicant\nA"
    },
    {
        "GPA": 3.92, "GRE_Reported": 0, "key": "upenn",
        "school": "University of Pennsylvania",
        "Profile": "A: GPA 3.92, no GRE -- UPenn",
        "Label": ""
    },
    {
        "GPA": 3.44, "GRE_Reported": 1, "key": "uoft",
        "school": "University of Toronto",
        "Profile": "B: GPA 3.44, GRE -- UofT",
        "Label": "Applicant\nB"
    },
    {
        "GPA": 3.44, "GRE_Reported": 1, "key": "upenn",
        "school": "University of Pennsylvania",
        "Profile": "B: GPA 3.44, GRE -- UPenn",
        "Label": ""
    },
]

# ----------------------------------------------------------------------------------------------------------
def build_logistic_regression(uoft_df: pd.DataFrame, upenn_df: pd.DataFrame) -> dict:
    """
    Train one logistic regression model per institution on GPA and GRE_Reported.
    """
    results = {}
    configs = [
        ("uoft",  uoft_df,  "University of Toronto"),
        ("upenn", upenn_df, "University of Pennsylvania"),
    ]

    for key, df, school_name in configs:
        features = df[["GPA", "GRE_Reported"]].copy()
        target   = df["Status_Binary"].copy()

        X_train, X_test, y_train, y_test = train_test_split(
            features, target,
            test_size=0.20,
            random_state=42,
            stratify=target,
        )

        scaler     = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        model = LogisticRegression(
            max_iter=500, class_weight="balanced", random_state=42
        )
        model.fit(X_train_sc, y_train)
        pi_train = 0.5 
        print(
            f"{school_name}\n"
            f"  Train: {len(y_train):,}  Test: {len(y_test):,}  "
        )

        results[key] = {
            "model":      model,
            "scaler":     scaler,
            "X_test_sc":  X_test_sc,
            "y_test":     y_test,
            "X_train_sc": X_train_sc,
            "y_train":    y_train,
            "pi_train":   pi_train,
            "n_train":    len(y_train),
            "n_test":     len(y_test),
            "school":     school_name,
        }

    return results
# ----------------------------------------------------------------------------------------------------------
def adjust_probabilities(probs_raw: np.ndarray,
                         pi_true: float,
                         pi_train: float) -> np.ndarray:
    """
    Saerens prior adjustment: correct predicted probabilities for GradCafe's
    inflated acceptance rate.
    Reference:https://doi.org/10.1162/089976602753284446
    """
    p   = np.asarray(probs_raw, dtype=float)
    num = p * (pi_true / pi_train)
    den = num + (1.0 - p) * ((1.0 - pi_true) / (1.0 - pi_train))
    return num / den
# ----------------------------------------------------------------------------------------------------------
def get_coefficients_table(models: dict) -> pd.DataFrame:
    """
    Coefficient table comparing GPA and GRE_Reported across both school models.
    """
    rows = []
    for key, label in [("uoft", "UofT"), ("upenn", "UPenn")]:
        m = models[key]
        coefs = m["model"].coef_[0]
        rows.append({
            "Institution": label,
            "Feature":     "GPA",
            "Coefficient": round(coefs[0], 4),
            "Odds Ratio":  round(np.exp(coefs[0]), 4),
        })
        rows.append({
            "Institution": label,
            "Feature":     "GRE_Reported",
            "Coefficient": round(coefs[1], 4),
            "Odds Ratio":  round(np.exp(coefs[1]), 4),
        })
    return pd.DataFrame(rows)
# ----------------------------------------------------------------------------------------------------------
def predict_applicant_cases(models: dict) -> pd.DataFrame:
    """
    Predict admission probability for two example applicant profiles
    at both schools using each school's own model.
    """
    rows = []
    for c in cases:
        m       = models[c["key"]]
        X       = np.array([[c["GPA"], c["GRE_Reported"]]])
        raw     = m["model"].predict_proba(m["scaler"].transform(X))[0, 1]
        pi_true = OFFICIAL_RATES_PM[c["school"]]
        adj     = adjust_probabilities(np.array([raw]), pi_true, m["pi_train"])[0]
        
        rows.append({
            "Profile":            c["Profile"],
            "Label":              c["Label"],
            "School_Key":         c["key"],
            "Raw_Prob_Num":       raw * 100,
            "Adj_Prob_Num":       adj * 100,
            "GradCafe P(admit)":  f"{raw*100:.1f}%",
            "Adjusted P(admit)":  f"{adj*100:.1f}%",
        })
    return pd.DataFrame(rows)
# ----------------------------------------------------------------------------------------------------------
#                                                   PLOTS
# ----------------------------------------------------------------------------------------------------------
def plot_roc_curves(models: dict) -> None:
    """
    Per-school ROC curves with AUC values in the legend.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    _style_ax(ax, grid=True)

    for key, color, label in [
        ("uoft",  UOFT,  "UofT"),
        ("upenn", UPENN, "UPenn"),
    ]:
        m        = models[key]
        y_prob   = m["model"].predict_proba(m["X_test_sc"])[:, 1]
        fpr, tpr, _ = roc_curve(m["y_test"], y_prob)
        auc      = roc_auc_score(m["y_test"], y_prob)
        ax.plot(fpr, tpr, color=color, lw=2.4,
                label=f"{label}  AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color=STONE, lw=1.2, label="Random classifier")

    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title("Figure 8: ROC Curves", fontsize=14, pad=14)
    ax.legend(loc="lower right", frameon=True, facecolor="white",
              edgecolor="none", fontsize=9)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.show()
# ----------------------------------------------------------------------------------------------------------
def plot_confusion_breakdown(models: dict) -> None:
    """
    For each school: what fraction of truly accepted and truly rejected
    applicants did that school's model correctly or incorrectly predict?
    Shown as grouped bars. A more interpretable alternative to a confusion matrix.
    """
    metrics_order = [
        "Rejected: correct",
        "Rejected: Incorrectly Predicted",
        "Accepted: correct",
        "Accepted: Incorrectly Predicted",
    ]

    results = {}
    for key, label in [("uoft", "UofT"), ("upenn", "UPenn")]:
        m      = models[key]
        y_pred = m["model"].predict(m["X_test_sc"])
        cm     = confusion_matrix(m["y_test"], y_pred)
        tn, fp, fn, tp = cm.ravel()
        results[label] = {
            "Rejected: correct":               tn / (tn + fp) * 100,
            "Rejected: Incorrectly Predicted": fp / (tn + fp) * 100,
            "Accepted: correct":               tp / (tp + fn) * 100,
            "Accepted: Incorrectly Predicted": fn / (tp + fn) * 100,
        }

    x     = np.arange(len(metrics_order))
    width = 0.32

    fig, ax = plt.subplots(figsize=(11, 5))
    _style_ax(ax, grid=True)
    ax.grid(visible=True, axis="y")
    ax.grid(visible=False, axis="x")
    ax.set_axisbelow(True)

    for i, (label, color) in enumerate(zip(["UofT", "UPenn"], [UOFT, UPENN])):
        vals   = [results[label][m] for m in metrics_order]
        offset = (i - 0.5) * width
        bars   = ax.bar(x + offset, vals, width, color=color,
                        edgecolor=BG, linewidth=0, label=label)
        for bar, v in zip(bars, vals):
            if bar.get_height() > 5:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() - 2.0,
                    f"{v:.0f}%",
                    ha="center", va="top",
                    fontsize=10, fontweight="bold", color=STONE,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [m.replace(": ", "\n") for m in metrics_order], fontsize=10
    )
    ax.set_ylabel("% of True Class")
    ax.set_ylim(0, 105)
    ax.set_title(
        "Figure 9: Model Prediction Breakdown by Class and Institution",
        fontsize=13, pad=14,
    )
    ax.legend(frameon=False, fontsize=11, loc="upper center",
              bbox_to_anchor=(0.5, 0.95), ncol=2)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.show()
# ----------------------------------------------------------------------------------------------------------
def print_model_metrics(models: dict) -> pd.DataFrame:
    """
    Classification metrics for each school model on its own test set.
    Displayed as an HTML table.
    """
    def _metrics(m):
        y_pred = m["model"].predict(m["X_test_sc"])
        y_prob = m["model"].predict_proba(m["X_test_sc"])[:, 1]
        r = classification_report(
            m["y_test"], y_pred,
            target_names=["Rejected", "Accepted"],
            output_dict=True,
        )
        return [
            accuracy_score(m["y_test"], y_pred),
            roc_auc_score(m["y_test"], y_prob),
            r["Accepted"]["precision"],
            r["Accepted"]["recall"],
            r["Accepted"]["f1-score"],
            r["Rejected"]["precision"],
            r["Rejected"]["recall"],
            r["Rejected"]["f1-score"],
        ]

    names = [
        "Accuracy", "ROC-AUC",
        "Precision (Accepted)", "Recall (Accepted)", "F1 (Accepted)",
        "Precision (Rejected)", "Recall (Rejected)", "F1 (Rejected)",
    ]

    rows = {
        "Metric": names,
        "UofT":   _metrics(models["uoft"]),
        "UPenn":  _metrics(models["upenn"]),
    }
    summary = pd.DataFrame(rows)
    for col in ["UofT", "UPenn"]:
        summary[col] = summary[col].map("{:.4f}".format)

    display(HTML(summary.to_html(index=False)))
    return summary
# ----------------------------------------------------------------------------------------------------------
def plot_applicant_cases(cases_df: pd.DataFrame) -> None:
    """
    Horizontal bar chart showing raw vs. bias-adjusted admission probabilities
    using the DataFrame generated by predict_applicant_cases.
    Light bars = GradCafe raw. Solid bars = Saerens-adjusted.
    """
    raw_vals = cases_df["Raw_Prob_Num"].tolist()
    adj_vals = cases_df["Adj_Prob_Num"].tolist()
    colors = [UOFT if k == "uoft" else UPENN for k in cases_df["School_Key"]]

    y = np.array([0, 0.7, 2.0, 2.7])
    h = 0.32

    fig, ax = plt.subplots(figsize=(12, 6.5))
    _style_ax(ax, grid=True)
    ax.grid(visible=True, axis="x")
    ax.grid(visible=False, axis="y")

    for i, (raw, adj, color) in enumerate(zip(raw_vals, adj_vals, colors)):
        ax.barh(y[i] + h / 2, raw, h, color=color, alpha=0.3, edgecolor=BG)
        ax.barh(y[i] - h / 2, adj, h, color=color, edgecolor=BG)

        ax.text(raw / 2, y[i] + h / 2, f"{raw:.1f}%",
                va="center", ha="center", fontsize=11, color=BODY)
        ax.text(adj / 2, y[i] - h / 2, f"{adj:.1f}%",
                va="center", ha="center", fontsize=11, color=BG)

    ax.set_yticks([0.35, 2.35])
    ax.set_yticklabels(["Applicant\nA", "Applicant\nB"], fontsize=13)
    
    ax.set_xlabel("Predicted Admission Probability (%)", fontsize=12)
    fig.suptitle(
        "Figure 10: Case Study of Two Applicant Profiles",
        fontsize=16, y=0.99
    )

    fig.text(
        0.5, 0.94,
        "Lighter bars based on raw (unadjusted) GradCafe training data",
        ha="center", va="top", fontsize=11.5, color=SLATE
    )
    ax.set_xlim(0, 80)

    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor=UOFT,  label="UofT", linewidth=0),
            Patch(facecolor=UPENN, label="UPenn",linewidth=0),
        ],
        frameon=False, fontsize=12, loc="lower right",
    )
    sns.despine(ax=ax, left=True)
    plt.tight_layout()
    plt.show()
# --------------------------------------------- EOF ----------------------------------------------------------