"""
plotting.py
Handles all visual outputs using matplotlib and seaborn.
---------------------------------------------------------
Note: Does not contain Logistic Regression visualizations.
"""

import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import (ACPT, BG, BODY, GRN, NAVY, OFFICIAL_RATES, RJCT, SLATE,
                    STONE, UOFT, UPENN)
from database import _parse_year, query_temporal_raw


# -------------------------------------------------------------------------------------------------------
def _style_ax(ax, grid: bool = True):
    """
    Plotting Helper:
    Applies consistent background and grid settings to an axes.
    ----------
    Parameters
    ----------
    ax : matplotlib.axes.Axes
    grid : bool
        Whether to show gridlines. Pass False for dense plots like histograms
        where gridlines aren't very useful.
    """
    ax.set_facecolor(BG)
    ax.figure.patch.set_facecolor(BG)
    if grid:
        ax.set_axisbelow(True)   
        ax.grid(True)
    else:
        ax.grid(False)

# -------------------------------------------------------------------------------------------------------
def true_acceptance_rates(upenn_csv, uoft_csv):
    """
    Calculates the aggregate acceptance rate for UPenn and UofT 
    based on available public and official admission data from
    csv files, and visualizes them in a simple bar chart.
    """
    upenn_pb = pd.read_csv(upenn_csv)
    uoft_pb = pd.read_csv(uoft_csv)

    if upenn_pb['Applied'].dtype == 'object':
        upenn_pb['Applied'] = upenn_pb['Applied'].astype(str).str.replace(',', '').astype(float)
        upenn_pb['Admitted'] = upenn_pb['Admitted'].astype(str).str.replace(',', '').astype(float)
        
    if uoft_pb['Applied'].dtype == 'object':
        uoft_pb['Applied'] = uoft_pb['Applied'].astype(str).str.replace(',', '').astype(float)
        uoft_pb['Admitted'] = uoft_pb['Admitted'].astype(str).str.replace(',', '').astype(float)

    # UPenn
    upenn_total_applied = upenn_pb['Applied'].sum()
    upenn_total_admitted = upenn_pb['Admitted'].sum()
    upenn_rate = (upenn_total_admitted / upenn_total_applied) * 100

    # UofT
    uoft_total_applied = uoft_pb['Applied'].sum()
    uoft_total_admitted = uoft_pb['Admitted'].sum()
    uoft_rate = (uoft_total_admitted / uoft_total_applied) * 100

    schools = ['UPenn', 'UofT']
    rates = [upenn_rate, uoft_rate]
    colors = [UPENN, UOFT]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.grid(visible=False, axis='x')
    bars = ax.bar(schools, rates, color=colors, width=0.5, zorder=3)
    ax.set_ylabel('Acceptance Rate (%)')
    ax.set_title("Figure 4: True$^*$ Graduate Acceptance Rates", fontsize=13)
    ax.set_ylim(0, max(rates) + 10) 

    # data labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height / 2),
                    xytext=(0, 5), 
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', color=STONE)
        
    ax.text(-0.02, -0.07, 
            r'$^*$UPenn data estimated. Source details in Section 2.2.',
            transform=ax.transAxes, 
            fontsize=10, 
            fontstyle='italic', 
            color=SLATE, 
            ha='left', 
            va='top')

    plt.tight_layout()
    plt.show()

    return {
        "UPenn Aggregate Acceptance Rate (%)": round(upenn_rate, 2),
        "UofT Aggregate Acceptance Rate (%)": round(uoft_rate, 2)
    }
# ------------------------------------------------------------------------------------------

def plot_gpa_histogram(df: pd.DataFrame, title: str, color: str = UOFT) -> None:
    """
    GPA distribution histogram with KDE overlay.
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned admissions DataFrame.
    title : str
        Plot title.
    color : str
        Bar fill color.
    """
    df_plot = df.dropna(subset=["GPA", "Status_Binary"])
    df_plot = df_plot[df_plot["GPA"] <= 4.0].copy()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    _style_ax(ax, grid=False)   

    sns.histplot(

        df_plot["GPA"],
        bins=20,
        kde=True,
        color=color,
        edgecolor=BG,
        linewidth=1.0,
        alpha=0.75,
        ax=ax,
    )

    ax.set_title(title, fontsize=12, pad=9)
    ax.set_xlabel("Undergraduate GPA")
    ax.set_ylabel("Count")
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.show() 
# ------------------------------------------------------------------------------------------
def plot_gpa_by_outcome(df_uoft: pd.DataFrame, df_upenn: pd.DataFrame) -> None:
    """
    Side-by-side scatter plots of GPA by admission outcome for both schools.
    Includes averages in the legend and boxes the plots for readability.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)
    fig.suptitle("Figure 2: Gradcafe records by outcome", fontsize=11, y=0.02)
    
    datasets = [
        (df_uoft, "University of Toronto", axes[0]),
        (df_upenn, "University of Pennsylvania", axes[1])
    ]
    
    for df, name, ax in datasets:
        _style_ax(ax, grid=True)
        df_plot = df[df["GPA"] <= 4.0].dropna(subset=["GPA", "Status_Binary"]).copy()
        
        n_acpt = len(df_plot[df_plot["Status_Binary"] == 1])
        n_rjct = len(df_plot[df_plot["Status_Binary"] == 0])
        
        mu_acpt = df_plot[df_plot["Status_Binary"] == 1]["GPA"].mean()
        mu_rjct = df_plot[df_plot["Status_Binary"] == 0]["GPA"].mean()
        
        label_acpt = f"Accepted (n={n_acpt}, μ={mu_acpt:.2f})"
        label_rjct = f"Rejected (n={n_rjct}, μ={mu_rjct:.2f})"
        
        df_plot["Outcome"] = df_plot["Status_Binary"].map({
            1: label_acpt, 
            0: label_rjct
        })

        df_plot["Group"] = ""
        
        sns.stripplot(
            data=df_plot, x="GPA", y="Group", hue="Outcome", 
            hue_order=[label_rjct, label_acpt],
            palette={label_acpt: ACPT, label_rjct: RJCT},
            jitter=0.4, size=4.0, alpha=0.6,
            ax=ax, zorder=1
        )
        
        ax.set_title(name,pad=9)
        ax.set_xlabel("Undergraduate GPA")
        ax.set_ylabel("")
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles[:2], labels[:2], 
            frameon=True, facecolor=BG, edgecolor=STONE, 
            fontsize=10, loc="upper left"
        )
        
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(STONE)
            spine.set_linewidth(1.2)
            
        ax.tick_params(axis='y', length=0)
        
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()
# ------------------------------------------------------------------------------------------
def plot_acceptance_by_gpa_range(df_range: pd.DataFrame) -> None:
    """
    Grouped bar chart of acceptance rate across GPA ranges

    Parameters
    ----------
    df_range : pd.DataFrame
        Output of query_acceptance_by_gpa_range().
    """
    range_order = ["< 3.0", "3.0–3.25", "3.25–3.5", "3.5–3.75", "3.75–4.0"]

    def _get(inst):
        return (
            df_range[df_range["institution"] == inst]
            .set_index("gpa_range")
            .reindex(range_order)
        )

    uoft  = _get("University of Toronto")
    upenn = _get("University of Pennsylvania")

    x     = np.arange(len(range_order))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 5.5))
    _style_ax(ax, grid=True)
    ax.grid(visible=True, axis="y")
    ax.grid(visible=False, axis="x")
    ax.set_axisbelow(True)

    bars_uoft  = ax.bar(x - width / 2, uoft["acceptance_rate_pct"],  width,
                        color=UOFT,  edgecolor=BG, linewidth=0.3,
                        label="University of Toronto")
    bars_upenn = ax.bar(x + width / 2, upenn["acceptance_rate_pct"], width,
                        color=UPENN, edgecolor=BG, linewidth=0.3,
                        label="University of Pennsylvania")

    ax.set_xticks(x)
    ax.set_xticklabels(range_order)
    ax.set_xlabel("GPA Range")
    ax.set_ylabel("Acceptance Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Figure 3: Acceptance Rate by GPA Range", fontsize=13, pad=14)
    ax.legend(frameon=True, fontsize=11)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.show()
# ------------------------------------------------------------------------------------------
def plot_bias_comparison(df_rates: pd.DataFrame) -> None:
    """
    Grouped bar chart comparing GradCafe's implied acceptance rate to the
    officially published rate at each institution.

    Parameters
    ----------
    df_rates : pd.DataFrame
        Output of query_groupby_acceptance_rate().
    """
    institutions = list(OFFICIAL_RATES.keys())
    x = np.arange(len(institutions))
    width = 0.36

    gradcafe_rates = []
    official_rates = []
    for inst in institutions:
        row = df_rates[df_rates["institution"] == inst]
        gradcafe_rates.append(row["acceptance_rate_pct"].values[0] if len(row) else 0)
        official_rates.append(OFFICIAL_RATES[inst])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    _style_ax(ax, grid=True)

    bars1 = ax.bar(
        x - width / 2, gradcafe_rates, width,
        label="GradCafe (self-reported)",
        color=NAVY, edgecolor=BG, linewidth=0,
    )
    bars2 = ax.bar(
        x + width / 2, official_rates, width,
        label="Available published rate",
        color=SLATE, edgecolor=BG, linewidth=0,
    )

    for container in [bars1, bars2]:
        for bar in container:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.8,
                f"{h:.1f}%",
                ha="center",
                fontsize=10,
                color=BODY,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(["University of Toronto", "University of Pennsylvania"])
    ax.set_ylabel("Acceptance Rate (%)")
    ax.set_title("Figure 5: GradCafe vs. Official Acceptance Rate",fontsize=13, pad=14)
    ax.set_ylim(0, max(max(gradcafe_rates), max(official_rates)) * 1.28)
    ax.legend(frameon=False, fontsize=11)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.show()
# ------------------------------------------------------------------------------------------
def plot_temporal_trends(conn: sqlite3.Connection) -> None:
    """
    Acceptance rate by year for each institution, derived from the season column.
    Only years with at least 15 applicants are shown for statistical significance.
    """
    df_raw = query_temporal_raw(conn)
    df_raw["year"] = df_raw["season"].apply(_parse_year)
    df_raw = df_raw.dropna(subset=["year"])
    df_raw["year"] = df_raw["year"].astype(int)

    yearly = (
        df_raw.groupby(["institution", "year"])
        .agg(total=("accepted", "count"), accepted=("accepted", "sum"))
        .reset_index()
    )
    yearly = yearly[yearly["total"] >= 15]
    yearly["acceptance_rate"] = 100.0 * yearly["accepted"] / yearly["total"]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    _style_ax(ax, grid=True)

    for inst, color in [
        ("University of Toronto",      UOFT),
        ("University of Pennsylvania", UPENN),
    ]:
        grp = yearly[yearly["institution"] == inst].sort_values("year")
        ax.plot(
            grp["year"],
            grp["acceptance_rate"],
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=6,
            markeredgecolor=BG,
            markeredgewidth=1.2,
            label=inst,
        )

    ax.set_title("Figure 6: Acceptance Rate Over Time",fontsize=13, pad=14)
    ax.set_xlabel("Application Year")
    ax.set_ylabel("Acceptance Rate (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.show()
# ------------------------------------------------------------------------------------------
def plot_gre_impact(df_gre: pd.DataFrame) -> None:
    """
    Grouped bar chart showing the impact of reporting a GRE score 
    on acceptance rates across institutions.
    """
    color_no_gre = "#C24E4E" 
    color_gre = "#3b906b" 
    
    pivot_df = df_gre.pivot(index="institution", columns="gre_reported", values="acceptance_rate_pct").fillna(0)
    n_df = df_gre.pivot(index="institution", columns="gre_reported", values="total").fillna(0)
    
    institutions = pivot_df.index.tolist()
    x = np.arange(len(institutions))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(9, 6))
    _style_ax(ax, grid=True)
    ax.grid(visible=True, axis="y", zorder=0)
    ax.grid(visible=False, axis="x")

    rates_no_gre = pivot_df[0].values
    rates_gre = pivot_df[1].values
 
    bars1 = ax.bar(x - width/2, rates_no_gre, width, label="No GRE", color=color_no_gre, edgecolor=BG, linewidth=1, zorder=3)
    bars2 = ax.bar(x + width/2, rates_gre, width, label="GRE Reported", color=color_gre, edgecolor=BG, linewidth=1, zorder=3)
    
    ax.set_xticks(x)
    ax.set_xticklabels(institutions, fontweight="medium")
    ax.set_ylabel("Acceptance Rate (%)")
    ax.set_title("Figure 7: Impact of GRE Reporting on Acceptance Rates",fontsize=13, pad=20)
    ax.set_ylim(0, max(max(rates_no_gre), max(rates_gre)) * 1.3)
    
    # data labels
    for i, bar in enumerate(bars1):
        n_val = int(n_df[0].values[i])
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{bar.get_height():.1f}%\n(n={n_val})", ha='center', va='bottom', fontsize=10, color=BODY)
        
    for i, bar in enumerate(bars2):
        n_val = int(n_df[1].values[i])
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{bar.get_height():.1f}%\n(n={n_val})", ha='center', va='bottom', fontsize=10, color=BODY)
                
    ax.legend(frameon=True, facecolor=BG, edgecolor=STONE, fontsize=11, loc="upper right")
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.show()
# ------------------------------------------------- EOF ---------------------------------------------------------