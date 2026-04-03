"""
config.py
Stores global configurations, constants, and color palettes.
"""

import matplotlib
import pandas as pd

pd.set_option('display.max_columns', 12)
pd.set_option('display.width', 120)

# Color Palette -------------------------------------------------------------
BG   = "#FFFFFF"
STONE  = "#DDD6CA"
NAVY   = "#0D1B2A"
INK    = "#162236"
GRN    = "#4c6a52"
GRN_L  = "#6e9676"
RJCT   =  "#c60000c0"
ACPT   = "#89c080d8"
UOFT   = "#1e3765"
UPENN  = "#990000"
SLATE  = "#5A6477"
BODY   = "#151516"

# Typography -------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Lora", "DejaVu Serif", "Times New Roman"],
    "font.size":          13,
    "axes.titlesize":     18,
    "axes.labelsize":     14,
    "xtick.labelsize":    12,
    "ytick.labelsize":    12,
    "text.color":         BODY,
    "axes.labelcolor":    BODY,
    "xtick.color":        SLATE,
    "ytick.color":        SLATE,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   True,
    "axes.spines.bottom": True,
    "axes.grid":          True,
    "grid.color":         STONE,
    "grid.linewidth":     0.6,
    "grid.alpha":         0.8,
    "figure.facecolor":   BG,
    "axes.facecolor":     BG,
    "savefig.facecolor":  BG,
    "savefig.dpi":        300,
})

# ------------------------------------------------------------------------------------------------
# Available acceptance rates used for bias comparison.
# UofT: https://data.utoronto.ca/data-and-reports/facts-and-figures/facts-and-figures-students/
# UPenn: weighted estimate from published Master's program data.

OFFICIAL_RATES = {
    "University of Toronto":      33.33,
    "University of Pennsylvania": 21.44,  
}

# Official rates/100 for statistical modeling in section 4

OFFICIAL_RATES_PM = {
    "University of Toronto":      0.3333,
    "University of Pennsylvania": 0.2144,
}


# ----------------------------------------- EOF ---------------------------------------------------------