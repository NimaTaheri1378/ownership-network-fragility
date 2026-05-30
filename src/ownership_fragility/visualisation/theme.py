from __future__ import annotations


def apply_matplotlib_theme() -> None:
    """Apply a clean publication-oriented matplotlib theme."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "figure.titlesize": 15,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
        }
    )


def apply_plotly_template() -> None:
    """Register a clean Plotly template when Plotly is installed."""
    import plotly.io as pio

    base = pio.templates["plotly_white"]
    base.layout.font = {"family": "Arial, sans-serif", "size": 13}
    base.layout.margin = {"l": 70, "r": 30, "t": 70, "b": 60}
    base.layout.hovermode = "x unified"
    pio.templates["onf_clean"] = base
    pio.templates.default = "onf_clean"
