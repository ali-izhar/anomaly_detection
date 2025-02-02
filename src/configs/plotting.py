# src/configs/plotting.py

"""Configuration settings for plotting and visualization."""

from typing import Dict, Any

# Figure dimensions (in inches)
FIGURE_DIMENSIONS = {
    "SINGLE_COLUMN_WIDTH": 5.5,  # standard for single column
    "DOUBLE_COLUMN_WIDTH": 7.2,  # standard for double column
    "STANDARD_HEIGHT": 4.0,
    "GRID_HEIGHT": 6.0,  # for grid layouts
    "GRID_SPACING": 0.3,  # spacing between subplots
    "MARGIN_PAD": 0.02,  # padding for tight layout
}

# Typography settings
TYPOGRAPHY = {
    "TITLE_SIZE": 10,
    "LABEL_SIZE": 8,
    "TICK_SIZE": 6,
    "LEGEND_SIZE": 7,
    "ANNOTATION_SIZE": 6,
    "FONT_FAMILY": "sans-serif",
    "FONT_WEIGHT": "normal",
    "TITLE_WEIGHT": "bold",
    "TITLE_PAD": 4,
    "LABEL_PAD": 2,
}

# Line styling
LINE_STYLE = {
    "LINE_WIDTH": 1.0,
    "LINE_ALPHA": 0.8,
    "GRID_ALPHA": 0.2,
    "GRID_WIDTH": 0.5,
    "MARKER_SIZE": 4,
    "DASH_PATTERN": (4, 2),  # for dashed lines
    "PREDICTION_LINE_STYLE": "--",
    "THRESHOLD_LINE_STYLE": "-.",
    "CHANGE_POINT_LINE_STYLE": ":",
}

# Color scheme
COLORS = {
    "actual": "#1f77b4",  # Blue
    "predicted": "#ff7f0e",  # Orange
    "average": "#2ca02c",  # Green
    "pred_avg": "#d62728",  # Red
    "change_point": "#e74c3c",  # Bright red
    "threshold": "#17becf",  # Cyan
    "feature": "#3498db",  # Light blue
    "pred_feature": "#e67e22",  # Dark orange
    "grid": "#cccccc",  # Light gray
    "background": "#ffffff",  # White
    "text": "#333333",  # Dark gray
}

# Legend settings
LEGEND_STYLE = {
    "LOCATION": "upper right",
    "COLUMNS": 2,
    "FRAME_ALPHA": 0.8,
    "BORDER_PAD": 0.2,
    "HANDLE_LENGTH": 1.0,
    "COLUMN_SPACING": 0.8,
    "MARKER_SCALE": 0.8,
}

# Grid settings
GRID_STYLE = {
    "MAJOR_ALPHA": 0.2,
    "MINOR_ALPHA": 0.1,
    "MAJOR_LINE_WIDTH": 0.5,
    "MINOR_LINE_WIDTH": 0.3,
    "LINE_STYLE": ":",
    "MAJOR_TICK_LENGTH": 4,
    "MINOR_TICK_LENGTH": 2,
}

# Default network visualization style
DEFAULT_NETWORK_STYLE = {
    "node_size": 150,
    "node_color": COLORS["actual"],
    "edge_color": "#7f7f7f",
    "font_size": TYPOGRAPHY["TICK_SIZE"],
    "width": LINE_STYLE["LINE_WIDTH"] * 0.8,
    "alpha": LINE_STYLE["LINE_ALPHA"],
    "cmap": "viridis",
    "with_labels": True,
    "arrows": False,
    "label_offset": 0.1,
    "dpi": 300,
}

# Export settings
EXPORT_SETTINGS = {
    "DPI": 300,
    "BBOX_INCHES": "tight",
    "PAD_INCHES": 0.02,
    "FORMAT": "png",
    "TRANSPARENT": False,
}


def get_network_style(custom_style: Dict = None) -> Dict:
    """Get network visualization style with optional customization.

    Args:
        custom_style: Optional dict of style overrides
    Returns:
        Dict of style settings
    """
    style = DEFAULT_NETWORK_STYLE.copy()
    if custom_style:
        style.update(custom_style)
    return style


def get_matplotlib_rc_params() -> Dict[str, Any]:
    """Get matplotlib RC parameters for consistent styling.

    Returns:
        Dict of matplotlib RC parameters
    """
    return {
        "font.family": TYPOGRAPHY["FONT_FAMILY"],
        "font.size": TYPOGRAPHY["LABEL_SIZE"],
        "axes.titlesize": TYPOGRAPHY["TITLE_SIZE"],
        "axes.labelsize": TYPOGRAPHY["LABEL_SIZE"],
        "xtick.labelsize": TYPOGRAPHY["TICK_SIZE"],
        "ytick.labelsize": TYPOGRAPHY["TICK_SIZE"],
        "legend.fontsize": TYPOGRAPHY["LEGEND_SIZE"],
        "figure.titlesize": TYPOGRAPHY["TITLE_SIZE"],
        "axes.grid": True,
        "grid.alpha": GRID_STYLE["MAJOR_ALPHA"],
        "grid.linestyle": GRID_STYLE["LINE_STYLE"],
        "axes.linewidth": LINE_STYLE["LINE_WIDTH"],
        "axes.edgecolor": COLORS["text"],
        "axes.facecolor": COLORS["background"],
        "figure.facecolor": COLORS["background"],
        "text.color": COLORS["text"],
        "axes.labelcolor": COLORS["text"],
        "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"],
    }
