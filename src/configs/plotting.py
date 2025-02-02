# src/configs/plotting.py

"""Configuration settings for plotting and visualization."""

from typing import Dict

# Figure dimensions (in inches)
FIGURE_DIMENSIONS = {
    "SINGLE_COLUMN_WIDTH": 5.5,  # standard for single column
    "DOUBLE_COLUMN_WIDTH": 7.2,  # standard for double column
    "STANDARD_HEIGHT": 4.0,
    "GRID_HEIGHT": 6.0,  # for grid layouts
    "GRID_SPACING": 0.3,  # spacing between subplots
}

# Typography settings
TYPOGRAPHY = {
    "TITLE_SIZE": 10,
    "LABEL_SIZE": 8,
    "TICK_SIZE": 6,
    "LEGEND_SIZE": 7,
    "ANNOTATION_SIZE": 6,
}

# Line styling
LINE_STYLE = {
    "LINE_WIDTH": 1.0,
    "LINE_ALPHA": 0.8,
    "GRID_ALPHA": 0.2,
    "GRID_WIDTH": 0.5,
}

# Color scheme
COLORS = {
    "actual": "#1f77b4",
    "predicted": "#ff7f0e",
    "average": "#2ca02c",
    "pred_avg": "#d62728",
    "change_point": "red",
    "threshold": "#17becf",
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
