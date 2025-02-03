# src/configs/plotting.py

"""Configuration settings for plotting and visualization."""

from typing import Dict, Any

# Figure dimensions (in inches)
FIGURE_DIMENSIONS = {
    "SINGLE_COLUMN_WIDTH": 3.5,  # IEEE single column (3.5")
    "DOUBLE_COLUMN_WIDTH": 5.2,  # Reduced from 7.2" for compactness
    "STANDARD_HEIGHT": 2.5,  # Reduced for compactness
    "GRID_HEIGHT": 4.0,  # Reduced for grid layouts
    "GRID_SPACING": 0.2,  # Tighter spacing between subplots
    "MARGIN_PAD": 0.01,  # Minimal padding
}

# Typography settings
TYPOGRAPHY = {
    "TITLE_SIZE": 8,  # Smaller title
    "LABEL_SIZE": 7,  # Smaller labels
    "TICK_SIZE": 6,  # Smaller ticks
    "LEGEND_SIZE": 6,  # Smaller legend
    "ANNOTATION_SIZE": 6,  # Smaller annotations
    "FONT_FAMILY": "Arial",  # Standard publication font
    "FONT_WEIGHT": "normal",
    "TITLE_WEIGHT": "bold",
    "TITLE_PAD": 2,  # Reduced padding
    "LABEL_PAD": 1,
}

# Line styling
LINE_STYLE = {
    "LINE_WIDTH": 0.8,  # Thinner lines
    "LINE_ALPHA": 0.9,  # Slightly more opaque
    "GRID_ALPHA": 0.15,  # Lighter grid
    "GRID_WIDTH": 0.3,  # Thinner grid
    "MARKER_SIZE": 3,  # Smaller markers
    "DASH_PATTERN": (2, 1),  # Tighter dash pattern
    "PREDICTION_LINE_STYLE": "--",
    "THRESHOLD_LINE_STYLE": "-.",
    "CHANGE_POINT_LINE_STYLE": ":",
}

# Color scheme (using ColorBrewer qualitative palette)
COLORS = {
    "actual": "#4575b4",  # Blue
    "predicted": "#d73027",  # Red
    "average": "#91bfdb",  # Light blue
    "pred_avg": "#fc8d59",  # Orange
    "change_point": "#666666",  # Dark gray
    "threshold": "#000000",  # Black
    "feature": "#4575b4",  # Blue
    "pred_feature": "#d73027",  # Red
    "grid": "#cccccc",  # Light gray
    "background": "#ffffff",  # White
    "text": "#000000",  # Black
}

# Legend settings
LEGEND_STYLE = {
    "LOCATION": "best",  # Automatic best location
    "COLUMNS": 2,
    "FRAME_ALPHA": 0.9,
    "BORDER_PAD": 0.1,  # Tighter padding
    "HANDLE_LENGTH": 0.8,  # Shorter handles
    "COLUMN_SPACING": 0.5,  # Tighter spacing
    "MARKER_SCALE": 0.7,  # Smaller markers
}

# Grid settings
GRID_STYLE = {
    "MAJOR_ALPHA": 0.15,  # Lighter grid
    "MINOR_ALPHA": 0.07,  # Lighter minor grid
    "MAJOR_LINE_WIDTH": 0.3,  # Thinner grid lines
    "MINOR_LINE_WIDTH": 0.2,  # Thinner minor grid lines
    "LINE_STYLE": ":",
    "MAJOR_TICK_LENGTH": 2,  # Shorter ticks
    "MINOR_TICK_LENGTH": 1,  # Shorter minor ticks
}

# Default network visualization style
DEFAULT_NETWORK_STYLE = {
    "node_size": 100,  # Smaller nodes
    "node_color": COLORS["actual"],
    "edge_color": "#999999",  # Lighter edges
    "font_size": TYPOGRAPHY["TICK_SIZE"],
    "width": LINE_STYLE["LINE_WIDTH"] * 0.8,
    "alpha": LINE_STYLE["LINE_ALPHA"],
    "cmap": "RdBu",  # Red-Blue diverging colormap
    "with_labels": True,
    "arrows": False,
    "label_offset": 0.05,  # Tighter label offset
    "dpi": 300,
}

# Export settings
EXPORT_SETTINGS = {
    "DPI": 600,  # Higher DPI for raster images
    "BBOX_INCHES": "tight",
    "PAD_INCHES": 0.01,  # Minimal padding
    "FORMAT": "png",  # Changed to PNG format
    "TRANSPARENT": False,  # Solid background for better visibility
}


def get_network_style(custom_style: Dict = None) -> Dict:
    """Get network visualization style with optional customization."""
    style = DEFAULT_NETWORK_STYLE.copy()
    if custom_style:
        style.update(custom_style)
    return style


def get_matplotlib_rc_params() -> Dict[str, Any]:
    """Get matplotlib RC parameters for publication-quality plots."""
    return {
        # Font settings
        "font.family": TYPOGRAPHY["FONT_FAMILY"],
        "font.size": TYPOGRAPHY["LABEL_SIZE"],
        "font.weight": TYPOGRAPHY["FONT_WEIGHT"],
        # Axes settings
        "axes.titlesize": TYPOGRAPHY["TITLE_SIZE"],
        "axes.labelsize": TYPOGRAPHY["LABEL_SIZE"],
        "axes.linewidth": LINE_STYLE["LINE_WIDTH"],
        "axes.edgecolor": COLORS["text"],
        "axes.facecolor": COLORS["background"],
        "axes.grid": True,
        "axes.labelweight": "normal",
        "axes.spines.top": False,  # Remove top spine
        "axes.spines.right": False,  # Remove right spine
        # Tick settings
        "xtick.labelsize": TYPOGRAPHY["TICK_SIZE"],
        "ytick.labelsize": TYPOGRAPHY["TICK_SIZE"],
        "xtick.major.width": LINE_STYLE["LINE_WIDTH"],
        "ytick.major.width": LINE_STYLE["LINE_WIDTH"],
        "xtick.minor.width": LINE_STYLE["LINE_WIDTH"] * 0.8,
        "ytick.minor.width": LINE_STYLE["LINE_WIDTH"] * 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        # Grid settings
        "grid.alpha": GRID_STYLE["MAJOR_ALPHA"],
        "grid.linestyle": GRID_STYLE["LINE_STYLE"],
        "grid.linewidth": GRID_STYLE["MAJOR_LINE_WIDTH"],
        # Legend settings
        "legend.fontsize": TYPOGRAPHY["LEGEND_SIZE"],
        "legend.frameon": True,
        "legend.edgecolor": COLORS["text"],
        "legend.facecolor": COLORS["background"],
        "legend.framealpha": LEGEND_STYLE["FRAME_ALPHA"],
        # Figure settings
        "figure.titlesize": TYPOGRAPHY["TITLE_SIZE"],
        "figure.facecolor": COLORS["background"],
        "figure.dpi": 300,
        "figure.constrained_layout.use": True,
        # Text settings
        "text.color": COLORS["text"],
        "text.usetex": False,  # Don't use LaTeX by default
    }
