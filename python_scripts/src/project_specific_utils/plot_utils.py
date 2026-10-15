import numpy as np
from matplotlib.ticker import ScalarFormatter


"""
CompactScientificFormatter
Tick formatter that, when the axis values are small or large enough to need
scientific notation, writes every tick with the fewest exact decimals, at
least one (e.g. 3.5, or 0.25 when needed), and shows
the common power of ten once above the axis (x10^n). Axes with ordinary values
keep matplotlib's default labels. Short labels keep the plot areas of
different figures aligned.

INPUT:
    - min_exponent: int -> values whose order of magnitude is at most this
        (e.g. -2 for 0.015) use the scientific form
    - max_exponent: int -> values whose order of magnitude is at least this
        (e.g. 4 for 25000) use the scientific form
"""
class CompactScientificFormatter(ScalarFormatter):
    def __init__(self, min_exponent=-2, max_exponent=4):
        super().__init__(useMathText=True)
        self.set_powerlimits((min_exponent, max_exponent))
    # EOF

    def _set_format(self):
        super()._set_format()
        # Rescaled ticks get the fewest decimals (at least 1, at most 3) that
        # write every tick exactly, e.g. 3.5 or 0.25 -- never a rounded 0.3.
        if self.orderOfMagnitude != 0:
            scaled_ticks = np.asarray(self.locs) / 10.0 ** self.orderOfMagnitude
            decimals = next(
                (
                    count for count in (1, 2, 3)
                    if np.allclose(np.round(scaled_ticks, count), scaled_ticks)
                ),
                3,
            )
            self.format = f"%1.{decimals}f"
        # end if scientific
    # EOF
# EOC


"""
use_compact_scientific_ticks
Apply CompactScientificFormatter to the y (and/or x) axis of a plot and set the
size of the x10^n label.

INPUT:
    - axis: matplotlib.axes.Axes -> axis to format
    - which: str -> "y", "x" or "both"
    - fontsize: float | None -> size of the x10^n label; None keeps the default

OUTPUT:
    - None
"""
def use_compact_scientific_ticks(axis, which="y", fontsize=None):
    targets = {"y": [axis.yaxis], "x": [axis.xaxis], "both": [axis.xaxis, axis.yaxis]}[which]
    for target in targets:
        target.set_major_formatter(CompactScientificFormatter())
        if fontsize is not None:
            target.get_offset_text().set_fontsize(fontsize)
        # end if fontsize
    # end for target
# EOF


"""
plot_layer_curves_with_centroids
Draw one similarity curve per layer, coloured by depth, with each layer's
centroid as a short tick below zero (Fig 3a-3d style).

INPUT:
    - axis: matplotlib.axes.Axes -> axis to draw on
    - time_ms: np.ndarray -> (time,) coordinate of the curves
    - curves: np.ndarray -> layers x time similarity
    - centroid_ms: np.ndarray -> (layers,) absolute centroid time, NaN = none
    - layer_colors: np.ndarray -> layers x RGBA colours
    - onset_ms: float -> frame onset, drawn as a dashed line
    - window_ms: tuple[float, float] | None -> latency window, shaded grey
    - tick_fraction: float -> centroid tick height as a fraction of the y max
    - show_negative: bool -> keep negative curve values in view (e.g.
        cross-validated R2 below its floor) instead of clipping them

OUTPUT:
    - None
"""
def plot_layer_curves_with_centroids(
        axis, time_ms, curves, centroid_ms, layer_colors, onset_ms,
        window_ms=None, tick_fraction=0.12, show_negative=False,
        ):
    y_top = 1.05 * np.nanmax(curves)
    # Ticks are sized on the full range when every curve can be negative.
    tick_height = tick_fraction * (
        y_top - np.nanmin(curves) if show_negative else y_top
    )
    if window_ms is not None:
        axis.axvspan(*window_ms, color="0.9", zorder=0)
    # end if window shaded
    for layer_index, layer_curve in enumerate(curves):
        axis.plot(time_ms, layer_curve, color=layer_colors[layer_index], linewidth=1.5)
        if np.isfinite(centroid_ms[layer_index]):
            axis.vlines(
                centroid_ms[layer_index], -tick_height, 0,
                color=layer_colors[layer_index], linewidth=3, zorder=15,
            )
        # end if centroid defined
    # end for layer_index
    axis.axhline(0, color="grey", linewidth=0.8)
    axis.axvline(onset_ms, color="grey", linewidth=0.8, linestyle="--")
    if show_negative:
        # Whole range in view, with room for the ticks below zero.
        axis.set_ylim(
            min(np.nanmin(curves), -1.2 * tick_height),
            max(y_top, 0.2 * tick_height),
        )
    else:
        # Negative curve values are clipped below the tick area, as in Fig 3.
        axis.set_ylim(-1.2 * tick_height, y_top)
    # end if show_negative
    return None
# EOF


"""
plot_depth_latency_points
Temporal-score scatter: layer latency (x) against normalized depth (y), with
the depth-latency Spearman rho and p value in a box.

INPUT:
    - axis: matplotlib.axes.Axes -> axis to draw on
    - latency_ms: np.ndarray -> (layers,) latency from the frame onset, NaN = none
    - layer_depths: np.ndarray -> (layers,) normalized depth in [0, 1]
    - layer_colors: np.ndarray -> layers x RGBA colours
    - rho: float -> depth-latency Spearman rho
    - pvalue: float -> its p value
    - marker: str -> scatter marker
    - label: str | None -> legend label

OUTPUT:
    - None
"""
def plot_depth_latency_points(
        axis, latency_ms, layer_depths, layer_colors, rho, pvalue,
        marker="o", label=None,
        ):
    valid_layers = np.isfinite(latency_ms)
    axis.scatter(
        latency_ms[valid_layers], layer_depths[valid_layers],
        c=layer_colors[valid_layers], s=80, marker=marker,
        edgecolors="black", linewidths=0.8, label=label,
    )
    axis.axvline(0, color="0.6", linewidth=1, linestyle="--")
    axis.set_ylim(-0.05, 1.05)
    axis.text(
        0.04, 0.96, f"rho={rho:.2f}\np={pvalue:.2g}", transform=axis.transAxes,
        ha="left", va="top", bbox=dict(facecolor="white", edgecolor="black"), zorder=20,
    )
    return None
# EOF
