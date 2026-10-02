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
