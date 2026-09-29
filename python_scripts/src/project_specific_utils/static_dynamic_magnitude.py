import numpy as np
from scipy.stats import ttest_ind


"""
select_tail_indices
Return the indices of all values, or of the largest or smallest percentage,
from one ranking.

INPUT:
    - values: np.ndarray -> one-dimensional values ranked within one condition
    - subset: str -> "all", "top", or "bottom"
    - percent: float -> percentage retained for the top or bottom subset

OUTPUT:
    - selected_indices: np.ndarray -> indices of the retained values, ordered
      by ascending value
"""
def select_tail_indices(
        values: np.ndarray,
        subset: str,
        percent: float = 100,
        ) -> np.ndarray:
    order = np.argsort(np.asarray(values, dtype=np.float64).ravel(), kind="stable")
    if subset == "all":
        return order
    # end if subset == "all"
    if not 0 < percent <= 100:
        raise ValueError("percent must be in (0, 100].")
    # end if invalid percent
    count = int(np.ceil(order.size * percent / 100))
    if count < 2:
        raise ValueError("The selected percentage must retain at least two values.")
    # end if too few values
    if subset == "top":
        return order[-count:]
    elif subset == "bottom":
        return order[:count]
    # end if subset
    raise ValueError("subset must be 'all', 'top', or 'bottom'.")
# EOF


"""
select_tail_values
Select all values, or the largest or smallest percentage, from one ranking.

INPUT:
    - values: np.ndarray -> one-dimensional values ranked within one condition
    - subset: str -> "all", "top", or "bottom"
    - percent: float -> percentage retained for the top or bottom subset

OUTPUT:
    - selected_values: np.ndarray -> retained values in ascending order
"""
def select_tail_values(
        values: np.ndarray,
        subset: str,
        percent: float = 100,
        ) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).ravel()
    return values[select_tail_indices(values, subset, percent)]
# EOF



"""
rdm_tail_timecourses
Summarize the distances of every timepoint's RDM for all pairs and for that
timepoint's own top and bottom tails.

INPUT:
    - rdm_timeseries: np.ndarray -> time x stimulus-pair distances
    - top_percent: float -> percentage of largest distances in the top subset
    - bottom_percent: float -> percentage of smallest distances in the bottom subset

OUTPUT:
    - timecourses: dict[str, dict] -> per subset ("all", "top", "bottom"):
      label, count, and "mean" and "var" arrays with one value per timepoint
"""
def rdm_tail_timecourses(
        rdm_timeseries: np.ndarray,
        top_percent: float,
        bottom_percent: float,
        ) -> dict:
    subset_percents = {"all": 100, "top": top_percent, "bottom": bottom_percent}
    timecourses = {}
    for subset, percent in subset_percents.items():
        # Tails are re-ranked at every timepoint, so they may hold different pairs.
        selected = np.stack([
            select_tail_values(rdm, subset, percent) for rdm in rdm_timeseries
        ])
        timecourses[subset] = {
            "label": "all" if subset == "all" else f"{subset} {percent:g}%",
            "count": selected.shape[1],
            "mean": selected.mean(axis=1),
            "var": selected.var(axis=1),
        }
    # end for subset
    return timecourses
# EOF



"""
cross_temporal_tail_ttest
Welch t-test between the static RDM distances at every static time and the
dynamic RDM distances at every dynamic time, for all pairs and for each RDM's
own top and bottom tails. Positive t means larger static distances.

INPUT:
    - static_rdm_timeseries: np.ndarray -> static time x stimulus-pair distances
    - dynamic_rdm_timeseries: np.ndarray -> dynamic time x stimulus-pair distances
    - top_percent: float -> percentage of largest distances in the top subset
    - bottom_percent: float -> percentage of smallest distances in the bottom subset

OUTPUT:
    - t_matrices: dict[str, dict] -> per subset ("all", "top", "bottom"): label,
      count, and "t" (dynamic time x static time Welch t values)
"""
def cross_temporal_tail_ttest(
        static_rdm_timeseries: np.ndarray,
        dynamic_rdm_timeseries: np.ndarray,
        top_percent: float,
        bottom_percent: float,
        ) -> dict:
    static_tails = rdm_tail_timecourses(
        static_rdm_timeseries, top_percent, bottom_percent,
    )
    dynamic_tails = rdm_tail_timecourses(
        dynamic_rdm_timeseries, top_percent, bottom_percent,
    )
    t_matrices = {}
    for subset, static_summary in static_tails.items():
        dynamic_summary = dynamic_tails[subset]
        n_static = static_summary["count"]
        n_dynamic = dynamic_summary["count"]
        # Squared standard errors with unbiased (ddof=1) variances, as in ttest_ind.
        static_sem2 = static_summary["var"] * n_static / (n_static - 1) / n_static
        dynamic_sem2 = dynamic_summary["var"] * n_dynamic / (n_dynamic - 1) / n_dynamic
        # Broadcast to dynamic time x static time.
        t_matrices[subset] = {
            "label": static_summary["label"],
            "count": n_static,
            "t": (
                (static_summary["mean"][None, :] - dynamic_summary["mean"][:, None])
                / np.sqrt(static_sem2[None, :] + dynamic_sem2[:, None])
            ),
        }
    # end for subset
    return t_matrices
# EOF


"""
compare_static_dynamic_tails
Compare static and dynamic values on all entries and on each condition's own tails.

INPUT:
    - static_values: np.ndarray -> one-dimensional static values
    - dynamic_values: np.ndarray -> one-dimensional dynamic values
    - top_percent: float -> percentage of largest values in the top subset
    - bottom_percent: float -> percentage of smallest values in the bottom subset

OUTPUT:
    - comparison: dict[str, dict] -> per subset ("all", "top", "bottom"): label,
      static and dynamic mean, SD, count, and Welch t-test statistic and p-value
"""
def compare_static_dynamic_tails(
        static_values: np.ndarray,
        dynamic_values: np.ndarray,
        top_percent: float,
        bottom_percent: float,
        ) -> dict:
    subset_percents = {"all": 100, "top": top_percent, "bottom": bottom_percent}
    comparison = {}
    for subset, percent in subset_percents.items():
        # Each condition is ranked on its own, so tails may hold different items.
        static_selected = select_tail_values(static_values, subset, percent)
        dynamic_selected = select_tail_values(dynamic_values, subset, percent)
        # Welch's t-test: the samples are unpaired and their spreads can differ.
        t_statistic, p_value = ttest_ind(
            static_selected, dynamic_selected, equal_var=False,
        )
        comparison[subset] = {
            "label": "all" if subset == "all" else f"{subset} {percent:g}%",
            "mean": np.array([static_selected.mean(), dynamic_selected.mean()]),
            "std": np.array([static_selected.std(), dynamic_selected.std()]),
            "count": static_selected.size,
            "t": t_statistic,
            "p": p_value,
        }
    # end for subset
    return comparison
# EOF


"""
plot_static_dynamic_bars
Draw one mean +- SD bar panel per subset for a static-dynamic comparison.

INPUT:
    - axes: sequence[matplotlib.axes.Axes] -> one axis per subset
    - comparison: dict -> output of compare_static_dynamic_tails
    - condition_labels: tuple[str, str] -> static and dynamic bar labels
    - ylabel: str -> y-axis label
    - item_name: str -> name of the compared items, e.g. "pairs" or "stimuli"

OUTPUT:
    - None
"""
def plot_static_dynamic_bars(axes, comparison, condition_labels, ylabel, item_name):
    for axis, summary in zip(axes, comparison.values()):
        axis.bar(
            condition_labels, summary["mean"], yerr=summary["std"], capsize=6,
            color=["tab:blue", "tab:orange"],
        )
        axis.set(
            title=(
                f"{summary['label']} ({summary['count']} {item_name})\n"
                f"t={summary['t']:.2f}, p={summary['p']:.2g}"
            ),
            ylabel=ylabel,
        )
        axis.grid(alpha=0.2, axis="y")
    # end for axis, summary
# EOF
