import argparse
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import sys

import matplotlib
import numpy as np
import yaml


matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Example from the project root, after running the per-monkey analysis:
# .venv/bin/python \
#     python_scripts/scripts/run_static_dynamic_model_hierarchy_across_monkeys.py


ENV = os.getenv("MY_ENV", "tiziano_mac_mini")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

with open(PROJECT_ROOT / "config.yaml", "r") as f:
    config = yaml.safe_load(f)
# end with open

paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
sys.path.append(paths["useful_stuff_path"])

from image_processing.model_zoo import MODEL_ZOO
from scipy.stats import spearmanr


# Image and movie sessions per monkey, matching the RSA sweep script.
MONKEY_SESSIONS = {
    "baby1": ("baby1_260718to27", "baby1_260716to24"),
    "red": ("red_20260726to27", "red_20260720to24"),
    "paul": ("paul_20260901", "paul_20260831to0902"),
}

ANALYSES = ("image", "first_frame", "last_frame")

ANALYSIS_LABELS = {
    "image": "Static image response",
    "first_frame": "Movie first-frame slice",
    "last_frame": "Movie last-frame slice",
}

MONKEY_MARKERS = {"baby1": "o", "red": "s", "paul": "^"}
MONKEY_COLORS = {"baby1": "#3474BD", "red": "#C4564C", "paul": "#11A579"}


@dataclass
class Cfg:
    monkeys: list[str] = field(default_factory=lambda: list(MONKEY_SESSIONS))
    results_root: str | None = None
    output_dir: str | None = None
    figs_dir: str | None = None
    latency_name: str = "centroid_latency_ms"
    # Models needing this many monkeys before they enter the summary figures.
    min_monkeys: int = 2
    dpi: int = 150
# EOF


"""
parse_args
Parse the monkey selection and output locations for the cross-monkey summary.

OUTPUT:
    - cfg: Cfg -> validated configuration
"""
def parse_args() -> Cfg:
    parser = argparse.ArgumentParser(
        description=(
            "Compare per-monkey model-hierarchy results and report how "
            "consistently each model reproduces the depth-latency gradient."
        )
    )
    parser.add_argument(
        "--monkeys", nargs="+", choices=tuple(MONKEY_SESSIONS),
        help="Monkeys to combine; defaults to all with saved results.",
    )
    parser.add_argument("--results_root")
    parser.add_argument("--output_dir")
    parser.add_argument("--figs_dir")
    parser.add_argument("--latency_name", default=Cfg.latency_name)
    parser.add_argument("--min_monkeys", type=int, default=Cfg.min_monkeys)
    args = parser.parse_args()

    cfg = Cfg()
    for field_name, value in vars(args).items():
        if value is not None:
            setattr(cfg, field_name, value)
        # end if the user supplied the argument
    # end for field_name
    if cfg.min_monkeys < 1:
        parser.error("--min_monkeys must be positive.")
    # end if invalid min_monkeys
    return cfg
# EOF


"""
resolve_cfg_paths
Fill in the default result and figure directories.

INPUT:
    - cfg: Cfg -> user configuration

OUTPUT:
    - cfg: Cfg -> configuration with explicit directories
"""
def resolve_cfg_paths(cfg: Cfg) -> Cfg:
    cfg.results_root = str(Path(
        cfg.results_root
        or PROJECT_ROOT / "results" / "static_dynamic_model_hierarchy"
    ).expanduser())
    cfg.output_dir = str(Path(
        cfg.output_dir or Path(cfg.results_root) / "across_monkeys"
    ).expanduser())
    cfg.figs_dir = str(Path(
        cfg.figs_dir
        or Path(paths["data_path"]) / "results" / "figs_model_hierarchy"
        / "across_monkeys"
    ).expanduser())
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.figs_dir).mkdir(parents=True, exist_ok=True)
    return cfg
# EOF


"""
load_monkey_summaries
Read every monkey's saved summary JSON, skipping the ones not yet computed.

INPUT:
    - cfg: Cfg -> resolved configuration

OUTPUT:
    - summaries: dict -> {monkey: parsed summary JSON}
"""
def load_monkey_summaries(cfg: Cfg) -> dict:
    summaries = {}
    for monkey in cfg.monkeys:
        static_name, dynamic_name = MONKEY_SESSIONS[monkey]
        summary_path = (
            Path(cfg.results_root) / f"{dynamic_name}_vs_{static_name}"
            / "model_hierarchy_summary.json"
        )
        if not summary_path.is_file():
            print(f"skipping {monkey}: no summary at {summary_path}", flush=True)
            continue
        # end if the monkey has no saved summary
        with open(summary_path) as summary_file:
            summaries[monkey] = json.load(summary_file)
        # end with open
        print(f"loaded {monkey} from {summary_path}", flush=True)
    # end for monkey
    if not summaries:
        raise FileNotFoundError(
            "No per-monkey summary was found; run the per-monkey analysis first."
        )
    # end if nothing loaded
    return summaries
# EOF


"""
collect_records
Index every monkey's records by analysis and model for direct comparison.

INPUT:
    - summaries: dict -> {monkey: summary JSON}

OUTPUT:
    - records: dict -> {analysis: {model_name: {monkey: record}}}
    - model_order: list[str] -> registry order of the models that appear
"""
def collect_records(summaries: dict):
    records = {analysis: {} for analysis in ANALYSES}
    seen_models = set()
    for monkey, summary in summaries.items():
        for record in summary["records"]:
            analysis = record["analysis"]
            model_name = record["model_name"]
            records[analysis].setdefault(model_name, {})[monkey] = record
            seen_models.add(model_name)
        # end for record
    # end for monkey
    model_order = [name for name in MODEL_ZOO if name in seen_models]
    return records, model_order
# EOF


"""
plot_across_monkeys
Plot one per-model quantity for every monkey side by side, per analysis.

INPUT:
    - records: dict -> {analysis: {model_name: {monkey: record}}}
    - model_order: list[str] -> models in registry order
    - cfg: Cfg -> plotting configuration
    - value_key: str -> record field plotted on the y axis
    - y_label: str -> axis label
    - figure_name: str -> output filename stem
    - hierarchical_only: bool -> drop the single-layer baselines

OUTPUT:
    - figure_path: Path -> saved figure
"""
def plot_across_monkeys(
        records: dict,
        model_order: list,
        cfg: Cfg,
        value_key: str,
        y_label: str,
        figure_name: str,
        hierarchical_only: bool = True,
        ) -> Path:
    monkeys = sorted(
        {monkey for by_model in records["image"].values() for monkey in by_model}
    )
    # Models are ordered by their mean correspondence so the reading order is
    # the same in every panel and matches the per-monkey report.
    def mean_correspondence(model_name):
        values = [
            record["best_peak_similarity"]
            for record in records["image"].get(model_name, {}).values()
        ]
        return -np.mean(values) if values else 0.0
    # EOF

    plotted_models = [
        name for name in model_order
        if len(records["image"].get(name, {})) >= cfg.min_monkeys
        and (
            not hierarchical_only
            or min(
                record["n_layers"]
                for record in records["image"][name].values()
            ) >= 3
        )
    ]
    plotted_models.sort(key=mean_correspondence)

    figure, axes = plt.subplots(
        len(ANALYSES), 1,
        figsize=(1.05 * len(plotted_models) + 4, 3.5 * len(ANALYSES)),
        squeeze=False, sharex=True,
    )
    positions = np.arange(len(plotted_models))
    for axis, analysis in zip(axes[:, 0], ANALYSES):
        means = []
        for position, model_name in zip(positions, plotted_models):
            values = []
            for monkey in monkeys:
                record = records[analysis].get(model_name, {}).get(monkey)
                if record is None or record[value_key] is None:
                    continue
                # end if the monkey lacks this model
                value = record[value_key]
                if value != value:
                    continue
                # end if the value is NaN
                values.append(value)
                axis.scatter(
                    position, value, s=70,
                    marker=MONKEY_MARKERS.get(monkey, "o"),
                    color=MONKEY_COLORS.get(monkey, "0.4"),
                    edgecolors="black", linewidths=0.6, zorder=3,
                    label=monkey if position == 0 else None,
                )
            # end for monkey
            means.append(np.mean(values) if values else np.nan)
        # end for position, model_name
        # A short horizontal tick marks the across-monkey mean per model.
        axis.scatter(
            positions, means, marker="_", s=520, color="0.25",
            linewidths=1.8, zorder=2,
        )
        axis.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        axis.set_title(ANALYSIS_LABELS[analysis], fontsize=11)
        axis.set_ylabel(y_label, fontsize=10)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    # end for axis, analysis

    axes[-1, 0].set_xticks(positions)
    axes[-1, 0].set_xticklabels(
        [
            records["image"][name][
                next(iter(records["image"][name]))
            ]["label"]
            for name in plotted_models
        ],
        rotation=35, ha="right", fontsize=9,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper right", frameon=False, fontsize=9,
        ncol=len(labels),
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure_path = Path(cfg.figs_dir) / f"{figure_name}.png"
    figure.savefig(figure_path, dpi=cfg.dpi, bbox_inches="tight")
    figure.savefig(figure_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return figure_path
# EOF


"""
ranking_agreement
Rank-correlate the model ordering between every pair of monkeys.

A model set that reproduces across animals should order the models the same
way; this is the quantity that says whether a per-monkey result generalizes.

INPUT:
    - records: dict -> {analysis: {model_name: {monkey: record}}}
    - value_key: str -> record field whose ordering is compared

OUTPUT:
    - agreement: list[dict] -> one entry per analysis and monkey pair
"""
def ranking_agreement(records: dict, value_key: str) -> list:
    agreement = []
    for analysis in ANALYSES:
        by_model = records[analysis]
        monkeys = sorted(
            {monkey for values in by_model.values() for monkey in values}
        )
        for first_index, first_monkey in enumerate(monkeys):
            for second_monkey in monkeys[first_index + 1:]:
                shared_models = [
                    name for name, values in by_model.items()
                    if first_monkey in values and second_monkey in values
                    and values[first_monkey]["n_layers"] >= 3
                ]
                first_values, second_values = [], []
                for name in shared_models:
                    first_value = by_model[name][first_monkey][value_key]
                    second_value = by_model[name][second_monkey][value_key]
                    if (
                            first_value is None or second_value is None
                            or first_value != first_value
                            or second_value != second_value
                            ):
                        continue
                    # end if either value is missing
                    first_values.append(first_value)
                    second_values.append(second_value)
                # end for name
                if len(first_values) < 3:
                    continue
                # end if too few shared models
                result = spearmanr(first_values, second_values)
                agreement.append({
                    "analysis": analysis,
                    "value_key": value_key,
                    "monkeys": [first_monkey, second_monkey],
                    "n_models": len(first_values),
                    "rho": float(result.statistic),
                    "pvalue": float(result.pvalue),
                })
            # end for second_monkey
        # end for first_index, first_monkey
    # end for analysis
    return agreement
# EOF


"""
build_combined_records
Flatten the per-monkey values into one record per analysis and model.

INPUT:
    - records: dict -> {analysis: {model_name: {monkey: record}}}
    - model_order: list[str] -> models in registry order
    - latency_name: str -> latency measure behind the temporal score

OUTPUT:
    - combined: list[dict] -> per-model summary across monkeys
"""
def build_combined_records(
        records: dict, model_order: list, latency_name: str,
        ) -> list:
    combined = []
    for analysis in ANALYSES:
        for model_name in model_order:
            by_monkey = records[analysis].get(model_name, {})
            if not by_monkey:
                continue
            # end if the model is absent everywhere
            any_record = next(iter(by_monkey.values()))
            scores = {
                monkey: record[f"{latency_name}_rho"]
                for monkey, record in by_monkey.items()
            }
            correspondences = {
                monkey: record["best_peak_similarity"]
                for monkey, record in by_monkey.items()
            }
            finite_scores = [v for v in scores.values() if v == v]
            combined.append({
                "analysis": analysis,
                "model_name": model_name,
                "label": any_record["label"],
                "family": any_record["family"],
                "n_layers": any_record["n_layers"],
                "n_monkeys": len(by_monkey),
                "temporal_score_by_monkey": scores,
                "correspondence_by_monkey": correspondences,
                "mean_temporal_score": (
                    float(np.mean(finite_scores)) if finite_scores else None
                ),
                "min_temporal_score": (
                    float(np.min(finite_scores)) if finite_scores else None
                ),
                "mean_correspondence": float(
                    np.mean(list(correspondences.values()))
                ),
            })
        # end for model_name
    # end for analysis
    return combined
# EOF


"""
main
Combine the per-monkey summaries into cross-monkey figures and a JSON table.
"""
def main() -> None:
    cfg = resolve_cfg_paths(parse_args())
    summaries = load_monkey_summaries(cfg)
    records, model_order = collect_records(summaries)

    figure_paths = [
        plot_across_monkeys(
            records, model_order, cfg,
            value_key=f"{cfg.latency_name}_rho",
            y_label="temporal score (rho)",
            figure_name=f"temporal_score_across_monkeys_{cfg.latency_name}",
        ),
        plot_across_monkeys(
            records, model_order, cfg,
            value_key="best_peak_similarity",
            y_label=r"best-layer peak RSA ($\rho$)",
            figure_name="correspondence_across_monkeys",
            hierarchical_only=False,
        ),
    ]

    combined = build_combined_records(records, model_order, cfg.latency_name)
    agreement = (
        ranking_agreement(records, "best_peak_similarity")
        + ranking_agreement(records, f"{cfg.latency_name}_rho")
    )
    output_path = Path(cfg.output_dir) / "across_monkeys_summary.json"
    with open(output_path, "w") as summary_file:
        json.dump(
            {
                "config": asdict(cfg),
                "monkeys": sorted(summaries),
                "noise_ceilings": {
                    monkey: summary.get("noise_ceilings", {})
                    for monkey, summary in summaries.items()
                },
                "records": combined,
                "ranking_agreement": agreement,
            },
            summary_file, indent=2, sort_keys=True,
        )
    # end with open

    print(f"\nSaved {len(figure_paths)} figures to {cfg.figs_dir}")
    print(f"Saved summary to {output_path}")
    for entry in agreement:
        print(
            f"  {entry['analysis']:<12} {entry['value_key']:<28} "
            f"{'/'.join(entry['monkeys']):<12} rho={entry['rho']:+.2f} "
            f"(n={entry['n_models']})"
        )
    # end for entry
    return None
# EOF


if __name__ == "__main__":
    main()
# EOF
