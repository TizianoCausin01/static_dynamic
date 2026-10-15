import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial.distance import squareform

from .last_frame_decay import window_mean
from .split_half_rsa import compute_rdm_timeseries, rowwise_similarity


"""
reference_rdm_timecourse
Correlate every timepoint's RDM with one reference RDM.

INPUT:
    - rdm_timeseries: np.ndarray -> time x stimulus-pair distances
    - reference_rdm: np.ndarray -> stimulus-pair distances of the reference
    - rsa_metric: str -> correlation or spearman

OUTPUT:
    - similarity: np.ndarray -> time RDM similarity with the reference
"""
def reference_rdm_timecourse(rdm_timeseries, reference_rdm, rsa_metric="correlation"):
    repeated_reference = np.broadcast_to(reference_rdm, rdm_timeseries.shape)
    return rowwise_similarity(rdm_timeseries, repeated_reference, metric=rsa_metric)
# EOF


"""
subsampled_reference_timecourses
Recompute the reference-RDM timecourse on random stimulus subsets by indexing
the full square RDMs (the RDM metric itself is not recomputed per subset).

INPUT:
    - rdm_timeseries: np.ndarray -> time x stimulus-pair distances
    - reference_rdm: np.ndarray -> stimulus-pair distances of the reference
    - n_subsamples: int -> number of random subsets
    - subsample_fraction: float -> fraction of stimuli kept without replacement
    - rsa_metric: str -> correlation or spearman
    - random_seed: int -> generator seed

OUTPUT:
    - timecourses: np.ndarray -> subsets x time RDM similarity
"""
def subsampled_reference_timecourses(
        rdm_timeseries,
        reference_rdm,
        n_subsamples=100,
        subsample_fraction=0.8,
        rsa_metric="correlation",
        random_seed=0,
        ):
    # time x stimuli x stimuli and stimuli x stimuli square forms.
    square_rdms = np.stack([squareform(rdm) for rdm in rdm_timeseries])
    square_reference = squareform(reference_rdm)
    n_stimuli = square_reference.shape[0]
    n_keep = int(round(subsample_fraction * n_stimuli))
    upper_rows, upper_columns = np.triu_indices(n_keep, k=1)

    generator = np.random.default_rng(random_seed)
    timecourses = np.full((n_subsamples, rdm_timeseries.shape[0]), np.nan)
    for subsample_index in range(n_subsamples):
        kept = np.sort(generator.choice(n_stimuli, n_keep, replace=False))
        # Upper-triangle pairs of the kept stimuli only.
        rows, columns = kept[upper_rows], kept[upper_columns]
        timecourses[subsample_index] = reference_rdm_timecourse(
            square_rdms[:, rows, columns], square_reference[rows, columns], rsa_metric,
        )
    # end for subsample_index
    return timecourses
# EOF


"""
compute_last_frame_rdm_timecourses
Build the neural and visual RDM similarity with the last-frame RDM over time.

The neural reference is the RDM of the mean response inside
neural_reference_window_ms; the visual reference is the RDM of the designated
last frame. Each timecourse is also recomputed on stimulus subsets.

INPUT:
    - data: dict -> output of load_last_frame_decay_data
    - last_frame_time_ms: float -> designated final movie frame
    - neural_reference_window_ms: tuple[float, float] -> last-frame response window
    - rdm_metric: str -> create_RDM distance
    - rsa_metric: str -> correlation or spearman
    - n_subsamples: int -> stimulus subsets for uncertainty
    - subsample_fraction: float -> fraction of stimuli kept per subset
    - random_seed: int -> generator seed

OUTPUT:
    - timecourses: dict -> neural and visual times, curves, and subset curves
"""
def compute_last_frame_rdm_timecourses(
        data,
        last_frame_time_ms=2500,
        neural_reference_window_ms=(2500, 2600),
        rdm_metric="cosine_cnt",
        rsa_metric="correlation",
        n_subsamples=100,
        subsample_fraction=0.8,
        random_seed=0,
        ):
    neural_times_ms = np.arange(data["dynamic"].shape[1]) * 1000 / data["neural_fs"]
    visual_times_ms = (
        np.arange(data["visual_features"].shape[1]) * 1000 / data["visual_fs"]
    )
    visual_last_index = int(np.argmin(np.abs(visual_times_ms - last_frame_time_ms)))

    # Neural: time x pairs RDMs and the RDM of the window-averaged response.
    neural_rdms = compute_rdm_timeseries(data["dynamic"], rdm_metric)
    neural_reference = compute_rdm_timeseries(
        window_mean(data["dynamic"], neural_times_ms, neural_reference_window_ms)[:, np.newaxis, :],
        rdm_metric,
    )[0]
    # Visual: frames up to and including the last frame.
    visual_rdms = compute_rdm_timeseries(
        np.asarray(data["visual_features"][:, :visual_last_index + 1, :], dtype=np.float64),
        rdm_metric,
    )
    visual_reference = visual_rdms[visual_last_index]

    timecourses = {
        "neural_times_ms": neural_times_ms,
        "visual_times_ms": visual_times_ms[:visual_last_index + 1],
        "neural_reference_window_ms": tuple(neural_reference_window_ms),
        "visual_label": data["visual_label"],
    }
    for name, rdms, reference in (
            ("neural", neural_rdms, neural_reference),
            ("visual", visual_rdms, visual_reference),
            ):
        timecourses[name] = reference_rdm_timecourse(rdms, reference, rsa_metric)
        timecourses[f"{name}_subsamples"] = subsampled_reference_timecourses(
            rdms, reference, n_subsamples, subsample_fraction, rsa_metric, random_seed,
        )
    # end for name
    return timecourses
# EOF


# Candidate decay shapes of the similarity as a function of the lag L (ms)
# before the reference. Every model has a baseline b and an amplitude a.
DECAY_MODELS = {
    "linear": {
        "function": lambda lag, a, b: b + a * (1 - lag / 1000),
        "parameter_names": ("a", "b"),
    },
    "exponential": {
        "function": lambda lag, a, b, tau: b + a * np.exp(-lag / tau),
        "parameter_names": ("a", "b", "tau"),
    },
    "hyperbolic": {
        "function": lambda lag, a, b, tau: b + a / (1 + lag / tau),
        "parameter_names": ("a", "b", "tau"),
    },
    "power_law": {
        "function": lambda lag, a, b, tau, alpha: b + a * (1 + lag / tau) ** (-alpha),
        "parameter_names": ("a", "b", "tau", "alpha"),
    },
    "stretched_exponential": {
        "function": lambda lag, a, b, tau, beta: b + a * np.exp(-(lag / tau) ** beta),
        "parameter_names": ("a", "b", "tau", "beta"),
    },
    "double_exponential": {
        "function": lambda lag, a1, b, tau1, a2, tau2: (
            b + a1 * np.exp(-lag / tau1) + a2 * np.exp(-lag / tau2)
        ),
        "parameter_names": ("a1", "b", "tau1", "a2", "tau2"),
    },
}


"""
decay_initial_guess
Starting values and bounds for one decay model, scaled to the data.

INPUT:
    - model_name: str -> key of DECAY_MODELS
    - lags_ms: np.ndarray -> fitted lags in ms
    - values: np.ndarray -> fitted similarity values

OUTPUT:
    - p0: list[float] -> initial parameters
    - bounds: tuple[list, list] -> lower and upper parameter bounds
"""
def decay_initial_guess(model_name, lags_ms, values):
    max_lag = lags_ms.max()
    amplitude = max(values[np.argmin(lags_ms)] - values[np.argmax(lags_ms)], 1e-3)
    baseline = values[np.argmax(lags_ms)]
    tau = max_lag / 4
    # Taus are bounded to the sampled lag range (with margin) so that they
    # remain identifiable; amplitudes are positive because similarity decays.
    guesses = {
        "linear": ([amplitude, baseline], ([-5, -2], [5, 2])),
        "exponential": (
            [amplitude, baseline, tau],
            ([0, -2, 1], [5, 2, 10 * max_lag]),
        ),
        "hyperbolic": (
            [amplitude, baseline, tau],
            ([0, -2, 1], [5, 2, 10 * max_lag]),
        ),
        "power_law": (
            [amplitude, baseline, tau, 1],
            ([0, -2, 1, 0.05], [50, 2, 10 * max_lag, 10]),
        ),
        "stretched_exponential": (
            [amplitude, baseline, tau, 1],
            ([0, -2, 1, 0.1], [5, 2, 10 * max_lag, 5]),
        ),
        "double_exponential": (
            [amplitude / 2, baseline, tau / 4, amplitude / 2, 2 * tau],
            ([0, -2, 1, 0, 1], [5, 2, 10 * max_lag, 5, 10 * max_lag]),
        ),
    }
    return guesses[model_name]
# EOF


"""
half_decay_lag
Lag at which a fitted curve has lost half of its drop over the fitted range.

INPUT:
    - function: callable -> fitted decay function of lag
    - parameters: np.ndarray -> fitted parameters
    - max_lag_ms: float -> largest fitted lag

OUTPUT:
    - lag_ms: float -> half-decay lag, NaN if the curve does not decay
"""
def half_decay_lag(function, parameters, max_lag_ms):
    dense_lags = np.linspace(0, max_lag_ms, 2001)
    curve = function(dense_lags, *parameters)
    midpoint = (curve[0] + curve[-1]) / 2
    if curve[0] <= curve[-1]:
        return np.nan
    # end if no decay
    return float(dense_lags[np.argmax(curve <= midpoint)])
# EOF


"""
fit_decay_models
Fit every DECAY_MODELS shape to one similarity-versus-lag curve and rank them.

AIC/BIC assume independent Gaussian residuals; consecutive samples are
autocorrelated, so the absolute evidence is overstated while the ranking
remains informative.

INPUT:
    - lags_ms: np.ndarray -> lags before the reference, in ms
    - values: np.ndarray -> similarity at every lag
    - model_names: tuple[str, ...] | None -> subset of DECAY_MODELS to fit

OUTPUT:
    - fits: dict -> per-model parameters, R^2, AIC, BIC, Akaike weight,
      half-decay lag, and fitted values
"""
def fit_decay_models(lags_ms, values, model_names=None):
    model_names = model_names or tuple(DECAY_MODELS)
    finite = np.isfinite(values)
    lags_ms, values = lags_ms[finite], values[finite]
    n_samples = values.size
    total_sum_squares = np.sum((values - values.mean()) ** 2)

    fits = {}
    for model_name in model_names:
        function = DECAY_MODELS[model_name]["function"]
        p0, bounds = decay_initial_guess(model_name, lags_ms, values)
        try:
            parameters, _ = curve_fit(
                function, lags_ms, values, p0=p0, bounds=bounds, maxfev=50000,
            )
        except (RuntimeError, ValueError):
            continue
        # end try
        fitted = function(lags_ms, *parameters)
        residual_sum_squares = np.sum((values - fitted) ** 2)
        n_parameters = len(parameters) + 1  # +1 for the residual variance
        log_term = n_samples * np.log(residual_sum_squares / n_samples)
        fits[model_name] = {
            "parameters": parameters,
            "parameter_names": DECAY_MODELS[model_name]["parameter_names"],
            "r_squared": 1 - residual_sum_squares / total_sum_squares,
            "aic": log_term + 2 * n_parameters,
            "bic": log_term + np.log(n_samples) * n_parameters,
            "half_decay_lag_ms": half_decay_lag(function, parameters, lags_ms.max()),
            "fitted": fitted,
            "lags_ms": lags_ms,
        }
    # end for model_name

    # Akaike weights: relative likelihood of each model within this set.
    aics = np.array([fit["aic"] for fit in fits.values()])
    weights = np.exp(-(aics - aics.min()) / 2)
    weights /= weights.sum()
    for fit, weight in zip(fits.values(), weights):
        fit["akaike_weight"] = weight
    # end for fit
    return fits
# EOF


"""
curve_lags
Select the samples inside the fit window and express them as lags before the
reference time.

INPUT:
    - times_ms: np.ndarray -> sample times
    - values: np.ndarray -> (... x time) similarity values
    - fit_window_ms: tuple[float, float] -> [start, end) sample times kept
    - reference_time_ms: float -> lag zero

OUTPUT:
    - lags_ms: np.ndarray -> lags of the kept samples
    - kept_values: np.ndarray -> values at the kept samples
"""
def curve_lags(times_ms, values, fit_window_ms, reference_time_ms):
    kept = (times_ms >= fit_window_ms[0]) & (times_ms < fit_window_ms[1])
    return reference_time_ms - times_ms[kept], values[..., kept]
# EOF


"""
plot_rdm_decay_fits
Plot one RDM-similarity curve against lag with every fitted decay shape.

INPUT:
    - axis: matplotlib.axes.Axes -> target axis
    - lags_ms: np.ndarray -> lags before the reference
    - values: np.ndarray -> similarity at every lag
    - subsample_values: np.ndarray -> subsets x lags similarity for a 95% band
    - fits: dict -> output of fit_decay_models
    - title: str -> axis title

OUTPUT:
    - None
"""
def plot_rdm_decay_fits(axis, lags_ms, values, subsample_values, fits, title):
    band = np.nanpercentile(subsample_values, [2.5, 97.5], axis=0)
    axis.fill_between(lags_ms, *band, color="0.85", label="80% stimulus subsets 95%")
    axis.plot(lags_ms, values, color="k", linewidth=1.2, label="Observed")
    best_model = min(fits, key=lambda name: fits[name]["aic"])
    for model_name, fit in fits.items():
        axis.plot(
            fit["lags_ms"], fit["fitted"],
            linewidth=2.2 if model_name == best_model else 1,
            label=(
                f"{model_name}: R²={fit['r_squared']:.3f}, "
                f"w={fit['akaike_weight']:.2f}"
            ),
        )
    # end for model_name
    axis.set(
        xlabel="Lag before last frame (ms)", ylabel="RDM r with last-frame RDM",
        title=title, xscale="log",
    )
    axis.legend(frameon=False, fontsize=6)
    axis.grid(alpha=0.2)
# EOF
