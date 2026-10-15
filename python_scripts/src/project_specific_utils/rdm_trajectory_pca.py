"""
PCA of RDM timeseries: every timepoint's stimulus RDM is one point in
stimulus-pair space, so a condition traces a trajectory through RDM space.
"""

from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA


"""
project_rdm_trajectories
Fit a PCA basis on the RDMs of one condition and project the RDM trajectories
of both conditions onto it.

INPUT:
    - static_rdms: np.ndarray -> static time x stimulus-pair distances
    - movie_rdms: np.ndarray -> movie time x stimulus-pair distances
    - basis_name: str -> 'static' or 'movie', the condition the PCA is fitted on
    - movie_fit_mask: np.ndarray | None -> movie samples used to fit the movie
        basis (e.g. excluding the onset transient); None uses all samples
    - n_components: int -> number of principal components
    - smoothing_bins: float -> Gaussian sigma in samples applied to the scores;
        0 = no smoothing

OUTPUT:
    - static_scores: np.ndarray -> static time x components
    - movie_scores: np.ndarray -> movie time x components
    - explained_variance_ratio: np.ndarray -> variance fraction of every component
        within the fitted condition
"""
def project_rdm_trajectories(
        static_rdms, movie_rdms, basis_name, movie_fit_mask=None,
        n_components=2, smoothing_bins=0,
        ):
    if basis_name == "static":
        fit_rdms = static_rdms
    elif basis_name == "movie":
        fit_rdms = movie_rdms if movie_fit_mask is None else movie_rdms[movie_fit_mask]
    else:
        raise ValueError("basis_name must be 'static' or 'movie'.")
    # end if basis_name

    # Timepoints are observations and stimulus pairs are features; PCA centres
    # on the fitted condition's mean RDM, so both trajectories share that origin.
    pca = PCA(n_components=n_components).fit(fit_rdms)
    static_scores = pca.transform(static_rdms)
    movie_scores = pca.transform(movie_rdms)
    if smoothing_bins > 0:
        # 'nearest' avoids pulling the trajectory ends toward zero.
        static_scores = gaussian_filter1d(static_scores, smoothing_bins, axis=0, mode="nearest")
        movie_scores = gaussian_filter1d(movie_scores, smoothing_bins, axis=0, mode="nearest")
    # end if smoothing
    return static_scores, movie_scores, pca.explained_variance_ratio_
# EOF


"""
plot_gradient_trajectory_3d
Draw a 3D trajectory as line segments coloured by time.

INPUT:
    - axis: mpl_toolkits.mplot3d.axes3d.Axes3D -> 3D axis to draw on
    - scores: np.ndarray -> time x 3 coordinates
    - times_ms: np.ndarray -> time of every sample, mapped through norm
    - cmap: matplotlib.colors.Colormap -> time colormap
    - norm: matplotlib.colors.Normalize -> time-to-colour normalization
    - linewidth: float -> line width
    - linestyle: str -> matplotlib line style ('solid' or 'dashed')

OUTPUT:
    - collection: mpl_toolkits.mplot3d.art3d.Line3DCollection -> drawn line
"""
def plot_gradient_trajectory_3d(
        axis, scores, times_ms, cmap, norm, linewidth=2.5, linestyle="solid",
        ):
    # Segment k joins samples k and k + 1 and takes its midpoint time's colour.
    segments = np.stack([scores[:-1, :3], scores[1:, :3]], axis=1)
    collection = Line3DCollection(
        segments, cmap=cmap, norm=norm, linewidth=linewidth, linestyle=linestyle,
    )
    collection.set_array((times_ms[:-1] + times_ms[1:]) / 2)
    axis.add_collection3d(collection)
    return collection
# EOF
