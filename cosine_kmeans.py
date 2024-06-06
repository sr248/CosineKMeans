"""K-means clustering."""
import scipy.sparse as sp
import warnings
from abc import ABC, abstractmethod
from numbers import Integral, Real
import numpy as np

import sys
sys.path.append('../')
from lib.cosine_kmeans._cosine_k_means_common import _cosine_inertia_sparse, _cosine_inertia_dense
from lib.cosine_kmeans._cosine_k_means_lloyd import cosine_lloyd_iter_chunked_sparse, cosine_lloyd_iter_chunked_dense

from sklearn.cluster import _kmeans
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster._kmeans import *
from sklearn.cluster._kmeans import (
    _openmp_effective_n_threads,
    _is_same_clustering,
    _check_sample_weight,
    _is_arraylike_not_scalar,
    _fit_context,
)

@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "n_clusters": [Interval(Integral, 1, None, closed="left")],
        "sample_weight": ["array-like", None],
        "x_squared_norms": ["array-like", None],
        "random_state": ["random_state"],
        "n_local_trials": [Interval(Integral, 1, None, closed="left"), None],
    },
    prefer_skip_nested_validation=True,
)

def _cosine_kmeans_plusplus(
    X, n_clusters, *, x_squared_norms, sample_weight, random_state, n_local_trials=None
):
    """Computational component for initialization of n_clusters by
    k-means++. Prior validation of data is assumed.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The data to pick seeds for.

    n_clusters : int
        The number of seeds to choose.

    sample_weight : ndarray of shape (n_samples,)
        The weights for each observation in `X`.

    x_squared_norms : ndarray of shape (n_samples,)
        Squared Euclidean norm of each data point.

    random_state : RandomState instance
        The generator used to initialize the centers.
        See :term:`Glossary <random_state>`.

    n_local_trials : int, default=None
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.

    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        The initial centers for k-means.

    indices : ndarray of shape (n_clusters,)
        The index location of the chosen centers in the data array X. For a
        given index and center, X[index] = center.
    """
    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly and track index of point
    center_id = random_state.choice(n_samples, p=sample_weight / sample_weight.sum())
    indices = np.full(n_clusters, -1, dtype=int)
    if sp.issparse(X):
        centers[0] = X[[center_id]].toarray()
    else:
        centers[0] = X[center_id]
    indices[0] = center_id

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = cosine_distances(centers[0, np.newaxis], X)
    current_pot = closest_dist_sq @ sample_weight

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.uniform(size=n_local_trials) * current_pot
        candidate_ids = np.searchsorted(
            stable_cumsum(sample_weight * closest_dist_sq), rand_vals
        )
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = cosine_distances(X[candidate_ids], X)

        # update closest distances squared and potential for each candidate
        np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidates_pot = distance_to_candidates @ sample_weight.reshape(-1, 1)

        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        if sp.issparse(X):
            centers[c] = X[[best_candidate]].toarray()
        else:
            centers[c] = X[best_candidate]
        indices[c] = best_candidate

    return centers, indices

def _cosine_kmeans_single_lloyd(
    X,
    sample_weight,
    centers_init,
    max_iter=300,
    verbose=False,
    tol=1e-4,
    n_threads=1,
):
    """A single run of k-means lloyd, assumes preparation completed prior.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The observations to cluster. If sparse matrix, must be in CSR format.

    sample_weight : ndarray of shape (n_samples,)
        The weights for each observation in X.

    centers_init : ndarray of shape (n_clusters, n_features)
        The initial centers.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm to run.

    verbose : bool, default=False
        Verbosity mode

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
        It's not advised to set `tol=0` since convergence might never be
        declared due to rounding errors. Use a very small number instead.

    n_threads : int, default=1
        The number of OpenMP threads to use for the computation. Parallelism is
        sample-wise on the main cython loop which assigns each sample to its
        closest center.

    Returns
    -------
    centroid : ndarray of shape (n_clusters, n_features)
        Centroids found at the last iteration of k-means.

    label : ndarray of shape (n_samples,)
        label[i] is the code or index of the centroid the
        i'th observation is closest to.

    inertia : float
        The final value of the inertia criterion (sum of squared distances to
        the closest centroid for all observations in the training set).

    n_iter : int
        Number of iterations run.
    """
    n_clusters = centers_init.shape[0]

    # Buffers to avoid new allocations at each iteration.
    centers = centers_init
    centers_new = np.zeros_like(centers)
    labels = np.full(X.shape[0], -1, dtype=np.int32)
    labels_old = labels.copy()
    weight_in_clusters = np.zeros(n_clusters, dtype=X.dtype)
    center_shift = np.zeros(n_clusters, dtype=X.dtype)

    if sp.issparse(X):
        lloyd_iter = cosine_lloyd_iter_chunked_sparse
        _inertia = _cosine_inertia_sparse
    else:
        lloyd_iter = cosine_lloyd_iter_chunked_dense
        _inertia = _cosine_inertia_dense

    strict_convergence = False

    # Threadpoolctl context to limit the number of threads in second level of
    # nested parallelism (i.e. BLAS) to avoid oversubscription.
    with threadpool_limits(limits=1, user_api="blas"):
        for i in range(max_iter):
            lloyd_iter(
                X,
                sample_weight,
                centers,
                centers_new,
                weight_in_clusters,
                labels,
                center_shift,
                n_threads,
            )

            if verbose:
                inertia = _inertia(X, sample_weight, centers, labels, n_threads)
                print(f"Iteration {i}, inertia {inertia}.")

            centers, centers_new = centers_new, centers

            if np.array_equal(labels, labels_old):
                # First check the labels for strict convergence.
                if verbose:
                    print(f"Converged at iteration {i}: strict convergence.")
                strict_convergence = True
                break
            else:
                # No strict convergence, check for tol based convergence.
                center_shift_tot = (center_shift**2).sum()
                if center_shift_tot <= tol:
                    if verbose:
                        print(
                            f"Converged at iteration {i}: center shift "
                            f"{center_shift_tot} within tolerance {tol}."
                        )
                    break

            labels_old[:] = labels

        if not strict_convergence:
            # rerun E-step so that predicted labels match cluster centers
            lloyd_iter(
                X,
                sample_weight,
                centers,
                centers,
                weight_in_clusters,
                labels,
                center_shift,
                n_threads,
                update_centers=False,
            )

    inertia = _inertia(X, sample_weight, centers, labels, n_threads)

    return labels, inertia, centers, i + 1


def _cosine_labels_inertia(X, sample_weight, centers, n_threads=1, return_inertia=True):
    """E step of the K-means EM algorithm.

    Compute the labels and the inertia of the given samples and centers.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The input samples to assign to the labels. If sparse matrix, must
        be in CSR format.

    sample_weight : ndarray of shape (n_samples,)
        The weights for each observation in X.

    x_squared_norms : ndarray of shape (n_samples,)
        Precomputed squared euclidean norm of each data point, to speed up
        computations.

    centers : ndarray of shape (n_clusters, n_features)
        The cluster centers.

    n_threads : int, default=1
        The number of OpenMP threads to use for the computation. Parallelism is
        sample-wise on the main cython loop which assigns each sample to its
        closest center.

    return_inertia : bool, default=True
        Whether to compute and return the inertia.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        The resulting assignment.

    inertia : float
        Sum of squared distances of samples to their closest cluster center.
        Inertia is only returned if return_inertia is True.
    """
    n_samples = X.shape[0]
    n_clusters = centers.shape[0]

    labels = np.full(n_samples, -1, dtype=np.int32)
    center_shift = np.zeros(n_clusters, dtype=centers.dtype)

    if sp.issparse(X):
        _labels = cosine_lloyd_iter_chunked_sparse
        _inertia = _cosine_inertia_sparse
    else:
        _labels = cosine_lloyd_iter_chunked_dense
        _inertia = _cosine_inertia_dense

    _labels(
        X,
        sample_weight,
        centers,
        centers_new=None,
        weight_in_clusters=None,
        labels=labels,
        center_shift=center_shift,
        n_threads=n_threads,
        update_centers=False,
    )

    if return_inertia:
        inertia = _inertia(X, sample_weight, centers, labels, n_threads)
        return labels, inertia

    return labels


def _cosine_labels_inertia_threadpool_limit(
    X, sample_weight, centers, n_threads=1, return_inertia=True
):
    """Same as _labels_inertia but in a threadpool_limits context."""
    with threadpool_limits(limits=1, user_api="blas"):
        result = _cosine_labels_inertia(X, sample_weight, centers, n_threads, return_inertia)

    return result


class CosineKMeans(_kmeans._BaseKMeans):
    _parameter_constraints: dict = {
        **_kmeans._BaseKMeans._parameter_constraints,
        "copy_x": ["boolean"],
        "algorithm": [StrOptions({"lloyd"})],
    }

    def __init__(
        self,
        n_clusters=8,
        *,
        init="k-means++",
        n_init="auto",
        max_iter=300,
        tol=1e-4,
        verbose=0,
        random_state=None,
        copy_x=True,
        algorithm="lloyd",
    ):
        super().__init__(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
        )

        self.copy_x = copy_x
        self.algorithm = algorithm

    def _check_params_vs_input(self, X):
        super()._check_params_vs_input(X, default_n_init=10)

        self._algorithm = self.algorithm
        # if self._algorithm == "elkan" and self.n_clusters == 1:
        #     warnings.warn(
        #         (
        #             "algorithm='elkan' doesn't make sense for a single "
        #             "cluster. Using 'lloyd' instead."
        #         ),
        #         RuntimeWarning,
        #     )
        #     self._algorithm = "lloyd"

    def _warn_mkl_vcomp(self, n_active_threads):
        """Warn when vcomp and mkl are both present"""
        warnings.warn(
            "KMeans is known to have a memory leak on Windows "
            "with MKL, when there are less chunks than available "
            "threads. You can avoid it by setting the environment"
            f" variable OMP_NUM_THREADS={n_active_threads}."
        )
    
    def _init_centroids(
        self,
        X,
        x_squared_norms,
        init,
        random_state,
        sample_weight,
        init_size=None,
        n_centroids=None,
    ):
        """Compute the initial centroids.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        x_squared_norms : ndarray of shape (n_samples,)
            Squared euclidean norm of each data point. Pass it if you have it
            at hands already to avoid it being recomputed here.

        init : {'k-means++', 'random'}, callable or ndarray of shape \
                (n_clusters, n_features)
            Method for initialization.

        random_state : RandomState instance
            Determines random number generation for centroid initialization.
            See :term:`Glossary <random_state>`.

        sample_weight : ndarray of shape (n_samples,)
            The weights for each observation in X. `sample_weight` is not used
            during initialization if `init` is a callable or a user provided
            array.

        init_size : int, default=None
            Number of samples to randomly sample for speeding up the
            initialization (sometimes at the expense of accuracy).

        n_centroids : int, default=None
            Number of centroids to initialize.
            If left to 'None' the number of centroids will be equal to
            number of clusters to form (self.n_clusters).

        Returns
        -------
        centers : ndarray of shape (n_clusters, n_features)
            Initial centroids of clusters.
        """
        n_samples = X.shape[0]
        n_clusters = self.n_clusters if n_centroids is None else n_centroids

        if init_size is not None and init_size < n_samples:
            init_indices = random_state.randint(0, n_samples, init_size)
            X = X[init_indices]
            x_squared_norms = x_squared_norms[init_indices]
            n_samples = X.shape[0]
            sample_weight = sample_weight[init_indices]

        if isinstance(init, str) and init == "k-means++":
            centers, _ = _cosine_kmeans_plusplus(
                X,
                n_clusters,
                random_state=random_state,
                x_squared_norms=x_squared_norms,
                sample_weight=sample_weight,
            )
        elif isinstance(init, str) and init == "random":
            seeds = random_state.choice(
                n_samples,
                size=n_clusters,
                replace=False,
                p=sample_weight / sample_weight.sum(),
            )
            centers = X[seeds]
        elif _is_arraylike_not_scalar(self.init):
            centers = init
        elif callable(init):
            centers = init(X, n_clusters, random_state=random_state)
            centers = check_array(centers, dtype=X.dtype, copy=False, order="C")
            self._validate_center_shape(X, centers)

        if sp.issparse(centers):
            centers = centers.toarray()
        
        return centers

    def predict(self, X, sample_weight="deprecated"):
        """Predict the closest cluster each sample in X belongs to.

        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

            .. deprecated:: 1.3
               The parameter `sample_weight` is deprecated in version 1.3
               and will be removed in 1.5.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self)

        X = self._check_test_data(X)
        if not (isinstance(sample_weight, str) and sample_weight == "deprecated"):
            warnings.warn(
                (
                    "'sample_weight' was deprecated in version 1.3 and "
                    "will be removed in 1.5."
                ),
                FutureWarning,
            )
            sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        else:
            sample_weight = _check_sample_weight(None, X, dtype=X.dtype)

        labels = _cosine_labels_inertia_threadpool_limit(
            X,
            sample_weight,
            self.cluster_centers_,
            n_threads=self._n_threads,
            return_inertia=False,
        )

        return labels

    def _transform(self, X):
        """Guts of transform method; no input validation."""
        return cosine_distances(X, self.cluster_centers_)

    def score(self, X, y=None, sample_weight=None):
        """Opposite of the value of X on the K-means objective.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        score : float
            Opposite of the value of X on the K-means objective.
        """
        check_is_fitted(self)

        X = self._check_test_data(X)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        _, scores = _cosine_labels_inertia_threadpool_limit(
            X, sample_weight, self.cluster_centers_, self._n_threads
        )
        return -scores

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None, sample_weight=None):
        X = self._validate_data(
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            copy=self.copy_x,
            accept_large_sparse=False,
        )

        self._check_params_vs_input(X)

        random_state = check_random_state(self.random_state)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        self._n_threads = _openmp_effective_n_threads()

        # Validate init array
        init = self.init
        init_is_array_like = _is_arraylike_not_scalar(init)
        if init_is_array_like:
            init = check_array(init, dtype=X.dtype, copy=True, order="C")
            self._validate_center_shape(X, init)

        # subtract of mean of x for more accurate distance computations
        if not sp.issparse(X):
            X_mean = X.mean(axis=0)
            # The copy was already done above
            X -= X_mean

            if init_is_array_like:
                init -= X_mean

        # precompute squared norms of data points
        x_squared_norms = row_norms(X, squared=True)

        # if self._algorithm == "elkan":
        #     kmeans_single = _kmeans_single_elkan
        # else:
        kmeans_single = _cosine_kmeans_single_lloyd
        self._check_mkl_vcomp(X, X.shape[0])

        best_inertia, best_labels = None, None

        for i in range(self._n_init):
            # Initialize centers
            centers_init = self._init_centroids(
                X,
                x_squared_norms=x_squared_norms,
                init=init,
                random_state=random_state,
                sample_weight=sample_weight,
            )
            if self.verbose:
                print("Initialization complete")

            # run a k-means once
            labels, inertia, centers, n_iter_ = kmeans_single(
                X,
                sample_weight,
                centers_init,
                max_iter=self.max_iter,
                verbose=self.verbose,
                tol=self._tol,
                n_threads=self._n_threads,
            )

            # determine if these results are the best so far
            # we chose a new run if it has a better inertia and the clustering is
            # different from the best so far (it's possible that the inertia is
            # slightly better even if the clustering is the same with potentially
            # permuted labels, due to rounding errors)
            if best_inertia is None or (
                inertia < best_inertia
                and not _is_same_clustering(labels, best_labels, self.n_clusters)
            ):
                best_labels = labels
                best_centers = centers
                best_inertia = inertia
                best_n_iter = n_iter_

        if not sp.issparse(X):
            if not self.copy_x:
                X += X_mean
            best_centers += X_mean

        distinct_clusters = len(set(best_labels))
        if distinct_clusters < self.n_clusters:
            warnings.warn(
                "Number of distinct clusters ({}) found smaller than "
                "n_clusters ({}). Possibly due to duplicate points "
                "in X.".format(distinct_clusters, self.n_clusters),
                ConvergenceWarning,
                stacklevel=2,
            )

        self.cluster_centers_ = best_centers
        self._n_features_out = self.cluster_centers_.shape[0]
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        return self

