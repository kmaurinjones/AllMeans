"""Clustering algorithm implementations."""

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, LatentDirichletAllocation

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


class KMeansClusterer:
    """K-Means clustering wrapper."""

    def __init__(
        self,
        n_clusters: int = 5,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init="auto",
            **kwargs,
        )
        self._fitted = False

    def fit(
        self,
        X: npt.NDArray[np.float32],
        n_clusters: int | None = None,
    ) -> "KMeansClusterer":
        """Fit K-Means clustering."""
        if n_clusters is not None:
            self.n_clusters = n_clusters
            self.model = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_init="auto",
            )
        self.model.fit(X)
        self._fitted = True
        return self

    def predict(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.int_]:
        """Predict cluster assignments."""
        if not self._fitted:
            raise ValueError("Must call fit() before predict()")
        return self.model.predict(X).astype(np.int_)

    @property
    def labels_(self) -> npt.NDArray[np.int_]:
        """Get cluster labels."""
        if not self._fitted:
            raise ValueError("Must call fit() before accessing labels_")
        return self.model.labels_.astype(np.int_)

    @property
    def n_clusters_(self) -> int:
        """Get number of clusters."""
        return self.n_clusters

    @property
    def cluster_centers_(self) -> npt.NDArray[np.float32]:
        """Get cluster centers."""
        if not self._fitted:
            raise ValueError("Must call fit() before accessing cluster_centers_")
        return self.model.cluster_centers_.astype(np.float32)


class NMFClusterer:
    """Non-Negative Matrix Factorization for topic modeling."""

    def __init__(
        self,
        n_clusters: int = 5,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = NMF(
            n_components=n_clusters,
            random_state=random_state,
            **kwargs,
        )
        self._fitted = False
        self._labels: npt.NDArray[np.int_] | None = None

    def fit(
        self,
        X: npt.NDArray[np.float32],
        n_clusters: int | None = None,
    ) -> "NMFClusterer":
        """Fit NMF model."""
        if n_clusters is not None:
            self.n_clusters = n_clusters
            self.model = NMF(
                n_components=n_clusters,
                random_state=self.random_state,
            )

        W = self.model.fit_transform(X)
        self._labels = np.argmax(W, axis=1).astype(np.int_)
        self._fitted = True
        return self

    def predict(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.int_]:
        """Predict cluster assignments."""
        if not self._fitted:
            raise ValueError("Must call fit() before predict()")
        W = self.model.transform(X)
        return np.argmax(W, axis=1).astype(np.int_)

    @property
    def labels_(self) -> npt.NDArray[np.int_]:
        """Get cluster labels."""
        if self._labels is None:
            raise ValueError("Must call fit() before accessing labels_")
        return self._labels

    @property
    def n_clusters_(self) -> int:
        """Get number of clusters."""
        return self.n_clusters


class LDAClusterer:
    """Latent Dirichlet Allocation for topic modeling."""

    def __init__(
        self,
        n_clusters: int = 5,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = LatentDirichletAllocation(
            n_components=n_clusters,
            random_state=random_state,
            **kwargs,
        )
        self._fitted = False
        self._labels: npt.NDArray[np.int_] | None = None

    def fit(
        self,
        X: npt.NDArray[np.float32],
        n_clusters: int | None = None,
    ) -> "LDAClusterer":
        """Fit LDA model."""
        if n_clusters is not None:
            self.n_clusters = n_clusters
            self.model = LatentDirichletAllocation(
                n_components=n_clusters,
                random_state=self.random_state,
            )

        doc_topic = self.model.fit_transform(X)
        self._labels = np.argmax(doc_topic, axis=1).astype(np.int_)
        self._fitted = True
        return self

    def predict(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.int_]:
        """Predict cluster assignments."""
        if not self._fitted:
            raise ValueError("Must call fit() before predict()")
        doc_topic = self.model.transform(X)
        return np.argmax(doc_topic, axis=1).astype(np.int_)

    @property
    def labels_(self) -> npt.NDArray[np.int_]:
        """Get cluster labels."""
        if self._labels is None:
            raise ValueError("Must call fit() before accessing labels_")
        return self._labels

    @property
    def n_clusters_(self) -> int:
        """Get number of clusters."""
        return self.n_clusters


class HDBSCANClusterer:
    """HDBSCAN clustering with UMAP dimensionality reduction."""

    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: int = 3,
        umap_n_components: int = 50,
        umap_n_neighbors: int = 15,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        if not HDBSCAN_AVAILABLE:
            raise ImportError("hdbscan is required for HDBSCANClusterer")
        if not UMAP_AVAILABLE:
            raise ImportError("umap-learn is required for HDBSCANClusterer")

        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.umap_n_components = umap_n_components
        self.umap_n_neighbors = umap_n_neighbors
        self.random_state = random_state

        self.umap_model = umap.UMAP(
            n_components=umap_n_components,
            n_neighbors=umap_n_neighbors,
            random_state=random_state,
        )

        self.cluster_model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            **kwargs,
        )

        self._fitted = False
        self._n_clusters = 0

    def fit(
        self,
        X: npt.NDArray[np.float32],
        n_clusters: int | None = None,
    ) -> "HDBSCANClusterer":
        """Fit HDBSCAN with UMAP."""
        # Apply UMAP
        X_reduced = self.umap_model.fit_transform(X)

        # Fit HDBSCAN
        self.cluster_model.fit(X_reduced)

        # Count number of clusters (excluding noise label -1)
        unique_labels = set(self.cluster_model.labels_)
        self._n_clusters = len(unique_labels - {-1})

        self._fitted = True
        return self

    def predict(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.int_]:
        """Predict cluster assignments."""
        if not self._fitted:
            raise ValueError("Must call fit() before predict()")

        X_reduced = self.umap_model.transform(X)
        labels, _ = hdbscan.approximate_predict(self.cluster_model, X_reduced)
        return labels.astype(np.int_)

    @property
    def labels_(self) -> npt.NDArray[np.int_]:
        """Get cluster labels."""
        if not self._fitted:
            raise ValueError("Must call fit() before accessing labels_")
        return self.cluster_model.labels_.astype(np.int_)

    @property
    def n_clusters_(self) -> int:
        """Get number of clusters found."""
        return self._n_clusters


def get_clustering_algorithm(
    method: Literal["kmeans", "nmf", "lda", "hdbscan"],
    **kwargs: Any,
) -> KMeansClusterer | NMFClusterer | LDAClusterer | HDBSCANClusterer:
    """Factory function to get clustering algorithm.

    Args:
        method: Clustering method ('kmeans', 'nmf', 'lda', 'hdbscan')
        **kwargs: Additional arguments passed to the algorithm

    Returns:
        Clustering algorithm instance
    """
    if method == "kmeans":
        return KMeansClusterer(**kwargs)
    elif method == "nmf":
        return NMFClusterer(**kwargs)
    elif method == "lda":
        return LDAClusterer(**kwargs)
    elif method == "hdbscan":
        return HDBSCANClusterer(**kwargs)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
