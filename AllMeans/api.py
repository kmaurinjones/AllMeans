"""Main API for AllMeans topic modeling."""

import warnings
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .algorithms.clustering import get_clustering_algorithm
from .core.features import get_feature_extractor
from .core.labeling import TopicLabeler
from .core.pipeline import TextPipeline
from .core.types import FeatureExtractor, Topic, TopicModelResults
from .evaluation.metrics import (
    calculate_diversity,
    calculate_intrinsic_metrics,
    coherence_c_v,
    select_best_k,
)

console = Console()


class TopicModel:
    """Modern topic modeling with fit/transform API."""

    def __init__(
        self,
        method: Literal["kmeans", "nmf", "lda", "hdbscan"] = "kmeans",
        feature_method: Literal["tfidf", "bow", "sif"] = "tfidf",
        n_clusters: int | None = None,
        auto_k: bool = True,
        k_range: tuple[int, int] = (2, 10),
        early_stop: int = 2,
        exclusions: list[str] | None = None,
        excl_sim: float = 0.9,
        filter_pos: bool = True,
        random_state: int = 42,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize TopicModel.

        Args:
            method: Clustering method ('kmeans', 'nmf', 'lda', 'hdbscan')
            feature_method: Feature extraction method ('tfidf', 'bow', 'sif')
            n_clusters: Fixed number of clusters (None for auto)
            auto_k: Auto-select optimal K
            k_range: Range of K values to try
            early_stop: Early stopping patience
            exclusions: Words to exclude from labels
            excl_sim: Similarity threshold for exclusions
            filter_pos: Filter keywords by part-of-speech (removes ordinals, numbers, etc.)
            random_state: Random seed
            verbose: Show progress and detailed output
            **kwargs: Additional args for clustering/features
        """
        self.method = method
        self.feature_method = feature_method
        self.n_clusters = n_clusters
        self.auto_k = auto_k
        self.k_range = k_range
        self.early_stop = early_stop
        self.exclusions = exclusions
        self.excl_sim = excl_sim
        self.filter_pos = filter_pos
        self.random_state = random_state
        self.verbose = verbose
        self.kwargs = kwargs

        # Components (initialized during fit)
        self.pipeline = TextPipeline()
        self.feature_extractor: FeatureExtractor | None = None
        self.clusterer: Any = None
        self.labeler = TopicLabeler(
            exclusions=exclusions,
            excl_sim=excl_sim,
            filter_pos=filter_pos,
        )

        # Results
        self.topics_: list[Topic] | None = None
        self.assignments_: npt.NDArray[np.int_] | None = None
        self.scores_: dict[str, float] | None = None
        self._sentences: list[str] = []  # Lemmatized sentences for computation
        self._sentences_original: list[str] = []  # Original sentences for display
        self._tokenized: list[list[str]] = []
        self._feature_matrix: npt.NDArray[np.float32] | None = None

    def fit(self, text: str | list[str]) -> "TopicModel":
        """Fit topic model on text.

        Args:
            text: Input text (single string or list of documents)

        Returns:
            Self
        """
        # Always suppress sklearn RuntimeWarnings (expected for sparse data)
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

        # Handle input
        if isinstance(text, str):
            sentences, self._tokenized = self.pipeline.preprocess(text)
        else:
            sentences = text
            self._tokenized = [self.pipeline.tokenizer.tokenize(s) for s in text]

        if len(sentences) < 2:
            raise ValueError("Need at least 2 sentences/documents")

        # Store original sentences for display
        self._sentences_original = sentences

        # Lemmatize sentences for computation (TF-IDF, clustering, keywords)
        self._sentences = self.pipeline.lemmatize_sentences(sentences)

        if self.verbose:
            console.print(f"[cyan]Preprocessing:[/cyan] Found {len(self._sentences)} sentences")

        # Extract features
        if self.verbose:
            console.print(f"[cyan]Extracting features:[/cyan] Using {self.feature_method}")

        self.feature_extractor = get_feature_extractor(
            self.feature_method,
            **self.kwargs,
        )
        self._feature_matrix = self.feature_extractor.fit_transform(self._sentences)

        # Determine optimal K if auto_k
        if self.auto_k and self.n_clusters is None:
            best_k = self._find_best_k()
        else:
            best_k = self.n_clusters or 5

        # Fit final model with best K
        self.clusterer = get_clustering_algorithm(
            self.method,
            n_clusters=best_k,
            random_state=self.random_state,
            **self.kwargs,
        )
        self.clusterer.fit(self._feature_matrix, n_clusters=best_k)
        self.assignments_ = self.clusterer.labels_

        # Extract labels and keywords
        labels, keywords_per_cluster = self.labeler.label_clusters(
            self.assignments_,
            self._sentences,
        )

        # Calculate final metrics
        intrinsic = calculate_intrinsic_metrics(self._feature_matrix, self.assignments_)
        coherence = coherence_c_v(keywords_per_cluster, self._sentences, self._tokenized)
        diversity = calculate_diversity(keywords_per_cluster)

        self.scores_ = {
            **intrinsic,
            "coherence": coherence,
            "diversity": diversity,
        }

        # Build topics
        self.topics_ = []
        unique_clusters = sorted(set(self.assignments_) - {-1})

        for i, cluster_id in enumerate(unique_clusters):
            cluster_mask = self.assignments_ == cluster_id
            cluster_size = int(cluster_mask.sum())
            # Use original sentences for exemplars (not lemmatized)
            cluster_sentences = [
                s for j, s in enumerate(self._sentences_original) if cluster_mask[j]
            ]
            exemplars = cluster_sentences[:3]

            topic = Topic(
                id=cluster_id,
                label=labels[i] if i < len(labels) else f"topic_{cluster_id}",
                keywords=keywords_per_cluster[i] if i < len(keywords_per_cluster) else [],
                size=cluster_size,
                coherence=coherence,
                diversity=diversity,
                exemplar_sentences=exemplars,
            )
            self.topics_.append(topic)

        return self

    def _find_best_k(self) -> int:
        """Find optimal number of clusters."""
        if self._feature_matrix is None:
            raise ValueError("Features not extracted")

        scores_by_k: dict[int, dict[str, float]] = {}
        worsening_count = 0
        prev_score = float("-inf")

        k_values = list(range(self.k_range[0], self.k_range[1] + 1))

        if self.verbose:
            console.print(f"[cyan]Auto-K selection:[/cyan] Testing K={self.k_range[0]} to K={self.k_range[1]}")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Finding optimal K...", total=len(k_values))

                for k in k_values:
                    progress.update(task, description=f"Testing K={k}...")

                    # Fit clustering
                    clusterer = get_clustering_algorithm(
                        self.method,
                        n_clusters=k,
                        random_state=self.random_state,
                        **self.kwargs,
                    )
                    clusterer.fit(self._feature_matrix, n_clusters=k)

                    # Calculate metrics
                    intrinsic = calculate_intrinsic_metrics(
                        self._feature_matrix,
                        clusterer.labels_,
                    )

                    # Extract keywords for coherence
                    labels, keywords = self.labeler.label_clusters(
                        clusterer.labels_,
                        self._sentences,
                    )

                    coherence = coherence_c_v(keywords, self._sentences, self._tokenized)
                    diversity = calculate_diversity(keywords)

                    scores = {
                        **intrinsic,
                        "coherence": coherence,
                        "diversity": diversity,
                    }
                    scores_by_k[k] = scores

                    # Early stopping based on weighted score
                    current_score = (
                        scores["silhouette"] * 0.3
                        - scores["davies_bouldin"] * 0.2
                        + coherence * 0.3
                        + diversity * 0.2
                    )

                    if current_score < prev_score:
                        worsening_count += 1
                    else:
                        worsening_count = 0

                    prev_score = current_score

                    progress.advance(task)

                    if worsening_count >= self.early_stop:
                        progress.update(task, completed=len(k_values))
                        break
        else:
            # Non-verbose version (original logic)
            for k in k_values:
                # Fit clustering
                clusterer = get_clustering_algorithm(
                    self.method,
                    n_clusters=k,
                    random_state=self.random_state,
                    **self.kwargs,
                )
                clusterer.fit(self._feature_matrix, n_clusters=k)

                # Calculate metrics
                intrinsic = calculate_intrinsic_metrics(
                    self._feature_matrix,
                    clusterer.labels_,
                )

                # Extract keywords for coherence
                labels, keywords = self.labeler.label_clusters(
                    clusterer.labels_,
                    self._sentences,
                )

                coherence = coherence_c_v(keywords, self._sentences, self._tokenized)
                diversity = calculate_diversity(keywords)

                scores = {
                    **intrinsic,
                    "coherence": coherence,
                    "diversity": diversity,
                }
                scores_by_k[k] = scores

                # Early stopping based on weighted score
                current_score = (
                    scores["silhouette"] * 0.3
                    - scores["davies_bouldin"] * 0.2
                    + coherence * 0.3
                    + diversity * 0.2
                )

                if current_score < prev_score:
                    worsening_count += 1
                else:
                    worsening_count = 0

                prev_score = current_score

                if worsening_count >= self.early_stop:
                    break

        best_k = select_best_k(scores_by_k)

        if self.verbose:
            console.print(f"[green]✓ Selected K={best_k}[/green]")

        return best_k

    def transform(self, text: str | list[str]) -> npt.NDArray[np.int_]:
        """Predict topic assignments for new text.

        Args:
            text: Input text

        Returns:
            Topic assignments
        """
        if self.feature_extractor is None or self.clusterer is None:
            raise ValueError("Must call fit() before transform()")

        # Preprocess
        if isinstance(text, str):
            sentences, _ = self.pipeline.preprocess(text)
        else:
            sentences = text

        # Lemmatize for feature extraction
        sentences = self.pipeline.lemmatize_sentences(sentences)

        # Extract features
        features = self.feature_extractor.transform(sentences)

        # Predict
        return self.clusterer.predict(features)

    def fit_transform(self, text: str | list[str]) -> npt.NDArray[np.int_]:
        """Fit and transform in one step.

        Args:
            text: Input text

        Returns:
            Topic assignments
        """
        self.fit(text)
        if self.assignments_ is None:
            raise ValueError("Fit failed")
        return self.assignments_

    def get_results(self) -> TopicModelResults:
        """Get modeling results.

        Returns:
            TopicModelResults object
        """
        if self.topics_ is None or self.assignments_ is None or self.scores_ is None:
            raise ValueError("Must call fit() first")

        return TopicModelResults(
            topics=self.topics_,
            assignments=self.assignments_,
            scores=self.scores_,
            config={
                "method": self.method,
                "feature_method": self.feature_method,
                "n_clusters": self.n_clusters,
                "filter_pos": self.filter_pos,
                "random_state": self.random_state,
            },
            feature_matrix=self._feature_matrix,
            sentences=self._sentences_original,  # Return original sentences, not lemmatized
        )
