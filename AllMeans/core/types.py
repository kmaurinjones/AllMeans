"""Core types and data structures for AllMeans."""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


@dataclass
class Topic:
    """Represents a discovered topic."""

    id: int
    label: str
    keywords: list[str]
    size: int
    coherence: float
    diversity: float
    exemplar_sentences: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Topic(id={self.id}, label='{self.label}', size={self.size}, coherence={self.coherence:.3f})"


@dataclass
class TopicModelResults:
    """Results from topic modeling."""

    topics: list[Topic]
    assignments: npt.NDArray[np.int_]  # sentence -> topic_id
    scores: dict[str, float]
    config: dict[str, Any]
    feature_matrix: npt.NDArray[np.float32] | None = None
    sentences: list[str] = field(default_factory=list)

    def get_topic(self, topic_id: int) -> Topic | None:
        """Get topic by ID."""
        for topic in self.topics:
            if topic.id == topic_id:
                return topic
        return None

    def get_sentences_for_topic(self, topic_id: int) -> list[str]:
        """Get all sentences assigned to a topic."""
        indices = np.where(self.assignments == topic_id)[0]
        return [self.sentences[i] for i in indices]


@runtime_checkable
class Tokenizer(Protocol):
    """Protocol for tokenizers."""

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into tokens."""
        ...


@runtime_checkable
class FeatureExtractor(Protocol):
    """Protocol for feature extractors."""

    def fit(self, documents: list[str]) -> "FeatureExtractor":
        """Fit the feature extractor."""
        ...

    def transform(self, documents: list[str]) -> npt.NDArray[np.float32]:
        """Transform documents to features."""
        ...

    def fit_transform(self, documents: list[str]) -> npt.NDArray[np.float32]:
        """Fit and transform in one step."""
        ...


@runtime_checkable
class ClusteringAlgorithm(Protocol):
    """Protocol for clustering algorithms."""

    def fit(self, X: npt.NDArray[np.float32], n_clusters: int | None = None) -> "ClusteringAlgorithm":
        """Fit the clustering algorithm."""
        ...

    def predict(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.int_]:
        """Predict cluster assignments."""
        ...

    @property
    def labels_(self) -> npt.NDArray[np.int_]:
        """Get cluster labels."""
        ...

    @property
    def n_clusters_(self) -> int:
        """Get number of clusters found."""
        ...


@runtime_checkable
class CoherenceMetric(Protocol):
    """Protocol for coherence metrics."""

    def score(
        self,
        topics: list[list[str]],
        documents: list[str],
        **kwargs: Any,
    ) -> float:
        """Calculate coherence score."""
        ...
