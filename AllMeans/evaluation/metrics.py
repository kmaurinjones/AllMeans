"""Topic model evaluation metrics."""


import numpy as np
import numpy.typing as npt
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


def coherence_c_v(
    topics: list[list[str]],
    documents: list[str],
    tokenized_docs: list[list[str]] | None = None,
) -> float:
    """Calculate C_V coherence score.

    Args:
        topics: List of topic keywords (each topic is a list of words)
        documents: Raw documents
        tokenized_docs: Pre-tokenized documents (optional)

    Returns:
        Coherence score (higher is better)
    """
    if not topics or not documents:
        return 0.0

    # Tokenize if not provided
    if tokenized_docs is None:
        tokenized_docs = [doc.lower().split() for doc in documents]

    # Create dictionary and corpus
    dictionary = Dictionary(tokenized_docs)

    try:
        cm = CoherenceModel(
            topics=topics,
            texts=tokenized_docs,
            dictionary=dictionary,
            coherence="c_v",
        )
        return cm.get_coherence()
    except Exception:
        # Fallback if coherence calculation fails
        return 0.0


def calculate_diversity(topics: list[list[str]], top_n: int = 10) -> float:
    """Calculate topic diversity.

    Measures how diverse topics are by counting unique words across all topics.

    Args:
        topics: List of topic keywords
        top_n: Number of top words to consider per topic

    Returns:
        Diversity score in [0, 1] (higher is better)
    """
    if not topics:
        return 0.0

    # Get top N words from each topic
    all_words = set()
    total_words = 0

    for topic in topics:
        topic_words = topic[:top_n]
        all_words.update(topic_words)
        total_words += len(topic_words)

    if total_words == 0:
        return 0.0

    # Diversity = unique words / total words
    return len(all_words) / total_words


def calculate_intrinsic_metrics(
    X: npt.NDArray[np.float32],
    labels: npt.NDArray[np.int_],
    metric: str = "cosine",
) -> dict[str, float]:
    """Calculate intrinsic clustering metrics.

    Args:
        X: Feature matrix
        labels: Cluster labels
        metric: Distance metric for silhouette score

    Returns:
        Dictionary of metric scores
    """
    metrics: dict[str, float] = {}

    # Filter out noise points (-1 labels from HDBSCAN)
    valid_mask = labels >= 0
    if not valid_mask.any():
        return {
            "silhouette": 0.0,
            "davies_bouldin": float("inf"),
            "calinski_harabasz": 0.0,
        }

    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]

    # Need at least 2 clusters
    n_clusters = len(set(labels_valid))
    if n_clusters < 2:
        return {
            "silhouette": 0.0,
            "davies_bouldin": float("inf"),
            "calinski_harabasz": 0.0,
        }

    try:
        metrics["silhouette"] = silhouette_score(X_valid, labels_valid, metric=metric)
    except Exception:
        metrics["silhouette"] = 0.0

    try:
        metrics["davies_bouldin"] = davies_bouldin_score(X_valid, labels_valid)
    except Exception:
        metrics["davies_bouldin"] = float("inf")

    try:
        metrics["calinski_harabasz"] = calinski_harabasz_score(X_valid, labels_valid)
    except Exception:
        metrics["calinski_harabasz"] = 0.0

    return metrics


def select_best_k(
    scores_by_k: dict[int, dict[str, float]],
    weights: dict[str, float] | None = None,
) -> int:
    """Select best number of clusters based on weighted multi-objective optimization.

    Args:
        scores_by_k: Dictionary mapping k -> scores
        weights: Optional weights for each metric (default: equal weights)

    Returns:
        Optimal k
    """
    if not scores_by_k:
        return 2

    # Default weights
    if weights is None:
        weights = {
            "silhouette": 0.3,
            "davies_bouldin": -0.2,  # Negative because lower is better
            "coherence": 0.3,
            "diversity": 0.2,
        }

    # Normalize scores and calculate weighted sum
    best_k = 2
    best_score = float("-inf")

    for k, metrics in scores_by_k.items():
        weighted_score = 0.0

        for metric_name, weight in weights.items():
            if metric_name in metrics:
                value = metrics[metric_name]

                # Handle inf values
                if np.isinf(value):
                    value = 0.0 if weight < 0 else float("-inf")

                weighted_score += weight * value

        if weighted_score > best_score:
            best_score = weighted_score
            best_k = k

    return best_k
