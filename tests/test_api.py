"""Tests for the main API."""

from AllMeans import TopicModel


def test_topic_model_basic() -> None:
    """Test basic TopicModel functionality."""
    text = """
    Machine learning is a subset of artificial intelligence.
    Deep learning uses neural networks with many layers.
    Natural language processing helps computers understand human language.
    Computer vision enables machines to interpret visual information.
    Reinforcement learning trains agents through rewards and penalties.
    Supervised learning uses labeled training data.
    Unsupervised learning discovers patterns in unlabeled data.
    Transfer learning applies knowledge from one task to another.
    """

    model = TopicModel(
        method="kmeans",
        n_clusters=3,
        auto_k=False,
        random_state=42,
    )

    model.fit(text)

    assert model.topics_ is not None
    assert len(model.topics_) > 0
    assert model.assignments_ is not None
    assert model.scores_ is not None
    assert "silhouette" in model.scores_


def test_topic_model_auto_k() -> None:
    """Test auto K selection."""
    text = """
    Machine learning is powerful.
    Deep learning uses neural networks.
    Natural language processing is important.
    Computer vision interprets images.
    Reinforcement learning trains agents.
    Supervised learning uses labels.
    Unsupervised learning finds patterns.
    Transfer learning reuses knowledge.
    """

    model = TopicModel(
        method="kmeans",
        auto_k=True,
        k_range=(2, 4),
        early_stop=1,
        random_state=42,
    )

    model.fit(text)

    assert model.topics_ is not None
    assert 2 <= len(model.topics_) <= 4


