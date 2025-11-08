"""AllMeans: Automatic topic discovery with zero LLMs, minimal input."""

__version__ = "2.0.0"

from .api import TopicModel
from .core.types import Topic, TopicModelResults

__all__ = [
    "TopicModel",
    "Topic",
    "TopicModelResults",
]
