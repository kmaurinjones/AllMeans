"""Feature extraction methods."""

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .types import FeatureExtractor, Tokenizer


class TfidfFeatureExtractor:
    """TF-IDF feature extractor."""

    def __init__(
        self,
        tokenizer: Tokenizer | None = None,
        max_features: int = 20000,
        min_df: int = 1,
        max_df: float = 0.95,
        ngram_range: tuple[int, int] = (1, 2),
        **kwargs: Any,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range

        # Build vectorizer kwargs
        vectorizer_kwargs: dict[str, Any] = {
            "max_features": max_features,
            "min_df": min_df,
            "max_df": max_df,
            "ngram_range": ngram_range,
            "stop_words": "english",
            **kwargs,
        }

        if tokenizer is not None:
            vectorizer_kwargs["tokenizer"] = tokenizer.tokenize

        self.vectorizer = TfidfVectorizer(**vectorizer_kwargs)
        self._fitted = False

    def fit(self, documents: list[str]) -> "TfidfFeatureExtractor":
        """Fit the TF-IDF vectorizer."""
        self.vectorizer.fit(documents)
        self._fitted = True
        return self

    def transform(self, documents: list[str]) -> npt.NDArray[np.float32]:
        """Transform documents to TF-IDF features."""
        if not self._fitted:
            raise ValueError("Must call fit() before transform()")
        matrix = self.vectorizer.transform(documents)
        return matrix.toarray().astype(np.float32)

    def fit_transform(self, documents: list[str]) -> npt.NDArray[np.float32]:
        """Fit and transform in one step."""
        matrix = self.vectorizer.fit_transform(documents)
        self._fitted = True
        return matrix.toarray().astype(np.float32)

    def get_feature_names(self) -> list[str]:
        """Get feature names."""
        if not self._fitted:
            raise ValueError("Must call fit() before get_feature_names()")
        return list(self.vectorizer.get_feature_names_out())


class BagOfWordsExtractor:
    """Bag of Words feature extractor."""

    def __init__(
        self,
        tokenizer: Tokenizer | None = None,
        max_features: int = 10000,
        min_df: int = 1,
        max_df: float = 0.95,
        **kwargs: Any,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df

        vectorizer_kwargs: dict[str, Any] = {
            "max_features": max_features,
            "min_df": min_df,
            "max_df": max_df,
            "stop_words": "english",
            **kwargs,
        }

        if tokenizer is not None:
            vectorizer_kwargs["tokenizer"] = tokenizer.tokenize

        self.vectorizer = CountVectorizer(**vectorizer_kwargs)
        self._fitted = False

    def fit(self, documents: list[str]) -> "BagOfWordsExtractor":
        """Fit the BoW vectorizer."""
        self.vectorizer.fit(documents)
        self._fitted = True
        return self

    def transform(self, documents: list[str]) -> npt.NDArray[np.float32]:
        """Transform documents to BoW features."""
        if not self._fitted:
            raise ValueError("Must call fit() before transform()")
        matrix = self.vectorizer.transform(documents)
        return matrix.toarray().astype(np.float32)

    def fit_transform(self, documents: list[str]) -> npt.NDArray[np.float32]:
        """Fit and transform in one step."""
        matrix = self.vectorizer.fit_transform(documents)
        self._fitted = True
        return matrix.toarray().astype(np.float32)

    def get_feature_names(self) -> list[str]:
        """Get feature names."""
        if not self._fitted:
            raise ValueError("Must call fit() before get_feature_names()")
        return list(self.vectorizer.get_feature_names_out())


class SIFEmbedding:
    """Smooth Inverse Frequency sentence embeddings."""

    def __init__(
        self,
        embedding_model: KeyedVectors | None = None,
        alpha: float = 1e-3,
        remove_pc: bool = True,
    ) -> None:
        self.embedding_model = embedding_model
        self.alpha = alpha
        self.remove_pc = remove_pc
        self.word_freq: dict[str, float] = {}
        self._fitted = False

    def _calculate_word_freq(self, documents: list[str]) -> None:
        """Calculate word frequencies from documents."""
        word_count: dict[str, int] = {}
        total_words = 0

        for doc in documents:
            tokens = doc.lower().split()
            for token in tokens:
                word_count[token] = word_count.get(token, 0) + 1
                total_words += 1

        self.word_freq = {
            word: count / total_words for word, count in word_count.items()
        }

    def _get_sentence_vector(self, sentence: str) -> npt.NDArray[np.float32]:
        """Get SIF-weighted sentence vector."""
        if self.embedding_model is None:
            raise ValueError("embedding_model must be provided")

        tokens = sentence.lower().split()
        vectors = []
        weights = []

        for token in tokens:
            if token in self.embedding_model:
                freq = self.word_freq.get(token, 1e-5)
                weight = self.alpha / (self.alpha + freq)
                vectors.append(self.embedding_model[token])
                weights.append(weight)

        if not vectors:
            # Return zero vector if no tokens found
            return np.zeros(self.embedding_model.vector_size, dtype=np.float32)

        weighted_avg = np.average(vectors, axis=0, weights=weights)
        return weighted_avg.astype(np.float32)

    def fit(self, documents: list[str]) -> "SIFEmbedding":
        """Fit by calculating word frequencies."""
        self._calculate_word_freq(documents)
        self._fitted = True
        return self

    def transform(self, documents: list[str]) -> npt.NDArray[np.float32]:
        """Transform documents to SIF embeddings."""
        if not self._fitted:
            raise ValueError("Must call fit() before transform()")

        vectors = np.array([self._get_sentence_vector(doc) for doc in documents])

        if self.remove_pc and len(vectors) > 1:
            # Remove first principal component
            mean = vectors.mean(axis=0)
            vectors -= mean
            u, _, _ = np.linalg.svd(vectors, full_matrices=False)
            pc = u[:, 0]
            vectors -= vectors.dot(pc[:, None]) * pc[None, :]

        return vectors

    def fit_transform(self, documents: list[str]) -> npt.NDArray[np.float32]:
        """Fit and transform in one step."""
        self.fit(documents)
        return self.transform(documents)


def get_feature_extractor(
    method: Literal["tfidf", "bow", "sif"],
    **kwargs: Any,
) -> FeatureExtractor:
    """Factory function to get feature extractor.

    Args:
        method: Feature extraction method ('tfidf', 'bow', 'sif')
        **kwargs: Additional arguments passed to the extractor

    Returns:
        Feature extractor instance
    """
    if method == "tfidf":
        return TfidfFeatureExtractor(**kwargs)  # type: ignore[return-value]
    elif method == "bow":
        return BagOfWordsExtractor(**kwargs)  # type: ignore[return-value]
    elif method == "sif":
        return SIFEmbedding(**kwargs)  # type: ignore[return-value]
    else:
        raise ValueError(f"Unknown feature extraction method: {method}")
