"""Topic labeling methods."""

from itertools import product

import nltk
import numpy as np
import numpy.typing as npt
from jellyfish import jaro_winkler_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure NLTK resources are available
try:
    nltk.data.find("taggers/averaged_perceptron_tagger")
except LookupError:
    nltk.download("averaged_perceptron_tagger", quiet=True)
try:
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
except LookupError:
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)


class TopicLabeler:
    """Extract topic labels from clusters."""

    def __init__(
        self,
        n_words: int = 5,
        exclusions: list[str] | None = None,
        excl_sim: float = 0.9,
        max_diversity: bool = True,
        filter_pos: bool = True,
        allowed_pos: set[str] | None = None,
    ) -> None:
        """Initialize TopicLabeler.

        Args:
            n_words: Number of keywords to extract per topic
            exclusions: Words to exclude from keywords
            excl_sim: Similarity threshold for exclusions
            max_diversity: Use diversity-based label selection
            filter_pos: Filter keywords by part-of-speech
            allowed_pos: POS tags to keep (default: nouns, verbs, adjectives)
        """
        self.n_words = n_words
        self.exclusions = [e.lower().strip() for e in (exclusions or [])]
        self.excl_sim = excl_sim
        self.max_diversity = max_diversity
        self.filter_pos = filter_pos

        # Default to meaningful POS tags: nouns, verbs, adjectives
        if allowed_pos is None:
            self.allowed_pos = {
                "NN",    # Noun, singular
                "NNS",   # Noun, plural
                "NNP",   # Proper noun, singular
                "NNPS",  # Proper noun, plural
                "VB",    # Verb, base form
                "VBD",   # Verb, past tense
                "VBG",   # Verb, gerund/present participle
                "VBN",   # Verb, past participle
                "VBP",   # Verb, non-3rd person singular present
                "VBZ",   # Verb, 3rd person singular present
                "JJ",    # Adjective
                "JJR",   # Adjective, comparative
                "JJS",   # Adjective, superlative
            }
        else:
            self.allowed_pos = allowed_pos

    def _is_uninformative(self, word: str) -> bool:
        """Check if word is uninformative (number, ordinal, etc.)."""
        word_lower = word.lower()

        # Ordinal numbers
        ordinals = {
            "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
            "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth",
            "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth",
            "nineteenth", "twentieth", "thirtieth", "fortieth", "fiftieth",
            "sixtieth", "seventieth", "eightieth", "ninetieth", "hundredth",
            "thousandth", "millionth", "billionth"
        }

        if word_lower in ordinals:
            return True

        # Pure numbers
        if word.replace(",", "").replace(".", "").isdigit():
            return True

        # Roman numerals
        if all(c in "IVXLCDM" for c in word.upper()) and len(word) <= 10:
            return True

        # Common uninformative words that slip through stopwords
        uninformative = {
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "many", "much", "several", "various", "different", "certain", "particular",
            "also", "however", "therefore", "thus", "moreover", "furthermore", "nevertheless"
        }

        return word_lower in uninformative

    def _filter_by_pos(self, words: list[str]) -> list[str]:
        """Filter words by part-of-speech tags."""
        if not self.filter_pos or not words:
            return words

        # Tag all words at once (more efficient)
        tagged = nltk.pos_tag(words)

        # Filter by allowed POS tags and uninformative words
        filtered = [
            word for word, pos in tagged
            if pos in self.allowed_pos and not self._is_uninformative(word)
        ]

        return filtered

    def _filter_exclusions(self, words: list[str], n_needed: int) -> list[str]:
        """Filter out words similar to exclusions."""
        filtered = []
        for word in words:
            if not any(
                jaro_winkler_similarity(word.lower(), excl) > self.excl_sim
                for excl in self.exclusions
            ):
                filtered.append(word)
                if len(filtered) >= n_needed:
                    break
        return filtered

    def extract_keywords_tfidf(
        self,
        cluster_docs: list[str],
        all_docs: list[str],
        vectorizer: TfidfVectorizer | None = None,
    ) -> list[str]:
        """Extract keywords using TF-IDF."""
        if not cluster_docs:
            return []

        # Use provided vectorizer or create new one
        if vectorizer is None:
            vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
            vectorizer.fit(all_docs)

        # Get TF-IDF scores for cluster documents
        cluster_tfidf = vectorizer.transform(cluster_docs)
        avg_tfidf = cluster_tfidf.mean(axis=0).A1

        # Get feature names
        feature_names = vectorizer.get_feature_names_out()

        # Sort by TF-IDF score
        top_indices = avg_tfidf.argsort()[::-1]
        top_words = [feature_names[i] for i in top_indices]

        # Filter by POS tags (before exclusions to have more candidates)
        if self.filter_pos:
            # Get more candidates for POS filtering
            pos_filtered = self._filter_by_pos(top_words[: self.n_words * 10])
        else:
            pos_filtered = top_words

        # Filter exclusions
        filtered = self._filter_exclusions(pos_filtered, self.n_words * 3)

        return filtered[: self.n_words]

    def calculate_dissimilarity(
        self,
        words: list[str],
        embeddings: dict[str, npt.NDArray[np.float32]],
    ) -> float:
        """Calculate pairwise dissimilarity between words."""
        if len(words) < 2:
            return 0.0

        word_vecs = [embeddings[w] for w in words if w in embeddings]
        if len(word_vecs) < 2:
            return 0.0

        sim_matrix = cosine_similarity(word_vecs)
        distance_matrix = 1 - sim_matrix
        total_dissimilarity = np.sum(np.triu(distance_matrix, k=1))

        return float(total_dissimilarity)

    def select_diverse_label(
        self,
        cluster_keywords: list[list[str]],
        embeddings: dict[str, npt.NDArray[np.float32]] | None = None,
    ) -> tuple[str, ...]:
        """Select most diverse combination of keywords across clusters.

        Args:
            cluster_keywords: List of keyword lists (one per cluster)
            embeddings: Optional embeddings for diversity calculation

        Returns:
            Tuple of selected keywords (one per cluster)
        """
        if not cluster_keywords:
            return tuple()

        if not self.max_diversity or embeddings is None:
            # Just take first keyword from each cluster
            return tuple(kw[0] if kw else f"topic_{i}" for i, kw in enumerate(cluster_keywords))

        # Try all combinations and find most diverse
        best_selection = None
        best_score = -np.inf

        for selection in product(*cluster_keywords):
            score = self.calculate_dissimilarity(list(selection), embeddings)
            if score > best_score:
                best_score = score
                best_selection = selection

        if best_selection is None:
            return tuple(f"topic_{i}" for i in range(len(cluster_keywords)))

        return best_selection

    def label_clusters(
        self,
        cluster_assignments: npt.NDArray[np.int_],
        documents: list[str],
        vectorizer: TfidfVectorizer | None = None,
        embeddings: dict[str, npt.NDArray[np.float32]] | None = None,
    ) -> tuple[list[str], list[list[str]]]:
        """Extract labels and keywords for all clusters.

        Args:
            cluster_assignments: Cluster assignment for each document
            documents: List of documents
            vectorizer: Optional pre-fitted TF-IDF vectorizer
            embeddings: Optional word embeddings for diversity

        Returns:
            Tuple of (labels, keywords_per_cluster)
        """
        unique_clusters = sorted(set(cluster_assignments) - {-1})  # Exclude noise
        cluster_keywords = []

        for cluster_id in unique_clusters:
            cluster_docs = [
                doc for i, doc in enumerate(documents) if cluster_assignments[i] == cluster_id
            ]
            keywords = self.extract_keywords_tfidf(cluster_docs, documents, vectorizer)
            cluster_keywords.append(keywords)

        # Select diverse labels
        labels_tuple = self.select_diverse_label(cluster_keywords, embeddings)
        labels = list(labels_tuple)

        # Ensure we have labels for all clusters
        while len(labels) < len(unique_clusters):
            labels.append(f"topic_{len(labels)}")

        return labels, cluster_keywords
