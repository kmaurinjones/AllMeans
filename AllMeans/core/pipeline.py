"""Text preprocessing pipeline."""

import re
from typing import Literal

import nltk
from nltk.corpus import names, stopwords, wordnet
from nltk.stem import WordNetLemmatizer

from .types import Tokenizer


class SimpleTokenizer:
    """Simple regex-based tokenizer."""

    def __init__(
        self,
        lowercase: bool = True,
        remove_punct: bool = True,
        min_length: int = 2,
    ) -> None:
        self.lowercase = lowercase
        self.remove_punct = remove_punct
        self.min_length = min_length

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text using regex."""
        if self.lowercase:
            text = text.lower()

        if self.remove_punct:
            tokens = re.findall(r"\b[a-z]+\b", text)
        else:
            tokens = re.findall(r"\b\w+\b", text)

        return [t for t in tokens if len(t) >= self.min_length]


class NLTKTokenizer:
    """NLTK-based tokenizer with lemmatization and stopword removal."""

    def __init__(
        self,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        pos_filter: list[str] | None = None,
        remove_names: bool = True,
        language: str = "english",
    ) -> None:
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.pos_filter = pos_filter or ["NN", "NNS", "NNP", "NNPS"]
        self.remove_names = remove_names
        self.language = language
        self._lemmatizer: WordNetLemmatizer | None = None
        self._stopwords: set[str] | None = None
        self._names: set[str] | None = None
        self._ensure_nltk_data()

    def _ensure_nltk_data(self) -> None:
        """Ensure NLTK data is downloaded."""
        resources = ["averaged_perceptron_tagger", "punkt", "wordnet", "stopwords", "names"]
        for resource in resources:
            try:
                nltk.data.find(f"tokenizers/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)

    @property
    def lemmatizer(self) -> WordNetLemmatizer:
        """Lazy load lemmatizer."""
        if self._lemmatizer is None:
            self._lemmatizer = WordNetLemmatizer()
        return self._lemmatizer

    @property
    def stopwords_set(self) -> set[str]:
        """Lazy load stopwords."""
        if self._stopwords is None:
            self._stopwords = set(stopwords.words(self.language))
        return self._stopwords

    @property
    def names_set(self) -> set[str]:
        """Lazy load names."""
        if self._names is None:
            male = names.words("male.txt")
            female = names.words("female.txt")
            self._names = {n.lower() for n in male + female}
        return self._names

    @staticmethod
    def _get_wordnet_pos(treebank_tag: str) -> str:
        """Map treebank POS tag to WordNet POS tag."""
        if treebank_tag.startswith("J"):
            return wordnet.ADJ
        elif treebank_tag.startswith("V"):
            return wordnet.VERB
        elif treebank_tag.startswith("N"):
            return wordnet.NOUN
        elif treebank_tag.startswith("R"):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def tokenize(self, text: str) -> list[str]:
        """Tokenize and preprocess text."""
        # Basic tokenization
        tokens = nltk.word_tokenize(text)

        # Keep only alphabetic tokens
        tokens = [t for t in tokens if t.isalpha()]

        # Remove stopwords
        if self.remove_stopwords:
            tokens = [t for t in tokens if t.lower() not in self.stopwords_set]

        # POS tagging and filtering
        tagged = nltk.pos_tag(tokens)

        if self.pos_filter:
            tagged = [(word, tag) for word, tag in tagged if tag in self.pos_filter]

        # Remove names
        if self.remove_names:
            tagged = [(word, tag) for word, tag in tagged if word.lower() not in self.names_set]

        # Lemmatization
        if self.lemmatize:
            tokens = [
                self.lemmatizer.lemmatize(word, self._get_wordnet_pos(tag))
                for word, tag in tagged
            ]
        else:
            tokens = [word for word, _ in tagged]

        return tokens


class TextPipeline:
    """Unified text preprocessing pipeline."""

    def __init__(
        self,
        tokenizer: Tokenizer | Literal["simple", "nltk"] = "simple",
        sentence_split: bool = True,
        **tokenizer_kwargs: dict[str, object],
    ) -> None:
        self.sentence_split = sentence_split
        self._lemmatizer: WordNetLemmatizer | None = None

        if isinstance(tokenizer, str):
            if tokenizer == "simple":
                self.tokenizer: Tokenizer = SimpleTokenizer(**tokenizer_kwargs)  # type: ignore[arg-type]
            elif tokenizer == "nltk":
                self.tokenizer = NLTKTokenizer(**tokenizer_kwargs)  # type: ignore[arg-type]
            else:
                raise ValueError(f"Unknown tokenizer: {tokenizer}")
        else:
            self.tokenizer = tokenizer

    @property
    def lemmatizer(self) -> WordNetLemmatizer:
        """Lazy load lemmatizer."""
        if self._lemmatizer is None:
            self._lemmatizer = WordNetLemmatizer()
        return self._lemmatizer

    @staticmethod
    def _get_wordnet_pos(treebank_tag: str) -> str:
        """Map treebank POS tag to WordNet POS tag."""
        if treebank_tag.startswith("J"):
            return wordnet.ADJ
        elif treebank_tag.startswith("V"):
            return wordnet.VERB
        elif treebank_tag.startswith("N"):
            return wordnet.NOUN
        elif treebank_tag.startswith("R"):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def lemmatize_sentence(self, sentence: str) -> str:
        """Lemmatize a sentence while preserving structure.

        Args:
            sentence: Input sentence

        Returns:
            Lemmatized sentence with normalized word forms
        """
        # Tokenize
        tokens = nltk.word_tokenize(sentence)

        # POS tag
        tagged = nltk.pos_tag(tokens)

        # Lemmatize each token with its POS
        lemmatized_tokens = []
        for word, tag in tagged:
            if word.isalpha():  # Only lemmatize alphabetic tokens
                pos = self._get_wordnet_pos(tag)
                lemma = self.lemmatizer.lemmatize(word.lower(), pos)
                lemmatized_tokens.append(lemma)
            else:
                lemmatized_tokens.append(word)

        # Reconstruct sentence
        return " ".join(lemmatized_tokens)

    def lemmatize_sentences(self, sentences: list[str]) -> list[str]:
        """Lemmatize a list of sentences.

        Args:
            sentences: List of sentences

        Returns:
            List of lemmatized sentences
        """
        return [self.lemmatize_sentence(sent) for sent in sentences]

    def split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        if not self.sentence_split:
            return [text]
        return nltk.sent_tokenize(text)

    def preprocess(self, text: str) -> tuple[list[str], list[list[str]]]:
        """Preprocess text into sentences and tokens.

        Returns:
            Tuple of (sentences, tokenized_sentences)
        """
        sentences = self.split_sentences(text)
        tokenized = [self.tokenizer.tokenize(sent) for sent in sentences]
        return sentences, tokenized
