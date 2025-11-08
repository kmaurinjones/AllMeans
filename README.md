# AllMeans

**Modern topic modeling with minimal user input.** AllMeans v2.0 provides automatic topic discovery using state-of-the-art clustering algorithms, real evaluation metrics, and intelligent keyword extraction with part-of-speech filtering and lemmatization.

[![PyPI version](https://badge.fury.io/py/allmeans.svg)](https://pypi.org/project/allmeans/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Features

- **Multiple Clustering Algorithms**: K-Means, NMF, LDA, HDBSCAN+UMAP
- **Automatic K Selection**: Multi-objective optimization finds optimal number of topics
- **Smart Keyword Extraction**: POS-based filtering removes ordinals, numbers, and uninformative words
- **Lemmatization**: Normalizes word forms (singular/plural, verb tenses) for better topic quality
- **Real Evaluation Metrics**: C_V Coherence, Topic Diversity, Silhouette, Davies-Bouldin
- **Scikit-learn API**: Familiar `fit()`/`transform()` pattern
- **Verbosity Controls**: Rich progress bars and detailed output options
- **CLI Interface**: Command-line tool for quick topic modeling

## Installation

```bash
pip install allmeans
```

Or with `uv`:
```bash
uv add allmeans
```

### Optional Dependencies

```bash
# Sentiment analysis
pip install allmeans[sentiment]

# Embeddings support (sentence-transformers, gensim)
pip install allmeans[embeddings]

# Visualization tools
pip install allmeans[viz]

# All extras
pip install allmeans[all]
```

## Quick Start

### Basic Usage

```python
from AllMeans import TopicModel

# Your text
text = """
Machine learning is a subset of artificial intelligence.
Deep learning uses neural networks with multiple layers.
Natural language processing helps computers understand human language.
Computer vision enables machines to interpret visual information.
"""

# Create and fit model
model = TopicModel(
    method="kmeans",           # or "nmf", "lda", "hdbscan"
    feature_method="tfidf",    # or "bow", "sif"
    auto_k=True,               # automatically find optimal K
    k_range=(2, 10),          # range to search
    verbose=True               # show progress
)

model.fit(text)

# Get results
results = model.get_results()

# Print discovered topics
for topic in results.topics:
    print(f"\n📌 {topic.label}")
    print(f"   Keywords: {', '.join(topic.keywords[:5])}")
    print(f"   Size: {topic.size} sentences")
    print(f"   Coherence: {topic.coherence:.3f}")
```

### Working with Documents

```python
# List of documents instead of single text
documents = [
    "Python is a high-level programming language.",
    "JavaScript is popular for web development.",
    "Machine learning models require training data.",
    "Data science combines statistics and programming.",
]

model = TopicModel(n_clusters=2, auto_k=False)
model.fit(documents)

# Transform new documents
new_docs = ["Deep learning is a subset of machine learning."]
assignments = model.transform(new_docs)
```

### Command-Line Interface

```bash
# Fit model on text file
allmeans fit --input article.txt --method kmeans --verbose

# With custom parameters
allmeans fit \
    --input data.txt \
    --output results.json \
    --method hdbscan \
    --features tfidf \
    --clusters 5

# View topics from saved results
allmeans topics --results results.json
```

## Advanced Examples

### Wikipedia Article Analysis

```python
import requests
from bs4 import BeautifulSoup
from AllMeans import TopicModel

# Fetch Wikipedia article
url = "https://en.wikipedia.org/wiki/Roman_Empire"
response = requests.get(url, headers={
    "User-Agent": "AllMeans/2.0"
})
soup = BeautifulSoup(response.content, 'html.parser')
content = soup.find('div', {'id': 'mw-content-text'})
paragraphs = content.find_all('p')
text = ' '.join([p.get_text() for p in paragraphs if p.get_text().strip()])

# Model topics with auto-K selection
model = TopicModel(
    method="kmeans",
    feature_method="tfidf",
    auto_k=True,
    k_range=(3, 8),
    early_stop=2,
    random_state=42,
    verbose=True
)

model.fit(text)
results = model.get_results()

# Display results
print(f"\nDiscovered {len(results.topics)} topics:")
for topic in results.topics:
    print(f"\n{topic.label} ({topic.size} sentences)")
    print(f"Keywords: {', '.join(topic.keywords)}")
    print(f"Example: {topic.exemplar_sentences[0][:100]}...")
```

### Custom Exclusions

```python
# Exclude specific words from keywords
model = TopicModel(
    exclusions=["said", "also", "however"],
    excl_sim=0.9,  # Jaro-Winkler similarity threshold
    filter_pos=True  # Enable POS filtering
)

model.fit(text)
```

### Evaluation Metrics

```python
results = model.get_results()

print("Evaluation Metrics:")
print(f"Coherence (C_V): {results.scores['coherence']:.3f}")
print(f"Diversity: {results.scores['diversity']:.3f}")
print(f"Silhouette: {results.scores['silhouette']:.3f}")
print(f"Davies-Bouldin: {results.scores['davies_bouldin']:.3f}")
```

## API Reference

### TopicModel

```python
TopicModel(
    method="kmeans",              # Clustering: "kmeans", "nmf", "lda", "hdbscan"
    feature_method="tfidf",       # Features: "tfidf", "bow", "sif"
    n_clusters=None,              # Fixed K (None for auto)
    auto_k=True,                  # Auto-select K
    k_range=(2, 10),             # K range to search
    early_stop=2,                 # Early stopping patience
    exclusions=None,              # Words to exclude
    excl_sim=0.9,                # Exclusion similarity threshold
    filter_pos=True,              # POS-based filtering
    random_state=42,              # Random seed
    verbose=False                 # Show progress
)
```

**Methods:**
- `fit(text)` - Fit model on text or documents
- `transform(text)` - Predict topics for new text
- `fit_transform(text)` - Fit and transform in one step
- `get_results()` - Get TopicModelResults object

### TopicModelResults

**Attributes:**
- `topics` - List of Topic objects
- `assignments` - Cluster assignments for each sentence
- `scores` - Dictionary of evaluation metrics
- `config` - Model configuration
- `feature_matrix` - TF-IDF or other features
- `sentences` - Original sentences

### Topic

**Attributes:**
- `id` - Topic ID
- `label` - Topic label (top keyword)
- `keywords` - List of keywords
- `size` - Number of sentences
- `coherence` - Topic coherence score
- `diversity` - Topic diversity score
- `exemplar_sentences` - Example sentences

## Migration from v1.x

v2.0 is a complete rewrite with breaking changes. See [MIGRATION.md](MIGRATION.md) for detailed migration guide.

**Quick migration:**

```python
# v1.x (deprecated)
from AllMeans import AllMeans
allmeans = AllMeans(text=text)
clusters = allmeans.model_topics(early_stop=2, verbose=False)

# v2.0 (current)
from AllMeans import TopicModel
model = TopicModel(early_stop=2, verbose=False)
model.fit(text)
results = model.get_results()
```

## Performance

AllMeans v2.0 has been tested on texts ranging from 1,000 to 100,000+ characters. Performance scales with text size:

- **Small texts** (1K-10K chars): < 1 second
- **Medium texts** (10K-50K chars): 1-5 seconds
- **Large texts** (50K-100K chars): 5-30 seconds

Auto-K selection adds overhead proportional to `k_range` size. Use `early_stop` to reduce computation time.

## How It Works

1. **Preprocessing**: Text is split into sentences and lemmatized
2. **Feature Extraction**: TF-IDF, BoW, or SIF embeddings
3. **Clustering**: K-Means, NMF, LDA, or HDBSCAN groups similar sentences
4. **Auto-K Selection** (optional): Tests multiple K values, selects optimal via metrics
5. **Keyword Extraction**: TF-IDF scores filtered by POS tags and lemmatization
6. **Label Selection**: Most diverse keywords chosen as topic labels
7. **Evaluation**: Coherence, diversity, and clustering metrics computed

## Contributing

Contributions welcome! Please open an issue or submit a pull request on [GitHub](https://github.com/kmaurinjones/AllMeans).

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use AllMeans in research, please cite:

```bibtex
@software{allmeans2024,
  author = {Maurin-Jones, Kai},
  title = {AllMeans: Modern Topic Modeling with Minimal User Input},
  year = {2024},
  url = {https://github.com/kmaurinjones/AllMeans},
  version = {2.0.0}
}
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and updates.
