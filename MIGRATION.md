# AllMeans v2.0 Migration Guide

## ⚠️ Breaking Changes

AllMeans v2.0 is a **complete rewrite** with breaking changes. The old API has been removed.

### What's Removed

- ❌ `AllMeans(text).model_topics()` - Use `TopicModel().fit(text)` instead
- ❌ `download_nltk_resources()` - Happens automatically
- ❌ `get_sentence_transformer_model()` - Use optional `[embeddings]` extra

### Migration Path

**Before (v1.x)**:
```python
from AllMeans import AllMeans

model = AllMeans(text)
clusters = model.model_topics(early_stop=3, exclusions=["word"])

# Returns: dict[str, list[str]]
for label, sentences in clusters.items():
    print(f"{label}: {len(sentences)} sentences")
```

**After (v2.0)**:
```python
from AllMeans import TopicModel

model = TopicModel(
    method="kmeans",       # or "nmf", "lda", "hdbscan"
    auto_k=True,           # auto-select K
    early_stop=3,          # same concept
    exclusions=["word"],   # same
    random_state=42,       # deterministic results
)

model.fit(text)
results = model.get_results()

# Returns: TopicModelResults with structured data
for topic in results.topics:
    print(f"{topic.label}: {topic.size} sentences")
    print(f"  Keywords: {topic.keywords}")
    print(f"  Coherence: {topic.coherence:.3f}")
```

## New API

### Basic Usage

```python
from AllMeans import TopicModel

# Fit model
model = TopicModel(method="kmeans", random_state=42)
model.fit(text)

# Get structured results
results = model.get_results()

# Access topics
for topic in results.topics:
    print(f"Topic {topic.id}: {topic.label}")
    print(f"  Keywords: {', '.join(topic.keywords)}")
    print(f"  Size: {topic.size}")
    print(f"  Coherence: {topic.coherence:.3f}")
    print(f"  Example: {topic.exemplar_sentences[0]}")

# Access metrics
print(f"\nSilhouette: {results.scores['silhouette']:.3f}")
print(f"Coherence: {results.scores['coherence']:.3f}")
print(f"Diversity: {results.scores['diversity']:.3f}")
```

### Multiple Algorithms

```python
# K-Means (default, fast)
model = TopicModel(method="kmeans")

# NMF (good for interpretability)
model = TopicModel(method="nmf")

# LDA (probabilistic)
model = TopicModel(method="lda")

# HDBSCAN (density-based, auto K)
model = TopicModel(method="hdbscan")
```

### Feature Extraction

```python
# TF-IDF (default)
model = TopicModel(feature_method="tfidf")

# Bag of Words
model = TopicModel(feature_method="bow")

# SIF embeddings (requires [embeddings] extra)
model = TopicModel(feature_method="sif")
```

### CLI

```bash
# Fit model
allmeans fit --input text.txt --output results.json \
    --method kmeans --features tfidf

# View topics
allmeans topics --results results.json --topn 10

# Check version
allmeans version
```

## New Features

### 1. Real Metrics
- C_V Coherence (gensim)
- Topic Diversity
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Score

### 2. Auto-K Selection
```python
model = TopicModel(
    auto_k=True,
    k_range=(2, 10),
    early_stop=2,
)
```

### 3. Deterministic Results
```python
model = TopicModel(random_state=42)
# Same results every time
```

### 4. Transform New Data
```python
model.fit(training_text)
assignments = model.transform(new_text)
```

### 5. Structured Results
```python
results = model.get_results()
results.topics          # List[Topic]
results.assignments     # numpy array
results.scores          # dict[str, float]
results.sentences       # list[str]
```

## Installation

```bash
# Core (classical methods only)
uv add allmeans

# With sentiment
uv add allmeans[sentiment]

# With embeddings
uv add allmeans[embeddings]

# With visualization
uv add allmeans[viz]

# Everything
uv add allmeans[all]

# Development
uv add allmeans[dev]
```

## Requirements

- Python 3.10+
- No LLMs required (all classical methods)
- Optional: sentence-transformers for embeddings

## Need Help?

- See `examples/basic_usage.py` for complete example
- Check tests in `tests/test_api.py`
- Open issue at https://github.com/kmaurinjones/AllMeans/issues
