# Changelog

All notable changes to AllMeans will be documented in this file.

## [Released]

### v2.0.0 - 2025-11-07 - Complete Modernization

**⚠️ BREAKING CHANGES**: v2.0 is a complete rewrite with no backward compatibility.

#### Removed
- ❌ Legacy `AllMeans(text).model_topics()` API - use `TopicModel().fit(text)` instead
- ❌ `download_nltk_resources()` - now automatic
- ❌ `get_sentence_transformer_model()` - use optional `[embeddings]` extra
- ❌ Old `all_means.py` and `resources.py` modules

#### Added - Core Features
- ✅ **New `TopicModel` API** with scikit-learn-style `fit()`/`transform()`
- ✅ **Multiple clustering algorithms**: K-Means, NMF, LDA, HDBSCAN+UMAP
- ✅ **Multiple feature extractors**: TF-IDF, Bag of Words, SIF embeddings
- ✅ **Real evaluation metrics**: C_V Coherence, Topic Diversity, Silhouette, Davies-Bouldin, Calinski-Harabasz
- ✅ **Auto-K selection** via multi-objective optimization
- ✅ **POS-based keyword filtering**: Removes ordinals, numbers, and uninformative words using NLTK part-of-speech tagging
- ✅ **Lemmatization**: Normalizes word forms (singular/plural, verb tenses) for better keyword extraction
- ✅ **Verbosity controls**: `verbose` parameter with rich progress bars, `--quiet` flag in CLI
- ✅ **Deterministic results** with `random_state` throughout
- ✅ **CLI** with typer (fit, topics, version commands)
- ✅ **Structured results** via `TopicModelResults` with `Topic` objects

#### Added - Development
- ✅ Full type hints (mypy strict mode)
- ✅ Ruff for linting/formatting
- ✅ Pytest test suite
- ✅ Pre-commit hooks
- ✅ GitHub Actions CI/CD (Py 3.10-3.13)
- ✅ Auto-publish to PyPI on tag

#### Changed
- 🔄 Python requirement: `>=3.10` (was `>=3.12`)
- 🔄 Package name: lowercase `allmeans` (was `AllMeans`)
- 🔄 Architecture: Modular (`core/`, `algorithms/`, `evaluation/`)
- 🔄 Dependencies: Properly organized with optional extras
  - `[sentiment]` - VADER sentiment analysis
  - `[embeddings]` - sentence-transformers, gensim
  - `[viz]` - streamlit, plotly
  - `[all]` - all extras
  - `[dev]` - development tools

#### Migration
See [MIGRATION.md](MIGRATION.md) for detailed migration guide from v1.x to v2.0.

---

## [Previous Versions]

### Added 20240218 - v1.0.4

- Initial features and documentation pushed to PyPI. AllMeans.model_topics() is currently the main method, with args 'early_stop', 'verbose', and 'model'.

### Added 20240221 - v1.0.5

- Updated Python and dependency version requirements for Python 3.12 compatibility
- Set minimum Python version to 3.12
- Updated numpy requirement to >=1.26.0
- Updated other core dependencies to their latest stable versions

### Added 20241110 - v1.1.1

- Skipped v1.0.6-v1.1.0 due to a bugs in package updating
- Updated dependencies and fixed dependency versioning, which was causing installation issues.
- Switched to using Hatchling for building and distributing the package (pyproject.toml instead of setup.py)

### Added 20251107 - v1.1.2 (Development)

- Converted project to use `uv` package manager exclusively
- Added development dependencies to pyproject.toml (build, twine, hatchling)
- Updated update_package.sh script to use `uv` commands throughout (removed all pip references)
- Removed deprecated setup-DEP.py file
- Initialized uv.lock for reproducible dependency resolution
- Created .venv with uv for local development
