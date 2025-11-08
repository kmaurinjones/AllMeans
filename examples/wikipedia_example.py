"""Example: Topic modeling on Wikipedia article about the Roman Empire."""

import matplotlib.pyplot as plt
import numpy as np
import requests
from bs4 import BeautifulSoup

from AllMeans import TopicModel


def fetch_wikipedia_text(url: str, verbose: bool = True) -> str:
    """Fetch and extract main text content from Wikipedia article."""
    if verbose:
        print(f"Fetching: {url}")
    headers = {
        "User-Agent": "AllMeans/2.0 (https://github.com/kmaurinjones/AllMeans) Python/requests"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract main content paragraphs
    content_div = soup.find('div', {'id': 'mw-content-text'})
    paragraphs = content_div.find_all('p') if content_div else []

    # Combine paragraph text
    text = ' '.join([p.get_text() for p in paragraphs if p.get_text().strip()])

    if verbose:
        print(f"Extracted {len(text)} characters, {len(paragraphs)} paragraphs")
    return text


def plot_topic_distribution(results, save_path: str = "topic_distribution.png", verbose: bool = True) -> None:
    """Plot the size distribution of topics."""
    fig, ax = plt.subplots(figsize=(12, 6))

    topic_labels = [f"{t.label[:30]}..." if len(t.label) > 30 else t.label for t in results.topics]
    topic_sizes = [t.size for t in results.topics]

    bars = ax.barh(topic_labels, topic_sizes, color='steelblue', alpha=0.7)
    ax.set_xlabel('Number of Sentences', fontsize=12)
    ax.set_ylabel('Topic', fontsize=12)
    ax.set_title('Topic Distribution - Roman Empire Wikipedia Article', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels on bars
    for i, (bar, size) in enumerate(zip(bars, topic_sizes)):
        ax.text(size + 0.5, i, str(size), va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if verbose:
        print(f"✓ Saved: {save_path}")


def plot_topic_keywords(results, top_n: int = 5, save_path: str = "topic_keywords.png", verbose: bool = True) -> None:
    """Plot top keywords for each topic."""
    n_topics = len(results.topics)
    fig, axes = plt.subplots(n_topics, 1, figsize=(14, 3 * n_topics))

    if n_topics == 1:
        axes = [axes]

    for idx, topic in enumerate(results.topics):
        ax = axes[idx]
        keywords = topic.keywords[:top_n]

        # Create color gradient
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(keywords)))

        ax.barh(range(len(keywords)), range(len(keywords), 0, -1), color=colors, alpha=0.7)
        ax.set_yticks(range(len(keywords)))
        ax.set_yticklabels(keywords, fontsize=11)
        ax.set_xlabel('Importance (Rank)', fontsize=10)
        ax.set_title(f'Topic {topic.id}: {topic.label}\n({topic.size} sentences, coherence: {topic.coherence:.3f})',
                     fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.set_xticks([])

        # Add keyword labels
        for i, keyword in enumerate(keywords):
            ax.text(len(keywords) - i - 0.1, i, f'#{i+1}',
                   ha='right', va='center', fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if verbose:
        print(f"✓ Saved: {save_path}")


def plot_metrics(results, save_path: str = "metrics.png", verbose: bool = True) -> None:
    """Plot model evaluation metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = results.scores
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    # Normalize metrics to 0-1 range for better visualization
    # Some metrics are better when higher, some when lower
    normalized = []
    colors = []
    for name, value in zip(metric_names, metric_values):
        if name in ['davies_bouldin']:  # Lower is better
            # Invert and normalize
            norm_val = max(0, 1 - (value / 3))  # Assuming typical range
            colors.append('salmon')
        else:  # Higher is better
            norm_val = min(1, max(0, value))
            colors.append('steelblue')
        normalized.append(norm_val)

    bars = ax.bar(metric_names, normalized, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Normalized Score (0-1)', fontsize=12)
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_title('Topic Model Evaluation Metrics', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

    # Add actual values on top of bars
    for bar, name, value in zip(bars, metric_names, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if verbose:
        print(f"✓ Saved: {save_path}")


def main() -> None:
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Topic modeling on Wikipedia article")
    parser.add_argument("--url", default="https://en.wikipedia.org/wiki/Roman_Empire",
                        help="Wikipedia URL to analyze")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed progress and output")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress all output except errors")
    args = parser.parse_args()

    # Validate flags
    if args.verbose and args.quiet:
        print("Error: Cannot use --verbose and --quiet together")
        return

    verbose = args.verbose and not args.quiet

    # Fetch Wikipedia text
    text = fetch_wikipedia_text(args.url, verbose=verbose)

    if verbose:
        print("\n" + "="*70)
        print("TOPIC MODELING: Wikipedia Article")
        print("="*70)

    # Create and fit model
    model = TopicModel(
        method="kmeans",
        feature_method="tfidf",
        auto_k=True,
        k_range=(3, 8),
        early_stop=2,
        random_state=42,
        verbose=verbose,
    )

    model.fit(text)
    results = model.get_results()

    # Display results
    if verbose:
        print(f"\n{'='*70}")
        print(f"DISCOVERED {len(results.topics)} TOPICS")
        print(f"{'='*70}\n")

        for topic in results.topics:
            print(f"📌 Topic {topic.id}: {topic.label}")
            print(f"   Keywords: {', '.join(topic.keywords[:8])}")
            print(f"   Size: {topic.size} sentences")
            print(f"   Coherence: {topic.coherence:.3f}")
            print(f"   Diversity: {topic.diversity:.3f}")
            if topic.exemplar_sentences:
                example = topic.exemplar_sentences[0][:150]
                print(f"   Example: {example}...")
            print()

        # Display metrics
        print(f"{'='*70}")
        print("EVALUATION METRICS")
        print(f"{'='*70}")
        for metric, value in results.scores.items():
            print(f"  {metric:20s}: {value:.4f}")
        print()

        # Create visualizations
        print(f"{'='*70}")
        print("GENERATING VISUALIZATIONS")
        print(f"{'='*70}\n")

    # Generate plots
    plot_topic_distribution(results, "roman_empire_topic_distribution.png", verbose=verbose)
    plot_topic_keywords(results, top_n=8, save_path="roman_empire_keywords.png", verbose=verbose)
    plot_metrics(results, "roman_empire_metrics.png", verbose=verbose)

    if verbose:
        print(f"\n{'='*70}")
        print("✅ COMPLETE")
        print(f"{'='*70}")
        print("\nGenerated files:")
        print("  - roman_empire_topic_distribution.png")
        print("  - roman_empire_keywords.png")
        print("  - roman_empire_metrics.png")


if __name__ == "__main__":
    main()
