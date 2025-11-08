"""Basic usage example for AllMeans."""

from AllMeans import TopicModel

# Sample text about machine learning
text = """
Machine learning is a subset of artificial intelligence that focuses on building systems
that learn from data. Deep learning, a subfield of machine learning, uses neural networks
with multiple layers to model complex patterns.

Natural language processing enables computers to understand, interpret, and generate
human language. It powers applications like chatbots, translation services, and
sentiment analysis tools.

Computer vision allows machines to interpret and understand visual information from
the world. It's used in facial recognition, autonomous vehicles, and medical image
analysis.

Reinforcement learning trains agents to make decisions by rewarding desired behaviors
and penalizing undesired ones. It's particularly useful for robotics, game playing,
and autonomous systems.

Supervised learning algorithms learn from labeled training data, making predictions
based on past examples. Common applications include spam detection, image classification,
and price prediction.

Unsupervised learning discovers hidden patterns in unlabeled data without explicit
guidance. It's useful for customer segmentation, anomaly detection, and data exploration.

Transfer learning allows models trained on one task to be adapted for related tasks,
saving time and computational resources. It's widely used in computer vision and NLP.
"""

# Create and fit model
model = TopicModel(
    method="kmeans",
    feature_method="tfidf",
    auto_k=True,
    k_range=(2, 6),
    early_stop=2,
    random_state=42,
)

print("Fitting topic model...")
model.fit(text)

# Get results
results = model.get_results()

# Display topics
print(f"\nFound {len(results.topics)} topics:\n")

for topic in results.topics:
    print(f"Topic {topic.id}: {topic.label}")
    print(f"  Keywords: {', '.join(topic.keywords[:5])}")
    print(f"  Size: {topic.size} sentences")
    print(f"  Example: {topic.exemplar_sentences[0][:100]}...")
    print()

# Display metrics
print("Metrics:")
for metric, value in results.scores.items():
    print(f"  {metric}: {value:.3f}")
