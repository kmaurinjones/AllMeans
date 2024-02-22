from jellyfish import jaro_winkler_similarity
import numpy as np
import nltk
from nltk.corpus import names, stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from itertools import product
from sklearn.metrics.pairwise import cosine_similarity
from .resources import download_nltk_resources, get_sentence_transformer_model
import warnings

class AllMeans:
    """
    A class for performing automatic topic modeling on text data using K-means clustering,
    TF-IDF vectorization, and word embeddings to identify and model topics within the text.

    Attributes:
        text (str): The input text to be analyzed.
        word_embeddings (dict): A dictionary to store word embeddings.
        avg_scores (list): A list to keep track of average scores for each number of clusters analyzed.
        documents (list): A list of sentences tokenized from the input text.
        all_names (set): A set containing all male and female names to exclude from topic modeling.
        clusters (dict): A dictionary mapping identified topics to lists of sentences related to those topics.

    Args:
        text (str): The input text to be analyzed.
        ignore_warnings (bool, optional): If True, ignores user warnings. Default is True.
    """

    def __init__(self, text, ignore_warnings = True):
        """
        Initializes the AllMeans object, tokenizes the input text into sentences,
        downloads required NLTK resources, loads names for filtering, and optionally
        ignores warnings based on user input.
        """
        self.text = text
        self.word_embeddings = {}
        self.avg_scores = []
        self.documents = nltk.sent_tokenize(text)
        download_nltk_resources()
        self.all_names = self._load_names()
        self.clusters = {}

        if ignore_warnings:
            warnings.filterwarnings("ignore", category=UserWarning)

    @staticmethod
    def _load_names():
        """
        Loads male and female names from the NLTK corpus and combines them into a single set.

        Returns:
            set: A set of lowercased names combining male and female names.
        """        
        male_names = names.words('male.txt')
        female_names = names.words('female.txt')
        all_names = set([name.lower() for name in male_names + female_names])
        return all_names

    @staticmethod
    def get_wordnet_pos(treebank_tag):
        """
        Maps NLTK's treebank POS tags to WordNet's POS tags for lemmatization purposes.

        Args:
            treebank_tag (str): A POS tag from NLTK's part-of-speech tagging.

        Returns:
            str: A WordNet POS tag.
        """        
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def custom_tokenizer(self, text): # used lambda function instead of decorator
        """
        Custom tokenizer that tokenizes, removes stop words, lemmatizes,
        and filters out names from the text.

        Args:
            text (str): The text to tokenize and process.

        Returns:
            list: A list of processed tokens from the input text.
        """        
        lemmatizer = WordNetLemmatizer()
        tokens = nltk.word_tokenize(text)
        tokens = [token for token in tokens if token.isalpha()]
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        tagged_tokens = nltk.pos_tag(tokens)
        noun_tokens = [word for word, tag in tagged_tokens if tag in ('NN', 'NNS', 'NNP', 'NNPS')]
        non_name_nouns = [word for word in noun_tokens if word.lower() not in self.all_names]
        lemmatized_nouns = [lemmatizer.lemmatize(word, self.get_wordnet_pos(tag)) for word, tag in tagged_tokens if word in non_name_nouns]
        return lemmatized_nouns

    @staticmethod
    def calculate_dissimilarity(selected_words, embeddings):
        """
        Calculates the total dissimilarity between selected words based on their embeddings.

        Args:
            selected_words (list): A list of words to calculate dissimilarity for.
            embeddings (dict): A dictionary containing embeddings for the selected words.

        Returns:
            float: The total dissimilarity score for the selected words.
        """
        embeddings = [embeddings[word] for word in selected_words]
        sim_matrix = cosine_similarity(embeddings)
        distance_matrix = 1 - sim_matrix
        total_dissimilarity = np.sum(np.triu(distance_matrix, k=1))
        return total_dissimilarity

    def model_topics(self, early_stop = 2, exclusions: list[str] = [], excl_sim: float = 0.9, verbose = False, model = "distilbert-base-nli-stsb-mean-tokens"):
        """
        Models topics from the text using TF-IDF, K-means clustering, and silhouette and Davies-Bouldin scores
        to evaluate clustering performance. Stops modeling when performance worsens for a specified
        number of iterations (early stopping). Allows excluding specific words from the topics based on
        a Jaro-Winkler similarity threshold with any words in the exclusions list, while ensuring the intended
        number of strings in each cluster is maintained after exclusions.

        Args:
            early_stop (int, optional): The number of iterations of worsening scores before stopping. Default is 2.
            exclusions (list[str], optional): List of words to exclude from the topics based on Jaro-Winkler similarity. Default is an empty list.
            excl_sim (float, optional): Jaro-Winkler Similarity threshold below which a potential cluster topic must be to not be excluded.
            verbose (bool, optional): If True, prints detailed progress information. Default is False.
            model (str, optional): See HuggingFace SentenceTransformers library for list of models that can be passed here. Default is "distilbert-base-nli-stsb-mean-tokens".

        Returns:
            dict: A dictionary mapping identified topics (as strings) to lists of sentences.
        """    
        assert 1 < early_stop, "You must enter an integer greater than 1 for `early_stop`. Please try again."
        self.avg_scores = []

        # Process exclusions: lowercase, strip, and remove duplicates
        exclusions = list(set(word.lower().strip() for word in exclusions))

        model = get_sentence_transformer_model(model=model)
        worsening_scores = 0

        for idx, num_clusters in enumerate(range(2, 10)):
            tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda text: self.custom_tokenizer(text), stop_words='english', min_df=1, max_features=20000)
            tfidf_matrix = tfidf_vectorizer.fit_transform(self.documents)
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
            kmeans.fit(tfidf_matrix)
            clusters = kmeans.labels_
            silhouette_avg = silhouette_score(tfidf_matrix, clusters)
            davies_bouldin = davies_bouldin_score(tfidf_matrix.toarray(), clusters)
            avg_score = np.mean([silhouette_avg, davies_bouldin])
            self.avg_scores.append(avg_score)

            if len(self.avg_scores) > 1:
                if avg_score < self.avg_scores[-2]:
                    worsening_scores += 1
                else:
                    worsening_scores = 0

            if verbose:
                print(f"Clusters: {num_clusters}. Score: {round(self.avg_scores[idx], 4)}")

            if worsening_scores >= early_stop:
                if verbose:
                    print("-" * 50)
                    print(f"Early stop triggered. Logged scores: {[round(scr, 4) for scr in self.avg_scores]}")
                break

            order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
            terms = tfidf_vectorizer.get_feature_names_out()
            initial_cluster_size = 5
            my_clusters = [[terms[ind] for ind in order_centroids[i, :initial_cluster_size]] for i in range(num_clusters)]

            for cluster_index, cluster in enumerate(my_clusters):
                filtered_cluster = [term for term in cluster if not any(jaro_winkler_similarity(term, exclusion) > excl_sim for exclusion in exclusions)]
                term_index = initial_cluster_size
                while len(filtered_cluster) < initial_cluster_size:
                    if term_index >= len(terms):
                        break  # Break if there are no more terms to check
                    term = terms[order_centroids[cluster_index, term_index]]
                    term_index += 1
                    if not any(jaro_winkler_similarity(term, exclusion) > excl_sim for exclusion in exclusions):
                        filtered_cluster.append(term)
                my_clusters[cluster_index] = filtered_cluster[:initial_cluster_size]

            best_selection = None
            best_score = -np.inf
            unique_words = set(word for cluster in my_clusters for word in cluster)
            self.word_embeddings.update({word: model.encode(word) for word in unique_words if word not in self.word_embeddings})

            for selection in product(*my_clusters):
                score = self.calculate_dissimilarity(selection, self.word_embeddings)
                if score > best_score:
                    best_score = score
                    best_selection = selection

        if best_selection:
            if verbose:
                print(f"Best selection before assigning clusters: {best_selection}")
            self._assign_clusters_to_sentences(clusters, best_selection, verbose)

        return self.clusters
    
    def _assign_clusters_to_sentences(self, clusters, best_selection, verbose):
        """
        Assigns clusters to sentences based on the best selection of words representing each cluster.

        Args:
            clusters (list): A list of cluster labels for each sentence.
            best_selection (tuple): A tuple of words representing the best selection for each cluster.
            verbose (bool): If True, prints detailed information about the assignment process.

        No explicit return value, but updates the `clusters` attribute of the class instance.
        """
        clustered_documents = {}
        for cluster_index in np.unique(clusters):
            clustered_documents[cluster_index] = []

        for document, cluster in zip(self.documents, clusters):
            clustered_documents[cluster].append(document)

        if verbose:
            print("-" * 50)
            for cluster_index in clustered_documents.keys():
                print(f"Cluster {cluster_index}: {len(clustered_documents[cluster_index])} documents assigned.")
            print("-" * 50)

        # Ensure that the indexing of best_selection matches the unique cluster indices
        for cluster_index in clustered_documents.keys():
            if cluster_index < len(best_selection):  # Check to prevent index out of range
                self.clusters[best_selection[cluster_index]] = clustered_documents[cluster_index]
                if verbose:
                    print(f"\nCluster {cluster_index}: '{best_selection[cluster_index]}': {len(clustered_documents[cluster_index])} documents")
                    for doc in clustered_documents[cluster_index][:5]:  # Show some example sentences
                        print(f" - {doc}")
                    print("\n---")
        if verbose:
            print("-" * 50)
            print(f"\nClusters have been created and can be accessed by calling .clusters attribute. Returns dict of str : list[str] pairs of labels : sentences.")
