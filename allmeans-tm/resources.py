import nltk
from sentence_transformers import SentenceTransformer

_nltk_resources_downloaded = False
_sentence_transformer_model = None

def download_nltk_resources():
    global _nltk_resources_downloaded
    if not _nltk_resources_downloaded:
        nltk.download('averaged_perceptron_tagger')
        nltk.download('names')
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('stopwords')
        _nltk_resources_downloaded = True

def get_sentence_transformer_model(model):
    global _sentence_transformer_model
    if _sentence_transformer_model is None:
        _sentence_transformer_model = SentenceTransformer(model) # you can change this to whichever embeddings model you'd like, but this gets good general performance
    return _sentence_transformer_model
