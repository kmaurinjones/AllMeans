from setuptools import setup, find_packages

setup(
    name = 'allmeans-tm',
    version = '0.1.0',
    author = 'Kai Maurin-Jones',
    description = 'A package for automatic topic modelling',
    packages = find_packages(),
    python_requires = '>=3.11.4',
    install_requires = [
        'nltk==3.8.1',
        'numpy==1.24.3',
        'scikit-learn==1.3.2',
        'sentence-transformers==2.2.2',
    ],
)
