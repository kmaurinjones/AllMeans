from setuptools import setup, find_packages

with open("README.md", "r") as f:
    description = f.read()

setup(
    name = 'allmeans-tm',
    version = '0.2.0',
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
    long_description = description,
    long_description_content_type = "text/markdown",    
)
