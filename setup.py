from setuptools import setup, find_packages

# README
with open("README.md", "r") as f:
    description = f.read()

# LICENSE
with open("LICENSE", "r") as f:
    license = f.read()

setup(
    name = 'AllMeans',
    version = '1.0.4',
    author = 'Kai Maurin-Jones',
    description = 'A package for fully automated topic modelling',
    packages = find_packages(),
    python_requires = '>=3.11.4',
    license = license,
    install_requires = [
        'jellyfish==1.0.3',
        'nltk==3.8.1',
        'numpy==1.24.3',
        'scikit-learn==1.3.2',
        'sentence-transformers==2.2.2',
    ],
    long_description = description,
    long_description_content_type = "text/markdown",    
)
