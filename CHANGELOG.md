# Changelog

All notable changes to this AllMeans will be documented in this file.

## [Future Additions]

- Nothing planned at the moment. Raise a Github issue at https://github.com/kmaurinjones/AllMeans/issues if there are features you would like to see added, in addition to specific methodological suggestions, if possible.

## [Released]

### Added 20240221

- Added `exclusions: list[str]` and `excl_sim` arguments to model.topics() method, allowing user to optionally pass a list of words to exclude from possible cluster labels (topics) (if any potential word has a Jaro Winkler Similarity - calculated using the Jellyfish implementation - above `excl_sim`). The impetus for this is that occasionally, a name (if not in NLTK's NAMES list), otherwise uninteresting word (such as "thing"), or painfully obvious label (something already known to the user) will be chosen. Suggested usage of this argument is to first pass nothing to it (the arg default is an empty list), and to iteratively and incrementally add more words to the passed list until chosen cluster labels are satisfactory.

### Added 20240218

- Initial features and documentation pushed to PyPI. AllMeans.model_topics() is currently the main method, with args 'early_stop', 'verbose', and 'model'.
