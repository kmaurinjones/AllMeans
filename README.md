# AllMeans

Automatic Topic Modelling (TM) using minimal user input and computational resources. I made this because my biggest issue with most TM modules is simple. If I knew how many topics I wanted, I would already have enough information about the text, such that performing TM would be redundant. AllMeans does not aim to replace existing TM frameworks, but instead aims to tackle the aspect of required user input to derive meaningful insights. With AllMeans, the user is simply required to pass a text, and run one method, with optionally ZERO decisions.

See `Basic Modelling` example, below. AllMeans is designed to be simple, user-friendly, and practical. It doesn't invent anything that doesn't already exist in the passed text (it doesn't require loading enormous Word Embeddings models like GloVe). All that is needed is a text, in one string (no pre-processing needed), to create an AllMeans object, and to run the .model_topics() method.

### `.model_topics()` Method

There are only two arguments to the .model_topics(), `early_stop` and `verbose`. Verbosity is a boolean, offering to print progress and a glimpse of the results as the method runs, and `early_stop` strongly positively correlates with the number of resulting topics found, though it is not a 1:1 relationship (i.e., passing early_stop = 3 will not necessarily result in 3 topics). As the method largely relies on iteratively comparing various Kmeans clustering results (through an averaged silhouette_score and davies_bouldin_score - both of which sklearn's implementations), the early_stop value (default = 2) determines after how many consecutively negatively trending iterations the method stops. The motivation for this being that there is typically a certain Kmeans value that scores best, after which point scores trend downwards, making these iterations often redundant. Thus, a lower early_stop value (\~2) will significantly decrease computational expense and time, but may also change performance. As each early_stop value does not necessarily build on lower values (for example, early_stop = 3 is not necessarily the same topics as early_stop = 2, plus *x* more topics), I suggest trying 2 or 3 values (I like to test something like early_stop = \[2, 3, 4, 5\]) to see how the passed text can be represented.

## Examples

### Basic Modelling

```         
# !pip install allmeans-tm
import AllMeans

# assuming you have a text in the variable `text`
allmeans = AllMeans(text = text)
clusters = allmeans.model_topics(
    early_stop = 2, # default value
    verbose = False # default value
)
>>> returns a dict of {str : list[str]} pairs of topics and lists of all sentences relevant to each topic
```

Note:

-   As a reminder, try different values for `early_stop`. I like to try \[2, 3, 4, 5\], but keep in mind that higher values will result in exponentially larger runtime, and more topics found

### Analysis - Plotting `AllMeans` Results

This example gets the text from the "Linguistics" Wikipedia, models its topics and plots the distribution of sentences relating to each topic and the mean sentiment (using NLTK's VADER module) of the context relating to each topic.

```         
# !pip3 install wikipedia-api
import wikipediaapi
wiki_wiki = wikipediaapi.Wikipedia(USER_AGENT_HERE', 'en') # check https://pypi.org/project/Wikipedia-API/ "user_agent" to understand this
page_py = wiki_wiki.page("Linguistics") # gets the text of entire Wikipedia "Linguistics" page
text = page_py.text # returns str of entire page text -> check package docs for more useful methods

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def average_compound_sentiment(texts):
    sia = SentimentIntensityAnalyzer()
    compound_scores = [sia.polarity_scores(text)['compound'] for text in texts]
    avg_score = sum(compound_scores) / len(compound_scores) if compound_scores else 0
    return avg_score

# Use AllMeans to model topics from page
allmeans = AllMeans(text = text)
clusters = allmeans.model_topics(early_stop = 5, verbose = True)
>>> Note: there will be many printouts here due to verbose = True

# Prepare the topics-sentences distribution data and mean sentiment per topic
dist = {lab: len(sents) for lab, sents in clusters.items()}
sorted_dist = dict(sorted(dist.items(), key = lambda item: item[1], reverse = True))
avg_sentiments = {key: average_compound_sentiment(value) for key, value in clusters.items()}

# Prepare data for plotting
labels = list(sorted_dist.keys())
counts = [dist[label] for label in labels]
avg_sa = [avg_sentiments[label] for label in labels if label in avg_sentiments]

# Create figure and primary axis
fig, ax1 = plt.subplots(figsize = (8, 5))

# Plot the topics-sentences distribution
color = 'tab:blue'
ax1.set_xlabel('Topics')
ax1.set_ylabel('Sentences Per Topic')
ax1.bar(labels, counts, color = color)
ax1.tick_params(axis = 'y')
ax1.tick_params(axis = 'x', rotation = 45)

# Create a secondary axis for the average sentiment
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Average Sentiment')
ax2.plot(labels, avg_sa, color = color, marker = 'o', linestyle = '-', linewidth = 2, markersize = 5)
ax2.tick_params(axis = 'y')

# Use integer locator for sentences count axis
ax1.yaxis.set_major_locator(MaxNLocator(integer = True))

fig.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
plt.title('Topic-Sentences Distribution and Average Sentiment')
plt.show()
```

![](example.png){width="637"}