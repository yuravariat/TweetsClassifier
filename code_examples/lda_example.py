from lda import lda
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from classifier.data import DataAdapter
from classifier.pre_processor import GetTextFromTweet
from pandas import np

disease = 'hiv'
categories = ['celeb', 'dont_know', 'family', 'himself', 'knows', 'none', 'subject']
dataAdapter = DataAdapter(disease)

# 1. Generate training set by splitting the input files multiple files (file per tweet)
dataAdapter.create_data(disease)

# 2. Load train data from files or cache
trainData = dataAdapter.get_data(categories=categories, subset='train')

vectorizer = CountVectorizer(ngram_range=(1, 1), stop_words='english', preprocessor=GetTextFromTweet)
matrix = vectorizer.fit_transform(trainData.data)
feature_names = vectorizer.get_feature_names()

vocab = feature_names

model = lda.LDA(n_topics=10, n_iter=500, random_state=1)
model.fit(matrix)
n_top_words = 10

for i, topic_dist in enumerate(model.topic_word_):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))