from pandas import DataFrame
from sklearn.base import TransformerMixin
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion

categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']

twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, random_state=42)

count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()

class TextLengthTransformer(TransformerMixin):

    def transform(self, X, **transform_params):
        lengths = DataFrame(map(lambda x: len(x),X))
        return lengths

    def fit(self, X, y=None, **fit_params):
        return self

pipeline = Pipeline([
  ('features', FeatureUnion([
    ('ngram_tf_idf', Pipeline([
      ('counts', CountVectorizer()),
      ('tf_idf', TfidfTransformer())
    ])),
    ('text_length', TextLengthTransformer())
  ])),
  ('classifier', MultinomialNB())
])

classifier = pipeline.fit(twenty_train.data, twenty_train.target)

# Prediction
docs_new = ['God is love', 'OpenGL on the GPU is fast']
predicted = classifier.predict(docs_new)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))

stop = 5