from pandas import DataFrame
from sklearn.base import TransformerMixin
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion

from Transformers import TextLengthTransformer

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