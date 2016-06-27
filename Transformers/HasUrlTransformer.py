from pandas import DataFrame
from sklearn.base import TransformerMixin

from PreProcessor import GetTextFromTweet


class HasUrlTransformer(TransformerMixin):
    #max_text_length = 0

    def transform(self, X, **transform_params):
        table = DataFrame(map(lambda x: "[URL]" in x.text, X), columns=['has_url'])
        return table

    def fit(self, X, y=None, **fit_params):
        #for text in X:
        #    if len(text)>self.max_text_length:
        #        self.max_text_length = len(text)
        return self

    def get_feature_names(self):
        return ['has_url']
