from pandas import DataFrame
from sklearn.base import TransformerMixin


class TextLengthTransformer(TransformerMixin):
    #max_text_length = 0

    def transform(self, X, **transform_params):
        lengths = DataFrame(map(lambda x: len(x.text),X),columns=['text_length'])
        return lengths

    def fit(self, X, y=None, **fit_params):
        #for text in X:
        #    if len(text)>self.max_text_length:
        #        self.max_text_length = len(text)
        return self

    def get_feature_names(self):
        return ['text_length']
