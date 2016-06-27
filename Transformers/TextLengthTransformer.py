from pandas import DataFrame
from sklearn.base import TransformerMixin


class TextLengthTransformer(TransformerMixin):
    max_text_length = 0

    def transform(self, X, **transform_params):
        table = DataFrame(map(lambda x: len(x.text)/self.max_text_length,X),columns=['text_length'])
        return table

    def fit(self, X, y=None, **fit_params):
        for x in X:
            if len(x.text)>self.max_text_length:
                self.max_text_length = len(x.text)
        return self

    def get_feature_names(self):
        return ['text_length']
