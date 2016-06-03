from pandas import DataFrame
from sklearn.base import TransformerMixin


class TextLengthTransformer(TransformerMixin):

    def transform(self, X, **transform_params):
        lengths = DataFrame(map(lambda x: len(x),X))
        return lengths

    def fit(self, X, y=None, **fit_params):
        return self

    def get_feature_names(self):
        return ['text_length']
