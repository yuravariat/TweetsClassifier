from pandas import DataFrame
from sklearn.base import TransformerMixin


class UsernameTransformer(TransformerMixin):

    def transform(self, X, **transform_params):
        lengths = DataFrame(map(lambda x: x.screen_name,X),columns=['username'])
        return lengths

    def fit(self, X, y=None, **fit_params):
        return self

    def get_feature_names(self):
        return ['username']
