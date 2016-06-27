from pandas import DataFrame
from sklearn.base import TransformerMixin
from sklearn.feature_extraction import DictVectorizer

class UsernameTransformer(TransformerMixin):

    vec=None

    def transform(self, X, **transform_params):
        user_names = [{'username': x.screen_name} for x in X]
        array = self.vec.fit_transform(user_names).toarray()
        table = DataFrame(array,columns=self.get_feature_names())
        return table

    def fit(self, X, y=None, **fit_params):
        self.vec = DictVectorizer()
        return self

    def get_feature_names(self):
        return self.vec.get_feature_names()
