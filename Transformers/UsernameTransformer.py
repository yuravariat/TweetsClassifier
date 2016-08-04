from pandas import DataFrame, np
from sklearn.base import TransformerMixin
from sklearn.feature_extraction import DictVectorizer


class UsernameTransformer(TransformerMixin):

    user_names=[]

    def transform(self, X, **transform_params):
        numpy_table = np.zeros((len(X), len(self.user_names)))

        for row_index,tweet in enumerate(X):
            for name_index,name in enumerate(self.user_names):
                if name==tweet.screen_name:
                    numpy_table[row_index,name_index] = 1
                    break

        table = DataFrame(numpy_table,columns=self.user_names)
        return table

    def fit(self, X, y=None, **fit_params):
        self.user_names = list([x.screen_name for x in X])
        return self

    def get_feature_names(self):
        return self.user_names
