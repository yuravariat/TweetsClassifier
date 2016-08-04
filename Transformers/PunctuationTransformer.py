from pandas import DataFrame, np
from sklearn.base import TransformerMixin
from sklearn.feature_extraction import DictVectorizer
import string


class PunctuationTransformer(TransformerMixin):
    punctuations=[]

    def transform(self, X, **transform_params):
        numpy_table = np.zeros((len(X), len(self.punctuations)))


        for row_index,tweet in enumerate(X):
            for punct_index, punct in enumerate(self.punctuations):
                if punct in tweet.text:
                    numpy_table[row_index, punct_index] = 1
                    break

        table = DataFrame(numpy_table,columns=self.punctuations)
        return table

    def fit(self, X, y=None, **fit_params):
        self.punctuations = list([c for c in string.punctuation])
        return self

    def get_feature_names(self):
        return self.punctuations
