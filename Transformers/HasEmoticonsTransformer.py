from pandas import DataFrame
from sklearn.base import TransformerMixin
import re

class HasEmoticonsTransformer(TransformerMixin):

    def transform(self, X, **transform_params):

        emoji_pattern = r'/[U0001F601-U0001F64F]/u'
        emoticon_string = r"""
            (?:
              [<>]?
              [:;=8]                     # eyes
              [\-o\*\']?                 # optional nose
              [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
              |
              [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
              [\-o\*\']?                 # optional nose
              [:;=8]                     # eyes
              [<>]?
            )"""
        emoticon_re = re.compile(emoji_pattern + "|" + emoticon_string, re.VERBOSE | re.I | re.UNICODE)

        has_emoticons = DataFrame(map(lambda x: emoticon_re.search(x) is not None, X), columns=['has_url'])
        return has_emoticons

    def fit(self, X, y=None, **fit_params):
        return self

    def get_feature_names(self):
        return ['has_emoticons']