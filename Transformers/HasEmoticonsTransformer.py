from pandas import DataFrame
from sklearn.base import TransformerMixin
import re


class HasEmoticonsTransformer(TransformerMixin):

    _emoji_pattern = r'/[U0001F601-U0001F64F]/u'
    _emoticon_string = r"""
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

    def __init__(self):
        self._emoticon_re = re.compile(self._emoji_pattern + "|" + self._emoticon_string, re.VERBOSE | re.I | re.UNICODE)

    def transform(self, X, **transform_params):
        has_emoticons = DataFrame(map(lambda x: self._emoticon_re.search(x.text) is not None, X), columns=['has_emoticons'])
        return has_emoticons

    def fit(self, X, y=None, **fit_params):
        return self

    def get_feature_names(self):
        return ['has_emoticons']