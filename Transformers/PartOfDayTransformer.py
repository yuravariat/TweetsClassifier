from pandas import DataFrame
from sklearn.base import TransformerMixin
from dateutil import parser


class PartOfDayTransformer(TransformerMixin):

    def transform(self, X, **transform_params):
        table = DataFrame(map(lambda x: PartOfDayTransformer.get_part_of_day(x.created_at), X), columns=['part_of_day'])
        return table

    def fit(self, X, y=None, **fit_params):
        return self

    def get_feature_names(self):
        return ['part_of_day']

    @staticmethod
    def get_part_of_day(datestr):
        if datestr is not None and bool(datestr.strip()):
            try:
                dt = parser.parse(datestr)
                if 5 < dt.hour < 12: # Morning
                    return 1
                if 12 < dt.hour < 16: # Midday
                    return 2
                if 16 < dt.hour < 22: # Evening
                    return 3
                if 22 < dt.hour < 5: # Night
                    return 4
            except Exception as inst:
                ttt =1
                #print("OS error: {0}".format(inst))
        return 0;