# -*- coding: utf-8 -*-
import itertools
from pandas import DataFrame
from sklearn.base import TransformerMixin
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion

from PreProcessor import PreProccessor
from Transformers import TextLengthTransformer

from enum import Enum

'''
Enum of Classifier types
'''
class ClassifierType(Enum):
    MultinomialNB = 1
    SVM = 2

'''
ClassifierFactory.
'''
class ClassifierFactory:
    annotated_data = None
    classifier_type = None
    '''
    annotated_data: tweets with related category.
    classifier_type: which classifier to use.
    '''
    def buildClassifier(self, annotated_data=None, classifier_type=None):

        if classifier_type is None:
            self.classifier_type = ClassifierType.MultinomialNB
        else:
            self.classifier_type = classifier_type

        if annotated_data is None:
            # insert some test annotated data
            categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
            self.annotated_data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
        else:
            self.annotated_data = annotated_data

        # Postprocessing (urls, numbers and user references replacement)
        preproccessor = PreProccessor()
        preproccessor.Perform(self.annotated_data.data)

        # Tokenizer unigram and bigram tokens (ngram_range=(1, 2)). Stop words removed (stop_words='english')
        count_vect = CountVectorizer(ngram_range=(1, 2),stop_words='english')
        tfidf_transformer = TfidfTransformer()

        # text length transformer
        textLengthTransformer = TextLengthTransformer()

        features = FeatureUnion([
            ('text_length', textLengthTransformer),
            ('ngram_tf_idf', Pipeline([
              ('counts', count_vect),
              ('tf_idf', tfidf_transformer)
            ]))
        ])

        ''' =============== Features to add =========================
        - Hashtags
        - Topics (extracted with LDA)
        - POS ישנן שמתאימים למדיה חברתית
        - Emotions  presence(yes or no)
        - URLs presence(yes or no)
        - s סימני פיסוק, אותיות גדולות קטנות
        - Username
        - Timestamp(break into 3 groups: morning, noon, evening)
        '''

        ################ Test area, check features count and names. ###################
        try:
            checkFeautureTable = features.fit(self.annotated_data.data)
            featursNames = textLengthTransformer.get_feature_names() + count_vect.get_feature_names()
        except Exception as inst:
            print("OS error: {0}".format(inst))
        ################################################################################3

        pipeline = Pipeline([
          ('features', features),
          ('classifier', MultinomialNB())
        ])

        # Actually builds the classifier
        classifier = pipeline.fit(self.annotated_data.data, self.annotated_data.target)
        return classifier

# test
classifierBuilder = ClassifierFactory()
classifier = classifierBuilder.buildClassifier()
rrr=1