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

from enum import Enum

from Transformers import HasEmoticonsTransformer
from Transformers.HasUrlTransformer import HasUrlTransformer
from Transformers.TextLengthTransformer import TextLengthTransformer
from Transformers.PosTransformer import PosTransformer

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

    __enable_text_length_transformer = True
    __enable_url_transformer = True
    __enable_pos_transformer = False
    __enable_ngrams_transformer = True
    __enable_emoticons_transformer = True

    '''
    annotated_data: tweets with related category.
    classifier_type: which classifier to use.
    '''
    def buildClassifier(self, annotated_data=None, classifier_type=None):

        if classifier_type is None:
            self.classifier_type = ClassifierType.MultinomialNB
        else:
            self.classifier_type = classifier_type

        # Annotated data
        if annotated_data is None:
            # insert some test annotated data
            #categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
            categories = ['soc.religion.christian', 'comp.graphics']
            self.annotated_data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
        else:
            self.annotated_data = annotated_data

        # Postprocessing (urls, numbers and user references replacement)
        preproccessor = PreProccessor()
        preproccessor.Perform(self.annotated_data.data)

        count_vect = None
        tfidf_transformer = None
        text_length_transformer = None
        has_url_transformer = None
        pos_transformer = None
        emoticons_transformer = None

        transformers_list = []
        if self.__enable_text_length_transformer:
            text_length_transformer = TextLengthTransformer()
            transformers_list.append(('text_length', text_length_transformer))
        if self.__enable_url_transformer:
            has_url_transformer = HasUrlTransformer()
            transformers_list.append(('has_url', has_url_transformer))
        if self.__enable_emoticons_transformer:
            emoticons_transformer = HasEmoticonsTransformer
            transformers_list.append(('has_emoticons', emoticons_transformer))
        if self.__enable_pos_transformer:
            pos_transformer = PosTransformer()
            transformers_list.append(('part_of_speech', pos_transformer))
        if self.__enable_ngrams_transformer:
            # Tokenizer unigram and bigram tokens (ngram_range=(1, 2)). Stop words removed (stop_words='english')
            count_vect = CountVectorizer(ngram_range=(1, 2), stop_words='english')
            tfidf_transformer = TfidfTransformer()
            transformers_list.append(('ngram_tf_idf', Pipeline([
                    ('counts', count_vect),
                    # ('tf_idf', tfidf_transformer)
                ]))
            )

        features = FeatureUnion(
            transformer_list=transformers_list,
            # weight components in FeatureUnion
            #transformer_weights={
            #    'text_length': 1.0,
            #    'part_of_speech': 1.0
            #}
        )

        ''' =============== Features to add =========================
        - Hashtags
        - Topics (extracted with LDA)
        - POS ישנן שמתאימים למדיה חברתית
        - Emotions  presence(yes or no)
        - s סימני פיסוק, אותיות גדולות קטנות
        - Username
        - Timestamp(break into 3 groups: morning, noon, evening)
        '''

        ################ Test area, check features count and names. ###################
        #try:
        #    checkFeautureTable = features.fit(self.annotated_data.data)
        #    features_names = []
        #    if self.__enable_text_length_transformer:
        #        features_names.append(text_length_transformer.get_feature_names())
        #    if self.__enable_url_transformer:
        #        features_names.append(has_url_transformer.get_feature_names())
        #    if self.__enable_emoticons_transformer:
        #        features_names.append(emoticons_transformer.get_feature_names())
        #    if self.__enable_pos_transformer:
        #        features_names.append(pos_transformer.get_feature_names())
        #    if self.__enable_ngrams_transformer:
        #        features_names.append(count_vect.get_feature_names())
        #    numpyTable = checkFeautureTable.transform(self.annotated_data.data)
        #except Exception as inst:
        #    print("OS error: {0}".format(inst))
        #################################################################################

        pipeline = Pipeline([
          ('features', features),
          ('classifier', MultinomialNB())
        ])

        # Actually builds the classifier
        classifier = pipeline.fit(self.annotated_data.data, self.annotated_data.target)
        classifier_cont = Classifier()
        classifier_cont.classifier = classifier
        classifier_cont.labels = self.annotated_data.target_names
        return classifier_cont
'''
Classifier container
'''
class Classifier:
    classifier = None
    labels = None

# Build classifier test
classifierBuilder = ClassifierFactory()
clf = classifierBuilder.buildClassifier()

# Prediction
docs_new = ['God is love', 'OpenGL on the GPU is fast']
predicted = clf.classifier.predict(docs_new)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, clf.labels[category]))

rrr=1