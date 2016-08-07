# -*- coding: utf-8 -*-
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from enum import Enum
from sklearn import tree, cross_validation
from sklearn.ensemble import AdaBoostClassifier

# our project packages
#from sympy.physics.quantum.circuitplot import np

from classifier.data import DataAdapter
from classifier.pre_processor import PreProccessor
from classifier.pre_processor import GetTextFromTweet
from Transformers.HasEmoticonsTransformer import HasEmoticonsTransformer
from Transformers.HasUrlTransformer import HasUrlTransformer
from Transformers.PartOfDayTransformer import PartOfDayTransformer
from Transformers.TextLengthTransformer import TextLengthTransformer
from Transformers.PosTransformer import PosTransformer
from Transformers.UsernameTransformer import UsernameTransformer
from Transformers.PunctuationTransformer import PunctuationTransformer


class ClassifierType(Enum):
    MultinomialNB = 1
    SVM = 2
    DecisionTree = 3
    RandomForest = 4
    LogisticRegression = 5
    AdaBoost = 6


class Classifier:

    def __init__(self):
        pass

    classifier = None
    labels = None


class ClassifierFactory:

    def __init__(self):
        pass

    annotated_data = None
    classifier_type = None

    __enable_text_length_transformer = True
    __enable_url_transformer = True
    __enable_pos_transformer = True
    __enable_ngrams_transformer = True
    __enable_emoticons_transformer = True
    __enable_username_transformer = True
    __enable_part_of_day_transformer = True
    __enable_punctuation_transformer = True

    def get_transformers(self):
        transformers_list = []
        if self.__enable_text_length_transformer:
            text_length_transformer = TextLengthTransformer()
            transformers_list.append(('text_length', text_length_transformer))
        if self.__enable_url_transformer:
            has_url_transformer = HasUrlTransformer()
            transformers_list.append(('has_url', has_url_transformer))
        if self.__enable_emoticons_transformer:
            emoticons_transformer = HasEmoticonsTransformer()
            transformers_list.append(('has_emoticons', emoticons_transformer))
        if self.__enable_username_transformer:
            username_transformer = UsernameTransformer()
            transformers_list.append(('user_name', username_transformer))
        if self.__enable_part_of_day_transformer:
            part_of_day_transformer = PartOfDayTransformer()
            transformers_list.append(('part_of_day', part_of_day_transformer))
        if self.__enable_pos_transformer:
            pos_transformer = PosTransformer()
            transformers_list.append(('part_of_speech', pos_transformer))
        if self.__enable_punctuation_transformer:
            punctuation_transformer = PunctuationTransformer()
            transformers_list.append(('punctuations', punctuation_transformer))
        if self.__enable_ngrams_transformer:
            # Tokenizer unigram and bigram tokens (ngram_range=(1, 2)). Stop words removed (stop_words='english')
            count_vect = CountVectorizer(ngram_range=(1, 3), stop_words='english', preprocessor=GetTextFromTweet)
            #tfidf_transformer = TfidfTransformer()
            transformers_list.append(('ngram_tf_idf', Pipeline([
                    ('counts', count_vect),
                    #('tf_idf', tfidf_transformer)
                ]))
            )
        print 'feature: ' + str([x[0] for x in transformers_list])
        return transformers_list

    def get_classifier(self):
        __classifier = None
        if self.classifier_type is ClassifierType.MultinomialNB:
            __classifier = MultinomialNB()
        if self.classifier_type is ClassifierType.SVM:
            __classifier = SVC()
        if self.classifier_type is ClassifierType.DecisionTree:
            __classifier = tree.DecisionTreeClassifier()
        if self.classifier_type is ClassifierType.RandomForest:
            __classifier = RandomForestClassifier()
        if self.classifier_type is ClassifierType.LogisticRegression:
            __classifier = LogisticRegression()
        if self.classifier_type is ClassifierType.AdaBoost:
            __classifier = AdaBoostClassifier()
        return __classifier

    def buildClassifier(self, train_data, classifier_type=None, categories=None, disease=None):

        print('start training classifier')

        if classifier_type is None and self.classifier_type is None:
            self.classifier_type = ClassifierType.MultinomialNB

        if classifier_type is not None and self.classifier_type is None:
            self.classifier_type = classifier_type

        # Annotated data
        self.annotated_data = train_data
        print('train contains ' + str(len(train_data.data)) + ' tweets and ' + str(len(train_data.target_names)) +
              ' categories: ' + str(train_data.target_names) )

        # Postprocessing (urls, numbers and user references replacement)
        preproccessor = PreProccessor()
        preproccessor.perform(self.annotated_data.data)

        print('pre-proccess done')

        count_vect = None
        tfidf_transformer = None
        text_length_transformer = None
        has_url_transformer = None
        pos_transformer = None
        emoticons_transformer = None
        username_transformer = None
        part_of_day_transformer = None
        punctuation_transformer = None

        print('classifier algorithm: ' + str(self.classifier_type) )

        transformers_list = self.get_transformers()

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
        - POS - We need an appropriate library for Tweeter
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

        __classifier = self.get_classifier()

        pipeline = Pipeline([
          ('features', features),
          ('classifier', __classifier)
        ])

        # Actually builds the classifier
        classifier = pipeline.fit(self.annotated_data.data, self.annotated_data.target)
        print 'classifier ready to use'

        classifier_obj = Classifier()
        classifier_obj.classifier = classifier
        classifier_obj.labels = self.annotated_data.target_names

        return classifier_obj

# ------------------------------------- classification area ---------------------------------

disease = 'hiv'
#categories = ['celeb', 'dont_know', 'family', 'himself', 'knows', 'none', 'subject']
categories = ['organization', 'individual']
dataAdapter = DataAdapter(disease, 'posted_by')

# 1. Generate training set by splitting the input files multiple files (file per tweet)
dataAdapter.create_data(disease)

# 2. Load train data from files or cache
trainData = dataAdapter.get_data(categories=categories, subset='train')

# 3. Train classifier
classifierBuilder = ClassifierFactory()
clf = classifierBuilder.buildClassifier(classifier_type=ClassifierType.LogisticRegression, disease=disease,train_data=trainData)


# Evaluation with cross validation test
print 'performing cross validation c=5 on train data'
scores = cross_validation.cross_val_score(clf.classifier,
                                          trainData.data,
                                          trainData.target,
                                          cv=5,
                                          scoring='precision_weighted')
scores_mean = scores.mean()
print 'cross validation done'
print 'scores: ' + str(scores)
print 'scores_mean: ' + str(scores_mean)

# Evaluation with splited data
#testData = dataAdapter.get_data(categories=categories, subset='train')
#predicted = clf.classifier.predict(testData)
#precision = precision_score(testData.target, predicted, average='weighted')
#recall = recall_score(testData.target, predicted, average='weighted')

print('done!')