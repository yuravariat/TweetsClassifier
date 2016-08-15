from sklearn import cross_validation

from ClassifierFactory import ClassifierFactory, ClassifierSettings, ClassifierType
from classifier.data import DataAdapter

predict_mode = True

disease = 'hiv'
categories = ['organization', 'individual']
dataAdapter = DataAdapter(disease,'ind_vs_org','posted_by')

# 1. Generate training set by splitting the input files multiple files (file per tweet)
dataAdapter.create_data()

# 2. Load train data from files or cache
trainData = dataAdapter.get_data(categories=categories, subset='train')

# 3. Train classifier
classifierBuilder = ClassifierFactory()
classifierSettings = ClassifierSettings()
classifierSettings.train_data = trainData
classifierSettings.classifier_type = ClassifierType.LogisticRegression
classifierSettings.disease = disease
classifierSettings.enable_text_length_transformer = True
classifierSettings.enable_url_transformer = True
classifierSettings.enable_pos_transformer = True
classifierSettings.enable_ngrams_transformer = True
classifierSettings.enable_emoticons_transformer = True
classifierSettings.enable_username_transformer = True
classifierSettings.enable_part_of_day_transformer = True
classifierSettings.enable_punctuation_transformer = True
clf = classifierBuilder.buildClassifier(classifierSettings)

if not predict_mode:
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

else:
    # make prediction
    testData = dataAdapter.get_unclassified_data()
    predicted = clf.classifier.predict(testData)
    print ('predict done')

print('done!')