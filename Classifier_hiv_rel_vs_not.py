import codecs
import os

from sklearn import cross_validation
from sklearn.datasets import get_data_home

from ClassifierFactory import ClassifierFactory, ClassifierSettings, ClassifierType
from classifier.data import DataAdapter, Tweet

disease = 'hiv'
predict_mode = False
categories = ['relevant', 'not_relevant']
cl_cut = 'rel_vs_not'
dataAdapter = DataAdapter(disease,cl_cut, 'relevancy', textOnlyTweets=True)

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
classifierSettings.enable_text_length_transformer = False
classifierSettings.enable_url_transformer = False
classifierSettings.enable_pos_transformer = False
classifierSettings.enable_ngrams_transformer = True
classifierSettings.enable_emoticons_transformer = False
classifierSettings.enable_username_transformer = False
classifierSettings.enable_part_of_day_transformer = False
classifierSettings.enable_punctuation_transformer = False
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

    for i in range(len(testData)):
        testData[i].relevant = str(clf.labels[predicted[i]])

    file_dir = os.path.join(get_data_home(), 'output', disease, cl_cut)

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    file_path = os.path.join(file_dir,'output.txt')

    with codecs.open(file_path, "w", "utf-8") as text_file:
        for i in range(len(testData)):
            try:
                tweet = "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\t{12}\t{13}\t{14}\t{15}\n". \
                    format(testData[i].tweet_id,
                           testData[i].query,
                           testData[i].disease,
                           testData[i].created_at,
                           testData[i].screen_name,
                           testData[i].text,
                           testData[i].description,
                           testData[i].location,
                           testData[i].timezone,
                           testData[i].user_id,
                           testData[i].coordinate,
                           testData[i].tweets_per_user,
                           testData[i].posted_by,
                           testData[i].talk_about,
                           testData[i].sarcasm,
                           testData[i].relevant)
                text_file.write(tweet.encode("utf-8"))
            except Exception as e:
                print("Exception!!! tweet " + str(i)  + " e=" + e.message)

    print("done")

print('done!')