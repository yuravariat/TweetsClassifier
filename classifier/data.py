import os
import logging
import pickle
import shutil
import codecs
import numpy as np

from sklearn.utils import validation
from sklearn.datasets import get_data_home
from sklearn.datasets import base


class Tweet:
    tweet_id = ''
    query = ''
    disease = ''
    created_at = ''
    screen_name = ''
    text = ''
    description = ''
    location = ''
    timezone = ''
    user_id = ''
    coordinate = ''
    tweets_per_user = ''
    posted_by = ''
    talk_about = ''
    sarcasm = ''

    def __init__(self, tweet_text=None):
        if tweet_text is not None:
            segments = str(tweet_text).split('\t')

            self.tweet_id = segments[0]
            self.query = segments[1]
            self.disease = segments[2]
            self.created_at = segments[3]
            self.screen_name = segments[4]
            self.text = segments[5]
            self.description = segments[6]
            self.location = segments[7]
            self.timezone = segments[8]
            self.user_id = segments[9]
            self.coordinate = segments[10]
            self.tweets_per_user = segments[11]
            self.posted_by = segments[12]
            self.talk_about = segments[13]
            self.sarcasm = segments[14]



class DataAdapter:

    logger = logging.getLogger(__name__)
    cache_name = "cache.pkz"
    train_folder = "train"
    test_folder = 'test'
    disease = None

    def __init__(self, disease):
        self.disease = disease

    def create_data(self, disease):
        data_home = get_data_home()
        # e.g. C:\Users\[user]\scikit_learn_data\hiv
        disease_path = os.path.join(data_home, disease)
        # e.g. C:\Users\[user]\scikit_learn_data\tweets\hiv
        tweets_path = os.path.join(data_home, 'tweets', disease)
        if not os.path.exists(tweets_path):
            return
        '''
        *** Manual process:
        Save annotation files as 'Text (MS-DOS)(*.txt)', e.g. tweets1.txt (all annotation files should keep the same format)

        *** Automated process:
        1. Get file names from the C:\Users\[user]\scikit_learn_data\tweets\hiv
        2. For each file read all tweets line by line (only those where the category is not empty)
        3. For each tweet generate a unique file
        '''
        file_paths = []
        for root, directories, files in os.walk(tweets_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                file_paths.append(file_path)
        print (file_paths)

        tweets = []
        for file_path in file_paths:
            with open(file_path, 'r') as f:
                for line in f:
                    tweets.append(line)
            f.closed

        counter = 0
        tweets_iterator = iter(tweets)
        # skip the first line - column names
        next(tweets_iterator)
        for tweet in tweets_iterator:
            segments = str(tweet).split('\t')

            tweet_id = segments[0]
            query = segments[1]
            disease = segments[2]
            created_at = segments[3]
            screen_name = segments[4]
            text = segments[5]
            description = segments[6]
            location = segments[7]
            timezone = segments[8]
            user_id = segments[9]
            coordinate = segments[10]
            tweets_per_user = segments[11]
            posted_by = segments[12]
            talk_about = segments[13]
            sarcasm = segments[14]

            # classification category
            if talk_about == '':
                continue

            train_path = os.path.join(disease_path, self.train_folder)
            if not os.path.exists(train_path):
                os.makedirs(train_path)

            category_path = os.path.join(train_path, talk_about)
            if not os.path.exists(category_path):
                os.makedirs(category_path)

            file_path = os.path.join(category_path, str(counter) + '.txt')
            with open(file_path, "w") as text_file:
                text_file.write(tweet)
            counter += 1

    def get_data(self, subset='train', categories=None, shuffle=True, random_state=42):
        """Load the filenames and data from the 20 newsgroups dataset.

        Read more in the :ref:`User Guide <20newsgroups>`.

        Parameters
        ----------
        subset: 'train' or 'test', 'all', optional
            Select the dataset to load: 'train' for the training set, 'test'
            for the test set, 'all' for both, with shuffled ordering.

        data_home: optional, default: None
            Specify a download and cache folder for the datasets. If None,
            all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

        categories: None or collection of string or unicode
            If None (default), load all the categories.
            If not None, list of category names to load (other categories
            ignored).

        shuffle: bool, optional
            Whether or not to shuffle the data: might be important for models that
            make the assumption that the samples are independent and identically
            distributed (i.i.d.), such as stochastic gradient descent.

        random_state: numpy random number generator or seed integer
            Used to shuffle the dataset.

        remove: tuple
            May contain any subset of ('headers', 'footers', 'quotes'). Each of
            these are kinds of text that will be detected and removed from the
            newsgroup posts, preventing classifiers from overfitting on
            metadata.

            'headers' removes newsgroup headers, 'footers' removes blocks at the
            ends of posts that look like signatures, and 'quotes' removes lines
            that appear to be quoting another post.

            'headers' follows an exact standard; the other filters are not always
            correct.
        """

        data_home = get_data_home()
        cache_path = os.path.join(data_home, self.cache_name)
        target_path = os.path.join(data_home, self.disease)
        cache = None
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    compressed_content = f.read()
                uncompressed_content = codecs.decode(
                    compressed_content, 'zlib_codec')
                cache = pickle.loads(uncompressed_content)
            except Exception as e:
                print(80 * '_')
                print('Cache loading failed')
                print(80 * '_')
                print(e)

        if cache is None:
            cache = self.get_cache(target_path, cache_path)

        if subset in ('train', 'test'):
            data = cache[subset]
        elif subset == 'all':
            data_lst = list()
            target = list()
            filenames = list()
            for subset in ('train', 'test'):
                data = cache[subset]
                data_lst.extend(data.data)
                target.extend(data.target)
                filenames.extend(data.filenames)

            data.data = data_lst
            data.target = np.array(target)
            data.filenames = np.array(filenames)
        else:
            raise ValueError(
                "subset can only be 'train', 'test' or 'all', got '%s'" % subset)

        data.description = 'The HIV dataset'

        if categories is not None:
            labels = [(data.target_names.index(cat), cat) for cat in categories]
            # Sort the categories to have the ordering of the labels
            labels.sort()
            labels, categories = zip(*labels)
            mask = np.in1d(data.target, labels)
            data.filenames = data.filenames[mask]
            data.target = data.target[mask]
            # searchsorted to have continuous labels
            data.target = np.searchsorted(labels, data.target)
            data.target_names = list(categories)
            # Use an object array to shuffle: avoids memory copy
            data_lst = np.array(data.data, dtype=object)
            data_lst = data_lst[mask]
            data.data = data_lst.tolist()

        if shuffle:
            random_state = validation.check_random_state(random_state)
            indices = np.arange(data.target.shape[0])
            random_state.shuffle(indices)
            data.filenames = data.filenames[indices]
            data.target = data.target[indices]
            # Use an object array to shuffle: avoids memory copy
            data_lst = np.array(data.data, dtype=object)
            data_lst = data_lst[indices]
            data.data = data_lst.tolist()

        return data

    def get_cache(self, target_path, cache_path):
        train_path = os.path.join(target_path, self.train_folder)
        test_path = os.path.join(target_path, self.test_folder)

        if not os.path.exists(target_path):
            os.makedirs(target_path)

        if not os.path.exists(train_path):
            os.makedirs(train_path)

        if not os.path.exists(test_path):
            os.makedirs(test_path)

        cache = dict(train=base.load_files(train_path, encoding='utf-8'),
                     test=base.load_files(test_path, encoding='utf-8'))

        # turn tweet text to Tweet objects.
        tweetsList = list()
        for tweet in cache['train'].data:
            tweetsList.append(Tweet(tweet))
        cache['train'].data = tweetsList
        tweetsList = list()
        for tweet in cache['test'].data:
            tweetsList.append(Tweet(tweet))
        cache['test'].data = tweetsList

        compressed_content = codecs.encode(pickle.dumps(cache), 'zlib_codec')

        with open(cache_path, 'wb') as f:
            f.write(compressed_content)

        shutil.rmtree(target_path)

        return cache