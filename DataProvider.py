import os
import logging
import pickle
import shutil
import codecs
import numpy as np

from sklearn.utils import validation
from sklearn.datasets import get_data_home
from sklearn.datasets import base
from sklearn.datasets.base import _pkl_filepath

from Tweet import Tweet


class AsthmaTweetsGenerator:

    def __init__(self):
        pass

    def generate(self, data_home=None, ):
        data_home = get_data_home(data_home=data_home)
        asthma_path = os.path.join(data_home, 'asthma')
        tweets_path = os.path.join(data_home, 'tweets')
        if not os.path.exists(tweets_path):
            return
        '''
        *** Manual process:
        Save annotation files as 'Text (MS-DOS)(*.txt)', e.g. Annotations_Alona.txt (all annotation files should keep the same format)

        *** Automated process:
        1. Get file names from the C:\Users\[logged in user]\scikit_learn_data\tweets
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
        next(tweets_iterator)
        for tweet in tweets_iterator:
            segments = str(tweet).split('\t')

            id = segments[0]
            posted_by = segments[1]
            talk_about = segments[2]
            text = segments[3]
            description = segments[4]
            time_zone = segments[5]
            user_id = segments[6]
            coordinates = segments[7]
            tweets_per_user = segments[8]
            created_at = segments[9]
            screen_name = segments[11]

            if posted_by == '':
                continue

            train_path = os.path.join(asthma_path, 'train')
            if not os.path.exists(train_path):
                os.makedirs(train_path)

            category_path = os.path.join(train_path, posted_by)
            if not os.path.exists(category_path):
                os.makedirs(category_path)

            file_path = os.path.join(category_path, id+'.txt')
            with open(file_path, "w") as text_file:
                text_file.write(tweet)









class DataLoader:

    logger = logging.getLogger(__name__)
    cache_name = "cache.pkz"
    train_folder = "train"
    test_folder = 'test'

    def __init__(self):
        pass

    def get_annotated_data(self,
                           data_home=None,
                           subset='train',
                           categories=None,
                           shuffle=True,
                           random_state=42,
                           remove=()):
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

        data_home = get_data_home(data_home=data_home)
        cache_path = _pkl_filepath(data_home, self.cache_name)
        target_path = os.path.join(data_home, 'asthma')
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

        if 'headers' in remove:
            data.data = [self.parse1(tweet.text) for tweet in data.data]
        if 'footers' in remove:
            data.data = [self.parse2(tweet.text) for tweet in data.data]
        if 'quotes' in remove:
            data.data = [self.parse3(tweet.text) for tweet in data.data]

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

    def parse1(self, text):
        pass

    def parse2(self, text):
        pass

    def parse3(self, text):
        pass



