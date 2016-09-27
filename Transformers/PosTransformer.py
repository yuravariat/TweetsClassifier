import os
import nltk
from pandas import DataFrame, np
from sklearn.base import TransformerMixin
from nltk.tag.perceptron import PerceptronTagger


class PerceptronTaggerError(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class PosTransformer(TransformerMixin):

    _tagger = PerceptronTagger(False)
    _nltk_paths = ['C:/Users/yurav/nltk_data', 'C:/Anaconda2/nltk_data']

    def __init__(self):

        loaded = False
        for nltk_path in self._nltk_paths:
            if os.path.exists(nltk_path):
                full_nltk_path = os.path.join(nltk_path, 'taggers/averaged_perceptron_tagger/averaged_perceptron_tagger.pickle')
                self._tagger.load('file:///{0}'.format(full_nltk_path))
                loaded = True
                break

        if not loaded:
            raise PerceptronTaggerError('The perceptron tagger was not found!')

    def transform(self, X, **transform_params):

        numpy_table = np.zeros((len(X), len(self.allTags)))

        for row_num, tweet in enumerate(X):
            tokens = nltk.wordpunct_tokenize(tweet.text)
            word_tags = self._tagger.tag(tokens)
            tag_fd = nltk.FreqDist(tag for (word, tag) in word_tags)
            for tag in tag_fd:
                if tag in self.allTags:
                    indexOfTag = self.allTags.index(tag)
                    if indexOfTag > -1:
                        # Proportion, how much part of speech appears in the text.
                        numpy_table[row_num, indexOfTag] = (tag_fd[tag]/(len(tokens)*1.0))

        data_table = DataFrame(numpy_table, columns=self.allTags)

        return data_table

    def fit(self, X, y=None, **fit_params):
        return self

    def get_feature_names(self):
        return self.allTags

    allTags = [
        'CC',  # Coordinating conjunction
        'CD',  # Cardinal number
        'DT',  # Determiner
        'EX',  # Existential there
        'FW',  # Foreign word
        'IN',  # Preposition or subordinating conjunction
        'JJ',  # Adjective
        'JJR',  # Adjective, comparative
        'JJS',  # Adjective, superlative
        'LS',  # List item marker
        'MD',  # Modal
        'NN',  # Noun, singular or mass
        'NNS',  # Noun, plural
        'NNP',  # Proper noun, singular
        'NNPS',  # Proper noun, plural
        'PDT',  # Predeterminer
        'POS',  # Possessive ending
        'PRP',  # Personal pronoun
        'PRP$',  # Possessive pronoun
        'RB',  # Adverb
        'RBR',  # Adverb, comparative
        'RBS',  # Adverb, superlative
        'RP',  # Particle
        'SYM',  # Symbol
        'TO',  # to
        'UH',  # Interjection
        'VB',  # Verb, base form
        'VBD',  # Verb, past tense
        'VBG',  # Verb, gerund or present participle
        'VBN',  # Verb, past participle
        'VBP',  # Verb, non-3rd person singular present
        'VBZ',  # Verb, 3rd person singular present
        'WDT',  # Wh-determiner
        'WP',  # Wh-pronoun
        'WP$',  # Possessive wh-pronoun
        'WRB',  # Wh-adverb
    ]

    # posTransformer = POSTransformer()
    # frame = DataFrame(np.zeros((2252, len(posTransformer.allTags))), columns=posTransformer.allTags)

    # stop = 5
