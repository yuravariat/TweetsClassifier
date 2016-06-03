from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import re

stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')
en_stop = get_stop_words('en')


def get_url(t):
    url_expression = re.search("(?P<url>https?://[^\s]+)", t)
    if url_expression is None:
        return ''

    url = url_expression.group("url")
    return url

for tweet in open('F:\\Development\\Data\\tweets\\A-AIDS1504.txt'):
    tweet_segments = tweet.split('\t')
    timestamp = tweet_segments[0]
    user_id = tweet_segments[1]
    tweet_id = tweet_segments[2]
    tweet_text = tweet_segments[3]
    unknown1 = tweet_segments[4]
    unknown2 = tweet_segments[5]
    timezone = tweet_segments[6]
    unknown3 = tweet_segments[7]
    unknown4 = tweet_segments[8]
    unknown5 = tweet_segments[9]

    print tweet_text

    text = tweet_text.lower()

    text_segments = text.split(' ')
    for segment in text_segments:
        if segment.startswith('@'):
            text_segments.remove(segment)

    text_segments_new = []
    for segment in text_segments:
        text_segments_new.append(segment)
        text_segments_new.append(' ')

    text = ''.join(text_segments_new)

    urls = []
    u = get_url(text)
    while u != '':
        urls.append(u)
        text = text.replace(u, '')
        u = get_url(text)

    tokens = tokenizer.tokenize(text)
    stopped_tokens = [i for i in tokens if not i in en_stop]
    stemmed_tokens = [stemmer.stem(i) for i in stopped_tokens]

    final_tokens = []
    for st in stemmed_tokens:
        if st not in final_tokens:
            final_tokens.append(st)

    print final_tokens
    print ''

