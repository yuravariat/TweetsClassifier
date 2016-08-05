import nltk
from nltk.tag.perceptron import PerceptronTagger

# nltk.download()

sentence = """At eight o'clock on Thursday morning... Arthur didn't feel very good."""
tokens = nltk.word_tokenize(sentence)

tagger = PerceptronTagger(False)
tagger.load('file:///C:/Users/yarov/nltk_data/taggers/averaged_perceptron_tagger/averaged_perceptron_tagger.pickle')
tagged = tagger.tag(tokens)
print tagged
