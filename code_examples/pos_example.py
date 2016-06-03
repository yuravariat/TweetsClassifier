import nltk

#test pos
#nltk.download()

words = ['is', 'table','apple','are','tree','run','running','sit']
pos_window = nltk.pos_tag(words)

tag_fd = nltk.FreqDist(tag for (word, tag) in pos_window)
frequencies = tag_fd.most_common()

jhgjhj = nltk.help.upenn_tagset()
tttt = nltk.help.upenn_tagset('VB')
from nltk.data import load
tagdict = load('help/tagsets/upenn_tagset.pickle')
kjhkj = tagdict.keys()

test = 'stop'
