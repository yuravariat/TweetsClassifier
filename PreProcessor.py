#from stop_words import get_stop_words
import re

class PreProccessor:
    remove_stop_words = True
    urls_replace = True
    numbers_replace = True
    user_references_replace = True
    stemming = True
    lemmatization = True

    def __init__(self, options=None):
        if options is not None and type(options) is dict:
            if options.has_key("remove_stop_words"):
                self.remove_stop_words = options["remove_stop_words"]
            if options.has_key("urls_replace"):
                self.replace_urls = options["urls_replace"]
            if options.has_key("numbers_replace"):
                self.numbers_replace = options["numbers_replace"]
            if options.has_key("user_references_replace"):
                self.user_references_replace = options["user_references_replace"]
            if options.has_key("stemming"):
                self.stemming = options["stemming"]
            if options.has_key("lemmatization"):
                self.lemmatization = options["lemmatization"]

    def __urls_replace(self,t):
        t = re.sub(r"(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)", "[URL]", t)
        return t

    def __user_references_replace(self, t):
        t = re.sub(r"([\s]@[^\s]+)", "[USER_REF]", t)
        return t

    def __numbers_replace(self, t):
        t = re.sub(r"([\s][0-9]+[\s])", " [NUM] ", t)
        return t

    def Perform(self, listOfStrings):
        for indx, text in enumerate(listOfStrings):
            if self.urls_replace :
                listOfStrings[indx] = self.__urls_replace(listOfStrings[indx])
            if self.user_references_replace:
                listOfStrings[indx] = self.__user_references_replace(listOfStrings[indx])
            if self.numbers_replace:
                listOfStrings[indx] = self.__numbers_replace(listOfStrings[indx])
        return listOfStrings

#opt = {}
#opt["remove_stop_words"] = False
#test = PreProccessor()
#newlist = test.Perform(['kjhkj @yura h5kjh 5323 kj h https://t.co/1KrcmPzBkl jhj h j '])
#ttt = 1
