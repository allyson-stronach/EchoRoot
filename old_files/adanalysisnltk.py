"""DO NOT USE THIS FILE"""

from model import session, AdAttribute, Ad
import nltk

"""making toy training data set"""

def retrieve_trafficky_text():
    trafficky_ad_list = []
    trafficky_ads_cmd = "SELECT ads.text AS text FROM ads_attributes JOIN ads ON ads.id = ads_id WHERE ads_attributes.value IN  ('9292103206', '4142395461', '4146870501') "
    e = session.execute(trafficky_ads_cmd)
    for text in e:
        trafficky_ad_list.append((text.text, 'trafficky'))
    print "running in retrieve trafficky text", type(trafficky_ad_list)
    return trafficky_ad_list

def retrieve_not_trafficky_text():
    not_trafficky_ad_list = []
    not_trafficky_ads_cmd = "SELECT ads.text AS text FROM ads_attributes JOIN ads ON ads.id = ads_id WHERE ads_attributes.value IN ('3104623985', '2139840845 ', '8183362736', '6032946322', '4088991922')"
    e = session.execute(not_trafficky_ads_cmd)
    for text in e:
        not_trafficky_ad_list.append((text.text, 'nottrafficky'))
    print "running in retrieve not trafficky text", type(not_trafficky_ad_list)
    return not_trafficky_ad_list

def make_training_ads(trafficky_ad_list, not_trafficky_ad_list):
    ads = []
    for (words, traffickyness) in trafficky_ad_list + not_trafficky_ad_list:
        words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
        ads.append((words_filtered, traffickyness))
        #maybe remove html? other stuffs?
    print "this is length of ads", len(ads), type(ads)
    return ads

"""classifier"""

def get_words_in_ads(ads):
    all_words = []
    for (words, traffickyness) in ads:
        all_words.extend(words)
    print "this is the length of all words", len(all_words), type(all_words)
    return all_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    print "this is the length of word features", len(word_features), type(word_features)
    return word_features

def extract_features(document, word_features):
    #what is document? we think it's a tuple and in this case, document is ad--do i want it to be ad? hmm.. 
    document_words = set(document)
    print type(document_words)
    #set only contains types
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    #when i print features, all booleans are true
    print "this is the length of extract features", len(features), type(features)
    return features

#understand?
def make_training_set(features, ads):
    training_set = nltk.classify.apply_features(features, ads)
    #type(training_set) returns <class 'nltk.util.LazyMap'> more on this in the NLTk book, ch.6
    print "this is the length of training set", len(training_set), type(training_set)
    return training_set

#this function still not working
#still need to find the SQL function that will actually make this work
def make_classifier(training_set):
    print "before classifier line"
    #ok so the training_set in the following line is definitely the problem; when i changed it to say training-set() (because it's some kind of class), the error message was still a Type Error, but instead of 'dict' object, it was then a 'LazyMap' object--what other way to make a classifier in NLTK is there? 
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    print "after classifier line"
    return classifier


def main():
    tal = retrieve_trafficky_text()
    ntal = retrieve_not_trafficky_text()
    ads = make_training_ads(tal, ntal)
    all_words = get_words_in_ads(ads)
    word_features = get_word_features(all_words)
    features = extract_features(all_words, word_features)
    training_set = make_training_set(features, ads)
    classifier = make_classifier(training_set)

if __name__ == "__main__":
    main()

