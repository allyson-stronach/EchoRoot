import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn import cross_validation
from sklearn import metrics

from model import Ad, AdAttribute, session

import nltk
import re
import random
import pickle


def analyze_ad():
    dl_list = retrieve_trafficky_text()
    dl_list = retrieve_not_trafficky_text(dl_list)
    vectorizer = TfidfVectorizer(stop_words='english')
    Xy = vectorize_ads(dl_list, vectorizer)
    classifier = classify_ads(Xy)
    random_id = get_random_id()
    pickle_classifier(classifier)
    test_document = generate_test_data(random_id)
    test_sample(vectorizer, test_document)
    #test_all_samples(vectorizer)
    describe_features(vectorizer, classifier)


def retrieve_trafficky_text():
    documents = []
    labels = []
    ads_text_cmd = "SELECT ads.text AS text FROM ads_attributes JOIN ads ON ads.id = ads_id WHERE ads_attributes.value IN  ('7087629612', '9292103206', '4142395461', '4146870501', '7045060509', '5203663536')"
    trafficky_text = session.execute(ads_text_cmd)

    for text in trafficky_text:
        string_text = str(text)
        string_text = re.sub('(\\()', '', string_text)
        string_text = re.sub('(,\\))', '', string_text)
        string_text = re.sub("(\\')", "", string_text)
        lower_text = string_text.lower()
        no_HTML_text = re.sub('<\s*\w.*?>', '', lower_text)
        no_unicode_text = re.sub('([\\\\]x..)', '', no_HTML_text)
        no_newline_text = re.sub('([\\\\]n)', '', no_unicode_text)
        filtered_text = no_newline_text
        documents.append(filtered_text)
        labels.append('trafficky')
    query = "SELECT text from ads WHERE text LIKE '%cherry11%'"
    known_trafficky_test_text = session.execute(query)

    for text in known_trafficky_test_text:
        string_text = str(text)
        string_text = re.sub('(\\()', '', string_text)
        string_text = re.sub('(,\\))', '', string_text)
        string_text = re.sub("(\\')", "", string_text)
        lower_text = string_text.lower()
        no_HTML_text = re.sub('<\s*\w.*?>', '', lower_text)
        no_unicode_text = re.sub('([\\\\]x..)', '', no_HTML_text)
        no_newline_text = re.sub('([\\\\]n)', '', no_unicode_text)
        filtered_text = no_newline_text
        documents.append(filtered_text)
        labels.append('trafficky')
    
    print 'documents length:', len(documents), 'labels length:', len(labels)
    dl_list = [documents, labels]
    return dl_list


def retrieve_not_trafficky_text(dl_list):
    ads_text_cmd = "SELECT ads.text AS text FROM ads_attributes JOIN ads ON ads.id = ads_id WHERE ads_attributes.value IN ('7027565783', '4702535139', '9172794962', '6149001084', '7865195399', '4048401717', '3133388625', '5106213824', '3374231635', '2622609175', '6465433780', '4388078188')"
    not_trafficky_text = session.execute(ads_text_cmd)

    for text in not_trafficky_text:
        string_text = str(text)
        string_text = re.sub('(\\()', '', string_text)
        string_text = re.sub('(,\\))', '', string_text)
        string_text = re.sub("(\\')", "", string_text)
        lower_text = string_text.lower()
        no_HTML_text = re.sub('<\s*\w.*?>', '', lower_text)
        no_unicode_text = re.sub('([\\\\]x..)', '', no_HTML_text)
        no_newline_text = re.sub('([\\\\]n)', '', no_unicode_text)
        filtered_text = no_newline_text
        dl_list[0].append(filtered_text)
        dl_list[1].append('not trafficky')
    
    # print 'documents length:', len(dl_list[0]), 'labels length:', len(dl_list[1])
    # print 'fraction trafficky:', len([item for item in dl_list[1] if item == 'trafficky'])/len(dl_list[1])
    return dl_list


def vectorize_ads(dl_list, vectorizer):
    X = vectorizer.fit_transform(dl_list[0])
    y = np.array(dl_list[1])
    Xy = [X, y]

    return Xy


def classify_ads(Xy):
    classifier = BernoulliNB()
    cv = cross_validation.StratifiedKFold(Xy[1],2)
    precision=[]
    recall=[]

    for train, test in cv:
        X_train = Xy[0][train]
        X_test = Xy[0][test]
        y_train = Xy[1][train]
        y_test = Xy[1][test]
        classifier.fit(X_train, y_train)
        y_hat = classifier.predict(X_test)
        p,r,_,_ = metrics.precision_recall_fscore_support(y_test, y_hat)
        precision.append(p[1])
        recall.append(r[1])

    print classifier
    print 'precision:',np.average(precision), '+/-', np.std(precision)
    print 'recall:', np.average(recall), '+/-', np.std(recall)
    return classifier


def get_random_id():
    id_list = []
    f = open('id_list.pickle', 'wb')
    pickle.dump(id_list, f)
    f.close()
    ads_id_cmd = "SELECT id FROM ads"
    ids = session.execute(ads_id_cmd)

    for i in ids:
        i = str(i)
        i = re.sub('(\\()', '', i)
        i = re.sub('(L,\\))', '', i)
        id_list.append(i)
    random_id = (random.choice(id_list))

    return random_id


def generate_test_data(random_id):
    test_document = []
    query = session.query(Ad.text).filter(Ad.id == random_id)
    test_text = session.execute(query)

    for text in test_text:
        string_text = str(text)
        string_text = re.sub('(\\()', '', string_text)
        string_text = re.sub('(,\\))', '', string_text)
        string_text = re.sub("(\\')", "", string_text)
        lower_text = string_text.lower()
        no_HTML_text = re.sub('<\s*\w.*?>', '', lower_text)
        no_unicode_text = re.sub('([\\\\]x..)', '', no_HTML_text)
        no_newline_text = re.sub('([\\\\]n)', '', no_unicode_text)
        filtered_text = no_newline_text
        test_document.append(filtered_text)

    print test_document
    return test_document


def pickle_classifier(classifier):
    f = open('classifier.pickle', 'wb')
    pickle.dump(classifier, f)
    f.close()


def test_sample(vectorizer, test_document):
    f = open('classifier.pickle')
    classifier = pickle.load(f)
    sample = vectorizer.transform(test_document)
    classification_new_ad = classifier.predict(sample)
    ndarray = classifier.predict_proba(sample)
    f.close()

    probability_classification = str(ndarray[0])
    probability_classification = re.sub('^(\[ +).+\s', '', probability_classification)
    probability_classification = re.sub('(\])', '', probability_classification)
    probability_classification = probability_classification.replace(',', '.')
    probability_classification = float(probability_classification)
    probability_classification = round(probability_classification, 2)
    
    print "binary classification:", classification_new_ad
    print "probability classification:", probability_classification
    return ndarray

def test_all_samples(vectorizer):
    test_samples_list = []
    probability_classification_list = []
    histogram_data = {0.0:0, 0.1:0, 0.2:0, 0.3:0, 0.4:0, 0.5:0, 0.6:0, 0.7:0, 0.8:0, 0.9:0, 1.0:0}
    ads_text_cmd = "SELECT text FROM ads LIMIT 1000000"
    test_samples = session.execute(ads_text_cmd)
    print test_samples

    for text in test_samples:
        string_text = str(text)
        string_text = re.sub('(\\()', '', string_text)
        string_text = re.sub('(,\\))', '', string_text)
        string_text = re.sub("(\\')", "", string_text)
        lower_text = string_text.lower()
        no_HTML_text = re.sub('<\s*\w.*?>', '', lower_text)
        no_unicode_text = re.sub('([\\\\]x..)', '', no_HTML_text)
        no_newline_text = re.sub('([\\\\]n)', '', no_unicode_text)
        filtered_text = no_newline_text
        test_samples_list.append(filtered_text)

    f = open('classifier.pickle')
    classifier = pickle.load(f)
    samples = vectorizer.transform(test_samples_list)
    classification_new_ads = classifier.predict(samples)
    ndarray = classifier.predict_proba(samples)
    f.close()

    for item in ndarray:
        probability_classification = str(item)
        probability_classification = re.sub('^(\[ +).+\s', '', probability_classification)
        probability_classification = re.sub('(\])', '', probability_classification)
        probability_classification = probability_classification.replace(',', '.')
        probability_classification = float(probability_classification)
        probability_classification = round(probability_classification, 1)
        probability_classification_list.append(probability_classification)
    
    print "binary classification:", classification_new_ads
    print "probability classification list:", probability_classification_list

    for item in probability_classification_list:
        if histogram_data[item]:
            histogram_data[item] = histogram_data[item] + 1
        else:
            histogram_data[item] = 1
    
    print histogram_data


def describe_features(vectorizer, classifier):
    probs=classifier.feature_log_prob_[1]
    a = vectorizer.get_stop_words()

    features=vectorizer.get_feature_names()
    
    print 'length of probs:', len(probs)
    print 'length of features:', len(features)
    print 'list of most important features:', sorted(zip(probs,features), reverse=True)[:10]


def main():
    analyze_ad()

if __name__ == "__main__":
    main()



