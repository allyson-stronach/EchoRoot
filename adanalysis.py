import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn import cross_validation
from sklearn import metrics

from model import session

import nltk

import re


def analyze_ad():
    dl_list = retrieve_trafficky_text()
    dl_list = retrieve_not_trafficky_text(dl_list)
    vectorizer = TfidfVectorizer()
    Xy = vectorize_ads(dl_list, vectorizer)
    classifier = classify_ads(Xy)
    test_sample(vectorizer, classifier)
    describe_features(vectorizer, classifier)


def retrieve_trafficky_text():
    documents = []
    labels = []
    ads_text_cmd = "SELECT ads.text AS text FROM ads_attributes JOIN ads ON ads.id = ads_id WHERE ads_attributes.value IN  ('7087629612', '9292103206', '4142395461', '4146870501', '7045060509') "
    trafficky_text = session.execute(ads_text_cmd)
    for text in trafficky_text:
        string_text = str(text)
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
        lower_text = string_text.lower()
        no_HTML_text = re.sub('<\s*\w.*?>', '', lower_text)
        no_unicode_text = re.sub('([\\\\]x..)', '', no_HTML_text)
        no_newline_text = re.sub('([\\\\]n)', '', no_unicode_text)
        filtered_text = no_newline_text
        dl_list[0].append(filtered_text)
        dl_list[1].append('not trafficky')
    #print 'documents length:', len(dl_list[0]), 'labels length:', len(dl_list[1])
    #print 'fraction trafficky:', len([item for item in dl_list[1] if item == 'trafficky'])/240.0

    return dl_list


def vectorize_ads(dl_list, vectorizer):
    X = vectorizer.fit_transform(dl_list[0])
    y = np.array(dl_list[1])
    #print 'X:', X, 'X.shape is:', X.shape, 'y:', y, 'y.shape is:', y.shape, 'vectorizer:', vectorizer
    Xy = [X, y]

    return Xy


def classify_ads(Xy):
    classifier = BernoulliNB()
    #this line has something to do with how the cross validation function splits up the train and test data, I think.
    cv = cross_validation.StratifiedKFold(Xy[1],2)
    #print "this is cv", cv
    precision=[]
    recall=[]
    #train is a list of indeces that correspond to the training data (roughtly 75%)
    #test is a list of indeces that correspond to the test data (25%)
    for train, test in cv:
        #print "this is train:", train, "this is test", test
        X_train = Xy[0][train]
        X_test = Xy[0][test]
        y_train = Xy[1][train]
        y_test = Xy[1][test]
        classifier.fit(X_train, y_train)
        #look up linear algebra notation
        y_hat = classifier.predict(X_test)
        #why is this variable titled this?
        p,r,_,_ = metrics.precision_recall_fscore_support(y_test, y_hat)
        precision.append(p[1])
        recall.append(r[1])
    print classifier
    print 'precision:',np.average(precision), '+/-', np.std(precision)
    print 'recall:', np.average(recall), '+/-', np.std(recall)
    
    return classifier


def test_sample(vectorizer, classifier):
    sample = 'Long hair... Long Legs Tall, Busty , Beautiful, Luscious Lips and Curvy Hips<br> All Service<br> In or Out Call<br> Available Days and Nights<br> call 336 307 5841 |'
    sample = vectorizer.transform([sample])
    classification_new_ad = classifier.predict(sample)
    
    print classification_new_ad


def describe_features(vectorizer, classifier):
    probs=classifier.feature_log_prob_[1]
    features=vectorizer.get_feature_names()
    
    print 'length of probs:', len(probs), 'length of features:', len(features), 'list of most important features:', sorted(zip(probs,features), reverse=True)[:10]


def main():
    analyze_ad()

if __name__ == "__main__":
    main()



