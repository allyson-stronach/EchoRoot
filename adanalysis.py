import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn import cross_validation
from sklearn import metrics

from model import session


def retrieve_trafficky_text():
    documents = []
    labels = []
    ads_text_cmd = "SELECT ads.text AS text FROM ads_attributes JOIN ads ON ads.id = ads_id WHERE ads_attributes.value IN  ('9292103206', '4142395461', '4146870501') "
    e = session.execute(ads_text_cmd)
    for text in e:
        documents.append(text.text)
        labels.append('trafficky')
    print 'documents length:', len(documents), 'labels length:', len(labels)
    dl_list = [documents, labels]

    return dl_list


def retrieve_not_trafficky_text(dl_list):
    ads_text_cmd = "SELECT ads.text AS text FROM ads_attributes JOIN ads ON ads.id = ads_id WHERE ads_attributes.value IN ('3104623985', '2139840845 ', '8183362736', '6032946322', '4088991922')"
    e = session.execute(ads_text_cmd)
    for text in e:
        dl_list[0].append(text.text)
        dl_list[1].append('not trafficky')
    print 'documents length:', len(dl_list[0]), 'labels length:', len(dl_list[1])
    print 'fraction trafficky:', len([item for item in dl_list[1] if item == 'trafficky'])/240.0

    return dl_list

def instantiate_vectorizer():
    vectorizer = TfidfVectorizer()

    return vectorizer

def vectorize_ads(dl_list, vectorizer):
    X = vectorizer.fit_transform(dl_list[0])
    y = np.array(dl_list[1])
    print 'X.shape is:', X.shape, 'y.shape is:', y.shape, 'vectorizer:', vectorizer
    xy = [X, y]

    return xy


def classify_ads(xy):
    clf = BernoulliNB()
    cv = cross_validation.StratifiedKFold(xy[1],5)
    precision=[]
    recall=[]
    for train, test in cv:
        X_train = xy[0][train]
        X_test = xy[0][test]
        y_train = xy[1][train]
        y_test = xy[1][test]
        clf.fit(X_train, y_train)
        y_hat = clf.predict(X_test)
        p,r,_,_ = metrics.precision_recall_fscore_support(y_test, y_hat)
        precision.append(p[1])
        recall.append(r[1])
    print clf
    print 'precision:',np.average(precision), '+/-', np.std(precision)
    print 'recall:', np.average(recall), '+/-', np.std(recall)
    
    return clf


def test_sample(vectorizer, clf):
    sample = 'Long hair... Long Legs Tall, Busty , Beautiful, Luscious Lips and Curvy Hips<br> All Service<br> In or Out Call<br> Available Days and Nights<br> call 336 307 5841 |'
    sample = vectorizer.transform([sample])
    c = clf.predict(sample)
    
    print c


def describe_features(vectorizer, clf):
    probs=clf.feature_log_prob_[1]
    features=vectorizer.get_feature_names()
    
    print 'length of probs:', len(probs), 'length of features:', len(features), 'list of most important features:', sorted(zip(probs,features), reverse=True)[:10]


def main():
    dl_list = retrieve_trafficky_text()
    dl_list = retrieve_not_trafficky_text(dl_list)
    vectorizer = instantiate_vectorizer()
    xy = vectorize_ads(dl_list, vectorizer)
    clf = classify_ads(xy)
    test_sample(vectorizer, clf)
    describe_features(vectorizer, clf)

if __name__ == "__main__":
    main()



