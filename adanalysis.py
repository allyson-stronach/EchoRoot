import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn import cross_validation
from sklearn import metrics

from model import session


def analyze_ad():
    dl_list = retrieve_trafficky_text()
    dl_list = retrieve_not_trafficky_text(dl_list)
    vectorizer = TfidfVectorizer()
    x_axis_y_axis = vectorize_ads(dl_list, vectorizer)
    classifier = classify_ads(x_axis_y_axis)
    test_sample(vectorizer, classifier)
    describe_features(vectorizer, classifier)


def retrieve_trafficky_text():
    documents = []
    labels = []
    ads_text_cmd = "SELECT ads.text AS text FROM ads_attributes JOIN ads ON ads.id = ads_id WHERE ads_attributes.value IN  ('9292103206', '4142395461', '4146870501') "
    trafficky_text = session.execute(ads_text_cmd)
    for text in trafficky_text:
        documents.append(text.text)
        labels.append('trafficky')
    print 'documents length:', len(documents), 'labels length:', len(labels)
    dl_list = [documents, labels]

    return dl_list


def retrieve_not_trafficky_text(dl_list):
    ads_text_cmd = "SELECT ads.text AS text FROM ads_attributes JOIN ads ON ads.id = ads_id WHERE ads_attributes.value IN ('3104623985', '2139840845 ', '8183362736', '6032946322', '4088991922')"
    not_trafficky_text = session.execute(ads_text_cmd)
    for text in not_trafficky_text:
        dl_list[0].append(text.text)
        dl_list[1].append('not trafficky')
    print 'documents length:', len(dl_list[0]), 'labels length:', len(dl_list[1])
    print 'fraction trafficky:', len([item for item in dl_list[1] if item == 'trafficky'])/240.0

    return dl_list


def vectorize_ads(dl_list, vectorizer):
    X = vectorizer.fit_transform(dl_list[0])
    y = np.array(dl_list[1])
    print 'X.shape is:', X.shape, 'y.shape is:', y.shape, 'vectorizer:', vectorizer
    x_axis_y_axis = [X, y]

    return x_axis_y_axis


def classify_ads(x_axis_y_axis):
    classifier = BernoulliNB()
    cv = cross_validation.StratifiedKFold(x_axis_y_axis[1],5)
    precision=[]
    recall=[]
    for train, test in cv:
        X_train = x_axis_y_axis[0][train]
        X_test = x_axis_y_axis[0][test]
        y_train = x_axis_y_axis[1][train]
        y_test = x_axis_y_axis[1][test]
        classifier.fit(X_train, y_train)
        y_hat = classifier.predict(X_test)
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
    c = classifier.predict(sample)
    
    print c


def describe_features(vectorizer, classifier):
    probs=classifier.feature_log_prob_[1]
    features=vectorizer.get_feature_names()
    
    print 'length of probs:', len(probs), 'length of features:', len(features), 'list of most important features:', sorted(zip(probs,features), reverse=True)[:10]


def main():
    analyze_ad()

if __name__ == "__main__":
    main()



