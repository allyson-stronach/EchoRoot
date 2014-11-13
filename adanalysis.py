from model import session, AdAttribute, Ad
import nltk

"""querying the database for traffick-y ads"""

def retrieve_trafficky_text():
	trafficky_ad_list = []
	trafficky_ads_cmd = "SELECT ads.text AS text FROM ads_attributes JOIN ads ON ads.id = ads_id WHERE ads_attributes.value = '9292103206' OR '4142395461' OR '4146870501' "
	e = session.execute(trafficky_ads_cmd)
	for text in e:
		trafficky_ad_list.append((text.text, 'trafficky'))
	return trafficky_ad_list










def main():
    tl = retrieve_trafficky_text()


if __name__ == "__main__":
    main()

# trafficky_ads = []
# non_trafficky_ads = []
# ads = []
# for (words, label) in trafficky_ads + non_trafficky_ads:
#     words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
#     tweets.append((words_filtered, label))
# test_ads = []

