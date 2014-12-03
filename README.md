EchoRoot
========

Overview
========

EchoRoot is a proof-of-concept app guided by one question: can machine learning techniques uncover signs of sex trafficking in online sex ads? 

Previous research has shown that across different ads, “identical phone numbers were sometimes used to advertise individuals of different ages/descriptions, in different locations, at different times." This research provides a starting point to analyze the text in sex ads from backpage.com. In an effort to use data science for social good, EchoRoot explores the application of a foundational machine learning concept, the Bayesian classifier, to exploit the very ads that enable human trafficking to be so pervasive. 

Features
========

Data scientist (user) can read in training text from a database, label text, filter text (removing HTML tags, UTF-literals, newline characters, and lowercasing all), convert text to a machine-readable TFIDF vector. The user can then train (and re-train) classifier. Type of classifier can be changed in one line of code. 

The user can then classify new input, and save information to dictionary, which can then be used to make a histogram, to see distributed classified data. 

Technologies and Stack
======================

This project uses a mySQL database with 1,000,000 ads from backpage.com.

This database is not included in this repo, and the SQL queries in adanalysis.py are specific to that information, so if used, must be created from scratch or adapted. 

<h4>Front-end</h4>

CSS, HTML, Twitter Bootstrap

<h4>Back-end</h4>

Python, SQLALchemy, mySQL, Flask
Scikit-learn, Regex, Pickle

Current State
=================
Classifier is still being updated and changed to accommodate new ads, and to maximize precision & recall while reducing overfit. 

Future Directions
=================
Attempt to use n-grams in classifier. 
Two different types of trafficky ads are currently in classifier; this should be split up into two classifiers to see if more meaningful features list emerges. 
Implement D3 visualization and live random ad function in data analysis UI. 
Clean up data in database. 
If, with changes, classifier is still overfitting data, try other classifiers (Logistic Regression and others).
Attempt to make a classifier that uses presence / non-presence of phone numbers, names, locations, etc. to determine trafficking instead of words. COmpare performance of various classifiers. 
Refactor by changing functions to classes to reduce redundant code. 
