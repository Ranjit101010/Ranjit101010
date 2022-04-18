import nltk

import pandas as pd
messages = pd.read_csv('SMSSpamCollection',sep = '\t',names = ['labels','message'])
print(messages)

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
ps = PorterStemmer()
ws = WordNetLemmatizer()

corpus = []
for i in range(0,len(messages)):
    review = re.sub('[^a-zA-Z]',' ',messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
print(corpus)

from sklearn.feature_extraction.text import TfidfVectorizer

tf_idf = TfidfVectorizer(max_features = 5000)
X = tf_idf.fit_transform(corpus).toarray()
y = pd.get_dummies(messages['labels'])

y = y.iloc[:,1].values

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)

from sklearn.naive_bayes import MultinomialNB

spam_detect_model = MultinomialNB()
spam_detect_model = spam_detect_model.fit(X_train,Y_train)
y_pred = spam_detect_model.predict(X_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
c_m = confusion_matrix(Y_test,y_pred)
accuracy_score = accuracy_score(Y_test,y_pred)
