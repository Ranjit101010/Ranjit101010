

from gensim.models import Word2Vec

import nltk
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize

text = ''' There are many types of sentences, all with different structures and complexities. In its most basic form, a sentence is made up of a subject and predicate, which is the verb and the words that follow. But no matter how simple or complex, a sentence consists of words. Words in a sentence are what make it come alive and make sense.
Understand how words are used within the sentence, no matter the structure, and get inspiration for writing your own sentence correctly with the help of these example sentences. '''

# Data preprocessing
text = re.sub(r'\[[0-9]*\]',' ',text)
text = re.sub(r'\s+',' ',text)
text = text.lower()
text = re.sub(r'\d',' ',text)
text = re.sub(r'\s+',' ',text)

text = sent_tokenize(text)

    
sentences = [word_tokenize(text) for text in text]
for i in range(0,len(text)):
    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]

model = Word2Vec(sentences,min_count=1)
words = model.wv.key_to_index
vector = model.wv['many']

similar = model.wv.most_similar('many')
