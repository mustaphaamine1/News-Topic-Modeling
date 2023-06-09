import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora
from string import punctuation
from nltk.tokenize import word_tokenize
from collections import Counter
import operator
import numpy as np
import nltk
from textblob import TextBlob
import string
from gensim.models.ldamodel import LdaModel
from gensim.test.utils import datapath
from pathlib import Path
import warnings
nltk.download('punkt')
nltk.download('stopwords')
warnings.filterwarnings('ignore')

if os.path.exists(str(Path(__file__).resolve().parent.parent.parent)+'/models/search_lda_model'):
    os.remove(str(Path(__file__).resolve().parent.parent.parent)+'/models/search_lda_model')

if os.path.exists(str(Path(__file__).resolve().parent.parent.parent)+'/models/latest_lda_model'):
    os.remove(str(Path(__file__).resolve().parent.parent.parent)+'/models/latest_lda_model')


search_news_pro=pd.read_csv(str(Path(__file__).resolve().parent.parent.parent)+'/data/processed/processed_search_news_dataset.csv')
latest_news_pro=pd.read_csv(str(Path(__file__).resolve().parent.parent.parent)+'/data/processed/processed_latest_news_dataset.csv')

latest_news_pro.fillna('', inplace=True)
search_news_pro.fillna('', inplace=True)

latest_news_pro['content']=latest_news_pro['content'].str.replace("[^a-zA-Z#]", " ")
search_news_pro['content']=search_news_pro['content'].str.replace("[^a-zA-Z#]", " ")

def clean_doc(text):
    #for token in text:
    text =  ' '.join([w.lower() for w in text.split() if len(w)>3])
    return text
tokenized_search_news_pro = search_news_pro['content'].apply(clean_doc)
tokenized_latest_news_pro = latest_news_pro['content'].apply(clean_doc)

stopwords_set = set(stopwords.words('english'))
custom = list(stopwords_set)+list(punctuation)
def stopWordRemoval(text):
    text = word_tokenize(text)
    text = ' '.join([word for word in text if word not in custom])
    return text

tokenized_latest_news_pro = tokenized_latest_news_pro.apply(stopWordRemoval)
tokenized_search_news_pro = tokenized_search_news_pro.apply(stopWordRemoval)


# Initializing wordnet lemmatizer
lemmatizer = WordNetLemmatizer()

def lemData(text):
    text = word_tokenize(text)
    newText = []
    for word in text:
        newText.append(lemmatizer.lemmatize(word))
    return ' '.join(newText)

tokenized_search_news_pro = tokenized_search_news_pro.apply(lemData)
tokenized_latest_news_pro = tokenized_latest_news_pro.apply(lemData)

stop = stopwords_set
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

# Function to lemmatize and remove the stopwords
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = "".join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


list_search_news_pro = tokenized_search_news_pro.tolist()
list_latest_news_pro = tokenized_latest_news_pro.tolist()

clean_search_news_pro = [clean(doc).split() for doc in list_search_news_pro]
clean_latest_news_pro = [clean(doc).split() for doc in list_latest_news_pro]





#LDA

# Creating the dictionary 
dictionary1 = corpora.Dictionary(clean_search_news_pro)
# Creating the corpus
search_news_term_matrix = [dictionary1.doc2bow(doc) for doc in clean_search_news_pro]

# Creating the dictionary 
dictionary2 = corpora.Dictionary(clean_latest_news_pro)
# Creating the corpus
latest_news_term_matrix = [dictionary2.doc2bow(doc) for doc in clean_latest_news_pro]

# Creating the LDA model
search_ldamodel = LdaModel(corpus=search_news_term_matrix, num_topics=4,id2word=dictionary1, random_state=20, passes=30)
latest_ldamodel = LdaModel(corpus=latest_news_term_matrix, num_topics=4,id2word=dictionary2, random_state=20, passes=30)





#saving models to disk.

temp_file1 = datapath(str(Path(__file__).resolve().parent.parent.parent)+'/models/search_lda_model')

search_ldamodel.save(temp_file1)


temp_file2 = datapath(str(Path(__file__).resolve().parent.parent.parent)+'/models/latest_lda_model')

latest_ldamodel.save(temp_file2)

p1=str(Path(__file__).resolve().parent.parent.parent)+'/models/search_lda_model'
p2=str(Path(__file__).resolve().parent.parent.parent)+'/models/latest_lda_model'
if os.path.exists(p1):
    print('Success !! \nthe LDA model is stored at {}'.format(p1))
else:
    print('the path to the file is not correct')

if os.path.exists(p2):
    print('Success !! \nthe LDA model is stored at {}'.format(p2))
else:
    print('the path to the file is not correct')