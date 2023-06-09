import json
import dash
from dash import dcc
from dash import html

from dash import dash_table
import plotly.express as px
import urllib.request as urllib
#from urllib.request import urlopen
import pandas as pd
import plotly.graph_objs as go
import plotly.figure_factory as ff
import numpy as np

from nltk.util import ngrams
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from iso3166 import countries
import os, shutil
import iso3166
import re





import dash_bootstrap_components as dbc

from dash.dependencies import Input, Output, State
from dash import Dash, Input, Output, State, html, dcc, dash_table, callback
#the news api client
from newsapi.newsapi_client import NewsApiClient
from pathlib import Path
import click

#libraries
from pandas import json_normalize
from bs4 import BeautifulSoup
from wordcloud import WordCloud
from newspaper import Article
import datetime
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os


import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
import nltk
from nltk.stem.snowball import SnowballStemmer
from pathlib import Path
from dash import dcc
import os
import pandas as pd
import nltk
from nltk.corpus import stopwords

from gensim import corpora
from string import punctuation
from nltk.tokenize import word_tokenize
from collections import Counter
import operator
import numpy as np
from textblob import TextBlob
import string
from gensim.models.ldamodel import LdaModel
from gensim.test.utils import datapath
import warnings


from dash.dependencies import Input, Output
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis


import plotly.express as px
from nltk.stem import WordNetLemmatizer



import warnings
import matplotlib
warnings.filterwarnings("ignore",category=UserWarning)
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 


nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('omw-1.4')
warnings.filterwarnings('ignore')
nltk.download('wordnet')
nltk.download('omw-1.4')


from dash_bootstrap_templates import load_figure_template
load_figure_template('LUX')

import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
from collections import Counter

###################################################
# design for mapbox
bgcolor = "#f3f3f1"  # mapbox light map land color
row_heights = [150, 500, 300]
template = {"layout": {"paper_bgcolor": bgcolor, "plot_bgcolor": bgcolor}}







ots = ['business','entertainment','general','health','science','sports','technology']

food_options = [
    dict(label=country, value=country)
    for country in ots]








ots1 = [x for x in iso3166.countries_by_alpha2]
ots_ = [
    dict(label=country, value=country)
    for country in ots1]

ots_behaviour = dcc.Dropdown(
        id='box_dd',
        options=ots_,
        value="US"
    )

radio_food_behaviour = dcc.RadioItems(
                            id='nutrition_types',
                            options=food_options,
                            value="general",
                            labelStyle={'display': 'block', "text-align": "justify"}
                            
                        )

                    

#####################################################################################



def date(base):    
    date_list=[]    
    yr=datetime.today().year    
    if (yr%400)==0 or ((yr%100!=0) and (yr%4==0)):          
        numdays=366        
        date_list.append([base - timedelta(days=x) for x in    
        range(366)])   
    else:        
        numdays=365        
        date_list.append([base - timedelta(days=x) for x in    
        range(365)])    
    newlist=[]    
    for i in date_list:        
        for j in sorted(i):            
            newlist.append(j)    
        return newlist  

#the last_30 function is to get the sorted list for the ;last 30 days starting from the date 'base' to 'base-30'
def last_30(base):     
    date_list=[base - timedelta(days=x) for x in range(30)]      
    return sorted(date_list) 
#the from_dt func is to get the list of the start days for the news like '[01/01/2001,02/01/2001,..]'
def from_dt(x):    
    from_dt=[]    
    for i in range(len(x)):          
        from_dt.append(last_30(datetime.today())[i-1].date())         
    return from_dt 

#the from_dt func is to get the list of the start days for the news like '[02/01/2001,03/01/2001,..]'
def to_dt(x):    
    to_dt=[]    
    for i in range(len(x)):        
        to_dt.append(last_30(datetime.today())[i].date())    
    return to_dt

from_list=from_dt(last_30(datetime.today()))
to_list=to_dt(last_30(datetime.today())) 



api_key='58fb265cc7aa463480ac2a9f2b0b1f69'
#str(input('Enter the api key from NewsAPI website:\t'))
#input the parameters for the newsapi
#print('The category you want to get headlines for:\n Possible options:\n1 --> business\n2 --> entertainment\n3 --> general\n4 --> health\n5 --> science\n6 --> sports\n7 --> technology')
#num_cat=int(input())


def get_latest_news(api_key,num_cat,count):

    if os.path.exists(str(Path(__file__).resolve().parent.parent.parent)+'\\data\\raw\\raw_latest_news_dataset.csv'):
        os.remove(str(Path(__file__).resolve().parent.parent.parent)+'\\data\\raw\\raw_latest_news_dataset.csv')
    #you need to create an account in NewsAPI website to get the api key 
    newsapi = NewsApiClient(api_key=api_key)
    dict1={1:'business',2:'entertainment',3:'general',4:'health',5:'science',6:'sports',7:'technology'}
    cat=num_cat
    #dict1[num_cat]
    top_headlines=newsapi.get_top_headlines(category=cat,country=count)
    top_headlines=pd.json_normalize(top_headlines['articles']) 
    newdf=top_headlines
    for k,v in zip(newdf.index,newdf["url"]):
            try:
                article=Article(v)
                article.download()
                article.parse()
                try:
                    t =str(article.text)
                except:
                    t=''
                newdf['content'][k]=t
            except:
                pass 
    '''newdf=top_headlines[["title","url","source.name","publishedAt"]]   
    #newdf=pd.DataFrame(newdf) 
    dic=newdf.set_index('title')['url'].to_dict()'''
    #return top_headlines
    #newdf
    path=str(Path(__file__).resolve().parent.parent.parent)+'\\data\\raw\\raw_latest_news_dataset.csv'
    newdf.to_csv(path)


#print('Enter the topic to seach :\n')
#keyword=input()



####################################################

def search_news(api_key,term):
    if os.path.exists(str(Path(__file__).resolve().parent.parent.parent)+'\\data\\raw\\raw_search_news_dataset.csv'):
        os.remove(str(Path(__file__).resolve().parent.parent.parent)+'\\data\\raw\\raw_search_news_dataset.csv')

    #you need to create an account in NewsAPI website to get the api key 
    newsapi = NewsApiClient(api_key=api_key)

#the query parameter which is the search term for the get_everything function.
    newsd={}
    newsdf=pd.DataFrame()
    #for (from_dt,to_dt) in zip(from_list,to_list):  for the past 30 days
    #for (from_dt,to_dt) in zip(from_list[0],to_list[0]): 
    from_dt=from_list[0]
    to_dt=to_list[0]
    all_articles = newsapi.get_everything(q=term,language='en',sort_by='relevancy',from_param=from_dt,to=to_dt)          
    d=json_normalize(all_articles['articles'])      
    #'''newdf=d[["url","description","publishedAt","source.name","author"]]
    #dic=newdf.set_index(["source.name","publishedAt","author"])["url"].to_dict()'''
    for k,v in zip(d.index,d['url']):
        try:
            article=Article(v)
            article.download()
            article.parse()
            try:
                t =str(article.text)
            except:
                t=''
            d['content'][k]=t
        except:
            pass
    #return d
    path=str(Path(__file__).resolve().parent.parent.parent)+'\\data\\raw\\raw_search_news_dataset.csv'
    d.to_csv(path)

#remove unwanted columns
'''
latest_news_raw=latest_news_raw.iloc[:, [ False, True, True,True,True, False,True,True, False,True]]
search_news_raw=search_news_raw.iloc[:, [ False, True, True,True,True, False,True,True, False,True]]
'''


######################
def clean_text(text):
    #for token in text:
    text=re.sub('[^A-Za-z0-9]+', ' ', text).strip()
    text=re.sub('[0-9]+', '', text)
    return text
###################



def clean_data():

    #remove old files
    if os.path.exists(str(Path(__file__).resolve().parent.parent.parent)+'\\data\\processed\\processed_search_news_dataset.csv'):
        os.remove(str(Path(__file__).resolve().parent.parent.parent)+'\\data\\processed\\processed_search_news_dataset.csv')

    if os.path.exists(str(Path(__file__).resolve().parent.parent.parent)+'\\data\\processed\\processed_latest_news_dataset.csv'):
        os.remove(str(Path(__file__).resolve().parent.parent.parent)+'\\data\\processed\\processed_latest_news_dataset.csv')


    #read the raw dataset 

    if os.path.exists(str(Path(__file__).resolve().parent.parent.parent)+'\\data\\raw\\raw_search_news_dataset.csv'):
        search_news_raw=pd.read_csv(str(Path(__file__).resolve().parent.parent.parent)+'\\data\\raw\\raw_search_news_dataset.csv')
    if os.path.exists(str(Path(__file__).resolve().parent.parent.parent)+'\\data\\raw\\raw_latest_news_dataset.csv'):
        latest_news_raw=pd.read_csv(str(Path(__file__).resolve().parent.parent.parent)+'\\data\\raw\\raw_latest_news_dataset.csv')

    latest_news_raw=latest_news_raw.rename(columns={'publishedAt':'Date','source.name':'Source'})
    search_news_raw=search_news_raw.rename(columns={'publishedAt':'Date','source.name':'Source'})

    #fix date column
    latest_news_raw['Date']=pd.to_datetime(latest_news_raw['Date'], format='%Y-%m-%dT%H:%M:%SZ')
    search_news_raw['Date']=pd.to_datetime(search_news_raw['Date'], format='%Y-%m-%dT%H:%M:%SZ')


    latest_news_raw.fillna('', inplace=True)
    search_news_raw.fillna('', inplace=True)

    # remove unwanted characters, numbers and symbols
    latest_news_raw['content']=latest_news_raw['content'].apply(clean_text)
    search_news_raw['content']=search_news_raw['content'].apply(clean_text)

    #os.remove(str(Path(__file__).resolve().parent.parent.parent)+'/data/raw/raw_latest_news_dataset.csv')
    #os.remove(str(Path(__file__).resolve().parent.parent.parent)+'/data/raw/raw_latest_news_dataset.csv')
    #latest_news_raw['content'] = latest_news_raw['content'].apply(clean_doc)
    #search_news_raw['content'] = search_news_raw['content'].apply(clean_doc)

    path=str(Path(__file__).resolve().parent.parent.parent)+'\\data\\processed\\processed_latest_news_dataset.csv'
    latest_news_raw.to_csv(path)

    path=str(Path(__file__).resolve().parent.parent.parent)+'\\data\\processed\\processed_search_news_dataset.csv'
    search_news_raw.to_csv(path)


###########################################################################


stopwords_set = set(stopwords.words('english'))
custom = list(stopwords_set)+list(punctuation)
stop = stopwords_set
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean_doc(text):        
    #for token in text:
    text =  ' '.join([w.lower() for w in text.split() if len(w)>3])
    return text

def stopWordRemoval(text):
    text = word_tokenize(text)
    text = ' '.join([word for word in text if word not in custom])
    return text
# Initializing wordnet lemmatizer
lemmatizer = WordNetLemmatizer()

def lemData(text):
    text = word_tokenize(text)
    newText = []
    for word in text:
        newText.append(lemmatizer.lemmatize(word))
    return ' '.join(newText)



# Function to lemmatize and remove the stopwords
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    #punc_free = " ".join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in stop_free.split())
    return normalized


def topic_model():

    folder = str(Path(__file__).resolve().parent.parent.parent)+'\\models'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))



    #if os.path.exists(str(Path(__file__).resolve().parent.parent.parent)+'\\models'):
        #
    #if os.path.exists(str(Path(__file__).resolve().parent.parent.parent)+'\\models\\latest_lda_model'):
       # os.remove(str(Path(__file__).resolve().parent.parent.parent)+'\\models\\latest_lda_model')

    search_news_pro=pd.read_csv(str(Path(__file__).resolve().parent.parent.parent)+'\\data\\processed\\processed_search_news_dataset.csv')
    latest_news_pro=pd.read_csv(str(Path(__file__).resolve().parent.parent.parent)+'\\data\\processed\\processed_latest_news_dataset.csv')

    latest_news_pro.fillna('', inplace=True)
    search_news_pro.fillna('', inplace=True)




    tokenized_search_news_pro = search_news_pro['content'].apply(clean_doc)
    tokenized_latest_news_pro = latest_news_pro['content'].apply(clean_doc)


    tokenized_latest_news_pro = tokenized_latest_news_pro.apply(stopWordRemoval)
    tokenized_search_news_pro = tokenized_search_news_pro.apply(stopWordRemoval)

    tokenized_search_news_pro = tokenized_search_news_pro.apply(lemData)
    tokenized_latest_news_pro = tokenized_latest_news_pro.apply(lemData)

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

    temp_file1 = datapath(str(Path(__file__).resolve().parent.parent.parent)+'\\models\\search_lda_model')
    search_ldamodel.save(temp_file1)
    temp_file2 = datapath(str(Path(__file__).resolve().parent.parent.parent)+'\\models\\latest_lda_model')
    latest_ldamodel.save(temp_file2)

#func to get the lemma and stemmed form of word
def lemmatize_stemming(text):
    return SnowballStemmer(language='english').stem(WordNetLemmatizer().lemmatize(text, pos='v'))

    
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


def latest_wordcloud():
    from wordcloud import STOPWORDS
    from wordcloud import WordCloud
    processed_latest_news=pd.read_csv(str(Path(__file__).resolve().parent.parent.parent)+'\\data\\processed\\processed_latest_news_dataset.csv')

    processed_docs = processed_latest_news['content'].fillna('').astype(str).map(preprocess)
    

    # Join multiple lists
    l=''

    for i in processed_latest_news['content']:
        #print(i)
        l=l+str(i)

    excluded_words = list(STOPWORDS)
    wordcloud_image = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(str(l)) 

    fig = go.Figure()
    fig.add_trace(go.Image(z=wordcloud_image))
    fig.update_layout(
        height=600,
    xaxis={"visible": False},
    yaxis={"visible": False},
    margin={"t": 0, "b": 0, "l": 0, "r": 0},
    hovermode=False,
    paper_bgcolor="#F9F9FA",
    plot_bgcolor="#F9F9FA",
        )
    return fig

def Search_wordcloud():
    processed_search_news=pd.read_csv(str(Path(__file__).resolve().parent.parent.parent)+'\\data\\processed\\processed_search_news_dataset.csv')
    processed_docs = processed_search_news['content'].fillna('').astype(str).map(preprocess)
    excluded_words = list(STOPWORDS)
    s=''

    for i in processed_search_news['content']:
        #print(i)
        s=s+str(i)

    wordcloud_image = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(str(s)) 

    fig = go.Figure()
    fig.add_trace(go.Image(z=wordcloud_image))
    fig.update_layout(
        height=600,
    xaxis={"visible": False},
    yaxis={"visible": False},
    margin={"t": 0, "b": 0, "l": 0, "r": 0},
    hovermode=False,
    paper_bgcolor="#F9F9FA",
    plot_bgcolor="#F9F9FA",
        )
    return fig


# function to plot most frequent terms
def freq_search_words(terms = 30):
    #all_words = ' '.join([text for text in x])
    #all_words = x.split(' ')
    processed_search_news=pd.read_csv(str(Path(__file__).resolve().parent.parent.parent)+'\\data\\processed\\processed_search_news_dataset.csv')
    processed_docs = processed_search_news['content'].fillna('').astype(str).map(preprocess)
    excluded_words = list(STOPWORDS)
    s=''

    for i in processed_search_news['content']:
        #print(i)
        s=s+str(i)
    ps=preprocess(s)
    all_words = ps

    fdist = nltk.FreqDist(all_words)
    words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

    # selecting top 20 most frequent words
    d = words_df.nlargest(columns="count", n = terms) 
    d=d.sort_values(by='count',ascending=True)
    fig = px.bar(d,x = "count",y= "word",
             hover_data=['word', 'count'], color='count',
             labels={'count':'frequecy of the word'}, height=400)
    return fig

# function to plot most frequent terms
def freq_latest_words(terms = 30):
    #all_words = ' '.join([text for text in x])
    #all_words = x.split(' ')
    processed_latest_news=pd.read_csv(str(Path(__file__).resolve().parent.parent.parent)+'\\data\\processed\\processed_latest_news_dataset.csv')

    processed_docs = processed_latest_news['content'].fillna('').astype(str).map(preprocess)
    
    # Join multiple lists
    l=''

    for i in processed_latest_news['content']:
        #print(i)
        l=l+str(i)

    pl=preprocess(l)
    all_words = pl

    fdist = nltk.FreqDist(all_words)
    words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

    # selecting top 20 most frequent words
    d = words_df.nlargest(columns="count", n = terms) 
    d=d.sort_values(by='count',ascending=True)
    fig = px.bar(d,x = "count",y= "word",
             hover_data=['word', 'count'], color='count',
             labels={'count':'frequecy of the word'},title='Top words', height=400)
    return fig


def sourse_plot():
    processed_latest_news=pd.read_csv(str(Path(__file__).resolve().parent.parent.parent)+'\\data\\processed\\processed_latest_news_dataset.csv')

    df=pd.DataFrame({'News source':processed_latest_news['Source'].value_counts().index,'Count':processed_latest_news['Source'].value_counts().values})
    fig = px.pie(df, values='Count', names='News source', title='News source distribution')
    return fig

def search_sourse_plot():
    processed_latest_news=pd.read_csv(str(Path(__file__).resolve().parent.parent.parent)+'\\data\\processed\\processed_search_news_dataset.csv')

    df=pd.DataFrame({'News source':processed_latest_news['Source'].value_counts().index,'Count':processed_latest_news['Source'].value_counts().values})
    fig = px.pie(df, values='Count', names='News source', title='News source distribution')
    return fig


def convertTuple(tup):
    str1 = ' '.join(tup)
    return str1

# function to plot most frequent terms
def freq_two_words():
    processed_latest_news=pd.read_csv(str(Path(__file__).resolve().parent.parent.parent)+'\\data\\processed\\processed_latest_news_dataset.csv')
    processed_docs = processed_latest_news['content'].fillna('').astype(str).map(preprocess)
    
    # Join multiple lists
    l=''

    for i in processed_latest_news['content']:
        #print(i)
        l=l+str(i)

    pl=preprocess(l)
    bigram_fd = nltk.FreqDist(nltk.bigrams(pl))

    l1=[]
    l2=[]
    bigram_fd.most_common()
    for i in range(20):
        l1.append(bigram_fd.most_common()[i][0])
        l2.append(bigram_fd.most_common()[i][1])

    df=pd.DataFrame({'two words':l1,'count':l2})
    df['two words']=df['two words'].apply(convertTuple)
    df=df.sort_values(by='count',ascending=True)
    fig = px.bar(df,x = "count",y= "two words"
             , color='count',
             labels={'count':'frequecy'},title='Twp word phrases ', height=400)
    return fig
    
def search_freq_two_words():
    processed_latest_news=pd.read_csv(str(Path(__file__).resolve().parent.parent.parent)+'\\data\\processed\\processed_search_news_dataset.csv')
    processed_docs = processed_latest_news['content'].fillna('').astype(str).map(preprocess)
    
    # Join multiple lists
    l=''

    for i in processed_latest_news['content']:
        #print(i)
        l=l+str(i)

    pl=preprocess(l)
    bigram_fd = nltk.FreqDist(nltk.bigrams(pl))

    l1=[]
    l2=[]
    bigram_fd.most_common()
    for i in range(20):
        l1.append(bigram_fd.most_common()[i][0])
        l2.append(bigram_fd.most_common()[i][1])

    df=pd.DataFrame({'two words':l1,'count':l2})
    df['two words']=df['two words'].apply(convertTuple)
    df=df.sort_values(by='count',ascending=True)
    fig = px.bar(df,x = "count",y= "two words"
             , color='count',
             labels={'count':'frequecy'},title='Twp word phrases ', height=400)
    return fig
    
def freq_three_words():
    processed_latest_news=pd.read_csv(str(Path(__file__).resolve().parent.parent.parent)+'\\data\\processed\\processed_latest_news_dataset.csv')
    processed_docs = processed_latest_news['content'].fillna('').astype(str).map(preprocess)
    
    # Join multiple lists
    l=''

    for i in processed_latest_news['content']:
        #print(i)
        l=l+str(i)

    pl=preprocess(l)
    trigram_fd = nltk.FreqDist(nltk.trigrams(pl))
    l1=[]
    l2=[]
    trigram_fd.most_common()
    for i in range(20):
        l1.append(trigram_fd.most_common()[i][0])
        l2.append(trigram_fd.most_common()[i][1])

    df1=pd.DataFrame({'three words':l1,'count':l2})
    df1['three words']=df1['three words'].apply(convertTuple)

    df1=df1.sort_values(by='count',ascending=True)
    fig = px.bar(df1,x = "count",y= "three words",
             hover_data=['three words', 'count'], color='count',
             labels={'count':'frequecy'},title='Three word phrases', height=400)
    return fig


def search_freq_three_words():
    processed_latest_news=pd.read_csv(str(Path(__file__).resolve().parent.parent.parent)+'\\data\\processed\\processed_search_news_dataset.csv')
    processed_docs = processed_latest_news['content'].fillna('').astype(str).map(preprocess)
    
    # Join multiple lists
    l=''

    for i in processed_latest_news['content']:
        #print(i)
        l=l+str(i)

    pl=preprocess(l)
    trigram_fd = nltk.FreqDist(nltk.trigrams(pl))
    l1=[]
    l2=[]
    trigram_fd.most_common()
    for i in range(20):
        l1.append(trigram_fd.most_common()[i][0])
        l2.append(trigram_fd.most_common()[i][1])

    df1=pd.DataFrame({'three words':l1,'count':l2})
    df1['three words']=df1['three words'].apply(convertTuple)

    df1=df1.sort_values(by='count',ascending=True)
    fig = px.bar(df1,x = "count",y= "three words",
             hover_data=['three words', 'count'], color='count',
             labels={'count':'frequecy'},title='Three word phrases', height=400)
    return fig

'''
def topic_model_plot():
    search_news_pro=pd.read_csv(str(Path(__file__).resolve().parent.parent.parent)+'/data/processed/processed_search_news_dataset.csv')
    latest_news_pro=pd.read_csv(str(Path(__file__).resolve().parent.parent.parent)+'/data/processed/processed_latest_news_dataset.csv')

    latest_news_pro.fillna('', inplace=True)
    search_news_pro.fillna('', inplace=True)

    latest_news_pro['content']=latest_news_pro['content'].str.replace("[^a-zA-Z#]", " ")
    search_news_pro['content']=search_news_pro['content'].str.replace("[^a-zA-Z#]", " ")


    tokenized_search_news_pro = search_news_pro['content'].apply(clean_doc)
    tokenized_latest_news_pro = latest_news_pro['content'].apply(clean_doc)

    stopwords_set = set(stopwords.words('english'))
    custom = list(stopwords_set)+list(punctuation)


    tokenized_latest_news_pro = tokenized_latest_news_pro.apply(stopWordRemoval)
    tokenized_search_news_pro = tokenized_search_news_pro.apply(stopWordRemoval)

    tokenized_search_news_pro = tokenized_search_news_pro.apply(lemData)
    tokenized_latest_news_pro = tokenized_latest_news_pro.apply(lemData)

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

    lda_model = gensim.models.LdaModel.load(datapath(str(Path(__file__).resolve().parent.parent.parent)+'/models/search_lda_model'))
    #corpus = gensim.corpora.MmCorpus(latest_news_term_matrix)
    vis_data = gensimvis.prepare(lda_model, latest_news_term_matrix, dictionary=lda_model.id2word)
    pyLDAvis.save_html(vis_data, 'lda.html')
    # Convert the pyLDAvis visualization to a Plotly figure
    #fig= pyLDAvis.display(vis_data)
    
    #return fig



'''



########


def plot_topic(d,topic_id):
    search_news_pro=pd.read_csv(str(Path(__file__).resolve().parent.parent.parent)+'\\data\\processed\\processed_search_news_dataset.csv')
    latest_news_pro=pd.read_csv(str(Path(__file__).resolve().parent.parent.parent)+'\\data\\processed\\processed_latest_news_dataset.csv')

    latest_news_pro.fillna('', inplace=True)
    search_news_pro.fillna('', inplace=True)




    tokenized_search_news_pro = search_news_pro['content'].apply(clean_doc)
    tokenized_latest_news_pro = latest_news_pro['content'].apply(clean_doc)


    tokenized_latest_news_pro = tokenized_latest_news_pro.apply(stopWordRemoval)
    tokenized_search_news_pro = tokenized_search_news_pro.apply(stopWordRemoval)

    tokenized_search_news_pro = tokenized_search_news_pro.apply(lemData)
    tokenized_latest_news_pro = tokenized_latest_news_pro.apply(lemData)

    list_search_news_pro = tokenized_search_news_pro.tolist()
    list_latest_news_pro = tokenized_latest_news_pro.tolist()

    clean_search_news_pro = [clean(doc).split() for doc in list_search_news_pro]
    clean_latest_news_pro = [clean(doc).split() for doc in list_latest_news_pro]


    #LDA
    if d=='s':
            # Creating the dictionary 
        dictionary1 = corpora.Dictionary(clean_search_news_pro)
        # Creating the corpus
        search_news_term_matrix = [dictionary1.doc2bow(doc) for doc in clean_search_news_pro]
        search_lda_model = gensim.models.LdaModel.load(datapath(str(Path(__file__).resolve().parent.parent.parent)+'\\models\\search_lda_model'))
        lda_model=search_lda_model
        


    if d =='l':
        # Creating the dictionary 
        dictionary2 = corpora.Dictionary(clean_latest_news_pro)
        # Creating the corpus
        latest_news_term_matrix = [dictionary2.doc2bow(doc) for doc in clean_latest_news_pro]
        latest_lda_model = gensim.models.LdaModel.load(datapath(str(Path(__file__).resolve().parent.parent.parent)+'\\models\\latest_lda_model'))
        lda_model=latest_lda_model



    topics = lda_model.show_topics(formatted=False)
    data_flat = [w for w_list in clean_search_news_pro for w in w_list]
    counter = Counter(data_flat)

    out = []
    for i, topic in topics:
        for word, weight in topic:
            out.append([word, i, weight, counter[word]])

    df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])

    # Create figure
    fig = go.Figure()

    cols = [color for color in pio.templates['simple_white'].layout.colorway]

  # Select the topic ID you want to plot
#   topic_id = 2
    topic_df = df.loc[df.topic_id == topic_id, :]

    # Add Word Count Bar Trace
    fig.add_trace(
        go.Bar(
            
            x=topic_df['word'],
            y=topic_df['word_count'],
            marker=dict(color=cols[topic_id], opacity=0.3),
            width=0.5,
            name='Word Count',
            yaxis='y1'
        )
    )

    # Add Importance Bar Trace
    fig.add_trace(
        go.Bar(
            x=topic_df['word'],
            y=topic_df['importance'],
            marker=dict(color=cols[topic_id]),
            width=0.2,
            name='Weights',
            yaxis='y2'
        )
    )

    fig.update_layout(
        xaxis=dict(tickangle=30),
        yaxis=dict(title='Word Count', color=cols[topic_id], range=[0, 3500]),
        yaxis2=dict(title='Weights', color=cols[topic_id], range=[0, 0.03], overlaying='y', side='right'),
        barmode='overlay',
        legend=dict(x=0, y=1.2, orientation='h')
    )

    fig.update_traces(showlegend=True, selector=dict(type='bar'))
    fig.update_layout(template='simple_white')

    fig.update_layout(
        title='Word Count and Importance of Topic Keywords (Topic {})'.format(topic_id),
        title_font=dict(size=16),
        height=400,
        width=600,
        showlegend=True
    )

    return fig



'''

path=str(Path(__file__).resolve().parent.parent.parent)+'/data/raw/raw_latest_news_dataset.csv'
s.to_csv(path)
if os.path.exists(path):
    print('Success !! \nthe dataset is stored at {}'.format(path))
else:
    print('the path to the file is not correct')'''

get_latest_news(api_key,'general','us')
search_news(api_key,'chatgpt')
clean_data()

topic_model()
a=latest_wordcloud()
h=Search_wordcloud()
b=freq_latest_words()
g=freq_search_words()
e=sourse_plot()
c=freq_two_words()
d=freq_three_words()
#f=topic_model_plot()


#########################################################


app = dash.Dash(__name__)

###
#df_new = pd.read_csv('https://plotly.github.io/datasets/country_indicators.csv')
#available_indicators = df_new['Indicator Name'].unique()
###


#

## FF ##

# Create app layout
app.layout = html.Div(
    [
        dcc.Store(id="aggregate_data"),
        # empty Div to trigger javascript file for graph resizing
        html.Div(id="output-clientside"),
        html.Div(
            [
                
                html.Div(
                    [
                     html.Div(
                    # create empty div for align cente
                    className="one-third column",
                 ),
                        html.Div(
                            [
                                html.H4(
                                    "Topic modeling on News Dataset",
                                    style={"font-weight": "bold"},
                                ),
                                html.H5(
                                    "Extracting Topics from The News using LDA model for "+datetime.today().strftime('%Y-%m-%d'), style={"margin-top": "0px"}
                                ),
                
                            ]
                        )
                    ],
                    className="three column",
                    id="title",
                ),
                html.Div(
                    # create empty div for align cente
                    className="one-third column",
                ),
            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
        ),
        html.Div(
            [
                html.Div(
                    [

                        html.H6("Filter ", style={"margin-top": "0","font-weight": "bold","text-align": "center"}),

                        html.P(
                            "Choose a country and category :",
                            className="control_label",style={"text-align": "justify"}
                        ),
                        html.P(),
                        
                        html.P("Country", className="control_label",style={"text-align": "center","font-weight":"bold"}),
                        ots_behaviour,
                        

                        html.P("Category", className="control_label",style={"text-align": "center","font-weight":"bold"}),
                        radio_food_behaviour,

                        html.Button('Apply',id='button-2'),


                        html.H6("Search a Topic", style={"margin-top": "0","font-weight": "bold","text-align": "center"}),
                        #html.P("Put text :", className="control_label",style={"text-align": "center","font-weight":"bold"}),

                        html.Div(dcc.Input(placeholder='type something',id='input-on-submit', type='text')),
                        html.Button('Search', id='submit-val', n_clicks=0),
                         html.P(
                            "Note: The search process will take a minute.",
                            className="control_label",style={"text-align": "justify"}
                        ),

                    ],
                    className="pretty_container four columns",
                    id="cross-filter-options",
                    style={"text-align": "justify"},


                ),

                html.Div(
                    [
                        ##################################
                        html.Div(
                            [
                            html.H6("News Word Cloud", style={"margin-top": "0","font-weight": "bold","text-align": "center"}),

                            dcc.Graph(id="plot1",figure=a)],
                            #id="countGraphContainer",
                            className="pretty_container",
                        ),
                    ],
                    id="right-column",
                    className="eight columns",
                ),
            ],
            className="row flex-display",
        ),

        html.Div(
            [
                html.H6("Most Frequent Words", style={"margin-top": "0","font-weight": "bold","text-align": "center"}),
                #html.P("Similarly to nutrition, the health status also varies from country to country. The bar chart below shows the differences between the countries in terms of the following variables: prevalence of obesity in the adult population in % (Obesity), prevalence of diabetes in the adult population in % (Diabetes Prevalence), cardiovascular death rate per 100,000 population (Cardiovascular Death Rate), average life expectancy in years (Life Expectancy) and the expenditure of the government on the country's health system in % of the respective GDP (Health Expenditure).", className="control_label",style={"text-align": "justify"}),
                
                html.Div([dcc.Graph(id="plot2", figure=b)],className="pretty_container twelve columns"),
            ],
            className="row pretty_container",
        ),


        html.Div(
            [
                html.H6("Most Frequent Two Words", style={"margin-top": "0","font-weight": "bold","text-align": "center"}),
                #html.P("Similarly to nutrition, the health status also varies from country to country. The bar chart below shows the differences between the countries in terms of the following variables: prevalence of obesity in the adult population in % (Obesity), prevalence of diabetes in the adult population in % (Diabetes Prevalence), cardiovascular death rate per 100,000 population (Cardiovascular Death Rate), average life expectancy in years (Life Expectancy) and the expenditure of the government on the country's health system in % of the respective GDP (Health Expenditure).", className="control_label",style={"text-align": "justify"}),
                
                html.Div([dcc.Graph(id="plot3", figure=c)],className="pretty_container twelve columns"),
            ],
            className="row pretty_container",
        ),


        html.Div(
            [
                html.H6("Most Frequent Three Words", style={"margin-top": "0","font-weight": "bold","text-align": "center"}),
                #html.P("Similarly to nutrition, the health status also varies from country to country. The bar chart below shows the differences between the countries in terms of the following variables: prevalence of obesity in the adult population in % (Obesity), prevalence of diabetes in the adult population in % (Diabetes Prevalence), cardiovascular death rate per 100,000 population (Cardiovascular Death Rate), average life expectancy in years (Life Expectancy) and the expenditure of the government on the country's health system in % of the respective GDP (Health Expenditure).", className="control_label",style={"text-align": "justify"}),
                
                html.Div([dcc.Graph(id="plot4", figure=d)],className="pretty_container twelve columns"),
            ],
            className="row pretty_container",
        ),

        


        html.Div(
            [   
                
                html.Div(
            [
                html.H6("Where The News come From ?", style={"margin-top": "0","font-weight": "bold","text-align": "center"}),
                #html.P("Similarly to nutrition, the health status also varies from country to country. The bar chart below shows the differences between the countries in terms of the following variables: prevalence of obesity in the adult population in % (Obesity), prevalence of diabetes in the adult population in % (Diabetes Prevalence), cardiovascular death rate per 100,000 population (Cardiovascular Death Rate), average life expectancy in years (Life Expectancy) and the expenditure of the government on the country's health system in % of the respective GDP (Health Expenditure).", className="control_label",style={"text-align": "justify"}),
                
                html.Div([dcc.Graph(id="source_plot", figure=e)],className="pretty_container twelve columns"),
            ],
            className="row pretty_container",
        ),

            ],
            className="row pretty_container",
        ),

        

        html.Div(
            [
                
                    html.H6("Topic Modeling", style={"margin-top": "0","font-weight": "bold","text-align": "center"}),
                    #html.P("Finally, k-means clustering is carried out. The criterion can be selected on the left side, whereby either the 18 food variables, the 5 health variables or all of them in combination may be chosen for the clustering. On the right side, the resulting clusters can then be compared with respect to a chosen variable.",className="control_label",style={"text-align": "justify"}),
                    html.Div([ 
                    #html.P("Select a clustering criterion", className="control_label",style={"text-align": "center","font-weight":"bold"}),
                    #radio_clust_behaviour,
                    
                    dcc.Graph(id="topic0",figure=plot_topic('l',0))
                    ],className="pretty_container sixish columns",),

                    html.Div([ 
                    
                    #html.P("Select a variable for cluster comparison", className="control_label",style={"text-align": "center","font-weight":"bold"}),
                    #ots_behaviour,
                    dcc.Graph(id="topic1",figure=plot_topic('l',1)),
                    ],className="pretty_container sixish columns",),


                    #####

                    html.Div([ 
                    #html.P("Select a clustering criterion", className="control_label",style={"text-align": "center","font-weight":"bold"}),
                    #radio_clust_behaviour,
                    
                    dcc.Graph(id="topic2",figure=plot_topic('l',2))
                    ],className="pretty_container sixish columns",),

                    #######

                    html.Div([ 
                    
                    #html.P("Select a variable for cluster comparison", className="control_label",style={"text-align": "center","font-weight":"bold"}),
                    #ots_behaviour,
                    dcc.Graph(id="topic3",figure=plot_topic('l',3)),
                    ],className="pretty_container sixish columns",),


                    
                
            ],
            className="row pretty_container",
        ),
        



        
    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)

colors = ['#0d0887', '#46039f', '#7201a8', '#9c179e', '#bd3786', '#d8576b', '#ed7953', '#fb9f3a', '#fdca26', '#f0f921']
colors2 = ['#fdca26', '#ed7953', '#bd3786', '#7201a8','#0d0887']











###########################




@app.callback(
    Output('plot1', 'figure'),
    Output('plot2', 'figure'),
    Output('plot3', 'figure'),
    Output('plot4', 'figure'),
    Output('source_plot', 'figure'),
    Output('topic0', 'figure'),
    Output('topic1', 'figure'),
    Output('topic2', 'figure'),
    Output('topic3', 'figure'),
    Output('input-on-submit', 'value'),
    Input('submit-val', 'n_clicks'),
    State('input-on-submit', 'value'))
def update_plots(n_clicks,value):
    if value != 'type something':
        search_news(api_key,term=value)
        clean_data()
        topic_model()
        f1=Search_wordcloud()
        f2=freq_search_words(terms = 30)
        f3=search_freq_two_words()
        f4=search_freq_three_words()
        f5=search_sourse_plot()
        f6=plot_topic('s',0)
        f7=plot_topic('s',1)
        f8=plot_topic('s',2)
        f9=plot_topic('s',3)
        return f1,f2,f3,f4,f5,f6,f7,f8,f9,value




@app.callback(
    Output('plot1', 'figure',allow_duplicate=True),
    Output('plot2', 'figure',allow_duplicate=True),
    Output('plot3', 'figure',allow_duplicate=True),
    Output('plot4', 'figure',allow_duplicate=True),
    Output('source_plot', 'figure',allow_duplicate=True),
    Output('topic0', 'figure',allow_duplicate=True),
    Output('topic1', 'figure',allow_duplicate=True),
    Output('topic2', 'figure',allow_duplicate=True),
    Output('topic3', 'figure',allow_duplicate=True),
    Output('box_dd', 'value'),
    Output('nutrition_types', 'value'),
    Input('button-2', 'n_clicks'),
    State('box_dd', 'value'),
    State('nutrition_types', 'value'),prevent_initial_call=True)
def display(nb,value1,value2):
    value1=value1.lower()
    get_latest_news(api_key,value2,value1)
    clean_data()
    topic_model()
    f1=latest_wordcloud()
    f2=freq_latest_words(terms = 30)
    f3=freq_two_words()
    f4=freq_three_words()
    f5=sourse_plot()
    f6=plot_topic('l',0)
    f7=plot_topic('l',1)
    f8=plot_topic('l',2)
    f9=plot_topic('l',3)

    return f1,f2,f3,f4,f5,f6,f7,f8,f9,value1,value2
    









####










   

server = app.server

if __name__ == '__main__':
    app.run_server(debug=False)
