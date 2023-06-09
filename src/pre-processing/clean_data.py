#libraries
import os
import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
import nltk
from nltk.stem.snowball import SnowballStemmer
from pathlib import Path
nltk.download('wordnet')
nltk.download('omw-1.4')

#remove old files
if os.path.exists(str(Path(__file__).resolve().parent.parent.parent)+'/data/processed/processed_search_news_dataset.csv'):
    os.remove(str(Path(__file__).resolve().parent.parent.parent)+'/data/processed/processed_search_news_dataset.csv')

if os.path.exists(str(Path(__file__).resolve().parent.parent.parent)+'/data/processed/processed_latest_news_dataset.csv'):
    os.remove(str(Path(__file__).resolve().parent.parent.parent)+'/data/processed/processed_latest_news_dataset.csv')


#read the raw dataset 

if os.path.exists(str(Path(__file__).resolve().parent.parent.parent)+'/data/raw/raw_search_news_dataset.csv'):
    search_news_raw=pd.read_csv(str(Path(__file__).resolve().parent.parent.parent)+'/data/raw/raw_search_news_dataset.csv')
else:
    print('The file {} doesnt exist\nRun the scripts in /src/data to get the dataset'.format(str(Path(__file__).resolve().parent.parent.parent)+'/data/raw/raw_search_news_dataset.csv'))

if os.path.exists(str(Path(__file__).resolve().parent.parent.parent)+'/data/raw/raw_latest_news_dataset.csv'):
    latest_news_raw=pd.read_csv(str(Path(__file__).resolve().parent.parent.parent)+'/data/raw/raw_latest_news_dataset.csv')
else:
    print('The file {} doesnt exist\nRun the scripts in /src/data to get the dataset'.format(str(Path(__file__).resolve().parent.parent.parent)+'/data/raw/raw_latest_news_dataset.csv'))


#remove unwanted columns
'''
latest_news_raw=latest_news_raw.iloc[:, [ False, True, True,True,True, False,True,True, False,True]]
search_news_raw=search_news_raw.iloc[:, [ False, True, True,True,True, False,True,True, False,True]]
'''
latest_news_raw=latest_news_raw.rename(columns={'publishedAt':'Date','source.name':'Source'})
search_news_raw=search_news_raw.rename(columns={'publishedAt':'Date','source.name':'Source'})

#fix date column
latest_news_raw['Date']=pd.to_datetime(latest_news_raw['Date'], format='%Y-%m-%dT%H:%M:%SZ')
search_news_raw['Date']=pd.to_datetime(search_news_raw['Date'], format='%Y-%m-%dT%H:%M:%SZ')

# remove unwanted characters, numbers and symbols
latest_news_raw['content']=latest_news_raw['content'].str.replace(" [^a-zA-Z#] |\n|\'s|[:@#$&=^*!?~-]|[()]|[/]|\[|\]|\"|\'|‘|—|“", " ")
search_news_raw['content']=search_news_raw['content'].str.replace(" [^a-zA-Z#] |\n|\'s|[:@#$&=^*!?~-]|[()]|[/]|\[|\]|\"|\'|‘|—|“", " ")

#os.remove(str(Path(__file__).resolve().parent.parent.parent)+'/data/raw/raw_latest_news_dataset.csv')
#os.remove(str(Path(__file__).resolve().parent.parent.parent)+'/data/raw/raw_latest_news_dataset.csv')


path=str(Path(__file__).resolve().parent.parent.parent)+'/data/processed/processed_latest_news_dataset.csv'
latest_news_raw.to_csv(path)
if os.path.exists(path):
    print('the dataset is cleaned and stored at {}'.format(path))
else:
    print('the path to the file is not correct')

path=str(Path(__file__).resolve().parent.parent.parent)+'/data/processed/processed_search_news_dataset.csv'
search_news_raw.to_csv(path)
if os.path.exists(path):
    print('Success !! \nthe dataset is cleaned and stored at {}'.format(path))
else:
    print('the path to the file is not correct')
