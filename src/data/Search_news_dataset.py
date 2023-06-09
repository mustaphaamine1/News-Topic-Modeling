# -*- coding: utf-8 -*-
'''
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
'''
#Search for a particular topic 
#the news api client
from newsapi.newsapi_client import NewsApiClient
from pathlib import Path


#libraries
from pandas.io.json import json_normalize
import pandas as pd
from bs4 import BeautifulSoup
from wordcloud import WordCloud
from newspaper import Article
import datetime
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

if os.path.exists(str(Path(__file__).resolve().parent.parent.parent)+'/data/raw/raw_search_news_dataset.csv'):
    os.remove(str(Path(__file__).resolve().parent.parent.parent)+'/data/raw/raw_search_news_dataset.csv')


#you need to create an account in NewsAPI website to get the api key 

api_key=str(input('Enter the api key from NewsAPI website:\t'))
newsapi = NewsApiClient(api_key=api_key)

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

#the query parameter which is the search term for the get_everything function.
def func(query):
    newsd={}
    newsdf=pd.DataFrame()
    #for (from_dt,to_dt) in zip(from_list,to_list):  for the past 30 days
    for (from_dt,to_dt) in zip(from_list[:1],to_list[:1]):           
        all_articles = newsapi.get_everything(q=query,language='en',sort_by='relevancy',from_param=from_dt,to=to_dt)          
        d=pd.json_normalize(all_articles['articles'])      
        '''newdf=d[["url","description","publishedAt","source.name","author"]]
        dic=newdf.set_index(["source.name","publishedAt","author"])["url"].to_dict()'''
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
    return d


#input the parameters for the newsapi
print('Enter the topic to seach :\n')
keyword=input()
y=func(keyword)

path=str(Path(__file__).resolve().parent.parent.parent)+'/data/raw/raw_search_news_dataset.csv'
y.to_csv(path)
if os.path.exists(path):
    print('Success !! \nthe dataset is stored at {}'.format(path))
else:
    print('the path to the file is not correct')
