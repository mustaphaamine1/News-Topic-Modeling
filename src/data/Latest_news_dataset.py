# -*- coding: utf-8 -*-
import click
'''

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

    main()'''

#

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

if os.path.exists(str(Path(__file__).resolve().parent.parent.parent)+'/data/raw/raw_latest_news_dataset.csv'):
    os.remove(str(Path(__file__).resolve().parent.parent.parent)+'/data/raw/raw_latest_news_dataset.csv')


#you need to create an account in NewsAPI website to get the api key 

api_key=str(input('Enter the api key from NewsAPI website:\t'))
newsapi = NewsApiClient(api_key=api_key)


#the function to get the top headlines
def top_headlines(num_cat):
    dict1={1:'business',2:'entertainment',3:'general',4:'health',5:'science',6:'sports',7:'technology'}
    cat=dict1[num_cat]
    top_headlines=newsapi.get_top_headlines(category=cat)
    top_headlines=pd.json_normalize(top_headlines['articles']) 
    newdf=top_headlines
    for k,v in zip(newdf.index,newdf['url']):
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
    return newdf

#input the parameters for the newsapi
print('The category you want to get headlines for:\n Possible options:\n1 --> business\n2 --> entertainment\n3 --> general\n4 --> health\n5 --> science\n6 --> sports\n7 --> technology')
num_cat=int(input())
s=top_headlines(num_cat)


path=str(Path(__file__).resolve().parent.parent.parent)+'/data/raw/raw_latest_news_dataset.csv'
s.to_csv(path)
if os.path.exists(path):
    print('Success !! \nthe dataset is stored at {}'.format(path))
else:
    print('the path to the file is not correct')