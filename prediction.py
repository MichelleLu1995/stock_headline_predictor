#!/usr/bin/env python

from newsapi import NewsApiClient
from pandas_datareader import data
import datetime
import numpy as np
import pandas as pd
import quandl
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# set api keys 
newsapi = NewsApiClient(api_key='cb202441f4bd4a74aa5e5326dc4eb51f')
quandl.ApiConfig.api_key = "REidDLJ8C4pPMRMZGLnA"

# retrieve s&p 500 companies 
df = pd.read_csv('constituents_csv.csv')
companies = df['Name']
company_symb = {}
# for each company query top news articles with relating information 
for company in companies:
	company_symb[company] = df[df['Name'] == company]['Symbol']


# calculate the sentiment of the headline or description
# calculate average sentiment 
def calculate_sentiment(sentence_arr):
	sia = SentimentIntensityAnalyzer()
	# this is for once sentence 
	# sentiment = sia.polarity_scores(sentence)
	return sentiment 


# return dataframe with price_t and price_t-1
def initialize_dataframe(ticker):
	dataframe = ""
	return dataframe


# query the news articles for the entire time frame and split by dates 
# returns dictionaries of headlines and dates 
# parse the headlines by date
# key is date and value is headlines 
def query_news_articles(company):
	company_dict = {}
	return company_dic