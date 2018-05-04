#!/usr/bin/env python

from newsapi import NewsApiClient
from pandas_datareader import data
import datetime
import numpy as np
import pandas as pd
import quandl
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Params
newsapi = NewsApiClient(api_key='cb202441f4bd4a74aa5e5326dc4eb51f')
quandl.ApiConfig.api_key = "REidDLJ8C4pPMRMZGLnA"
start_date = datetime.datetime.now().isoformat()
end_date = (datetime.datetime.now() - datetime.timedelta(days=2*365)).isoformat()
sources = 'bbc-news,the-verge'
# domain

def main():
	# retrieve s&p 500 companies 
	df = pd.read_csv('constituents_csv.csv')
	companies = df['Name']
	company_symb = {}


	# Iterate through companies
	for company in companies:

		# get company ticker
		company_symb[company] = df[df['Name'] == company]['Symbol']

		# initialize the dataframe
		df_current = initialzie_dataframe(company_symb[company], start_date, end_date)

		# Query news articles
		dict_current = query_news_articles(company, start_date, end_date, sources)

		# iterate through dates
		for date in dict_current.keys():
			# when you enter seniment into the dataframe, use the before date not after
			average_sentiment_dict = calculate_sentiment(dict_current[date])

			# Plug this into df current

		# Regression
		







# calculate the sentiment of the headline or description
# calculate average sentiment 
def calculate_sentiment(sentence_arr):
	""" Returns the average sentiment of the array
	Params:
		sentence_arr(Array): Array of setences that we have to calculate
		the sentiment of.
	Returns:
		sentiment (dictionary): Takes the average of all sentences
		format of score is {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
	"""
	sia = SentimentIntensityAnalyzer()
	# this is for once sentence 
	# sentiment = sia.polarity_scores(sentence)
	return sentiment 


# return dataframe with price_t and price_t-1
def initialize_dataframe(ticker, start_date, end_date):
	""" Initializes a data frame for a certain ticker
	Params:
		ticker (String): Stock ticker to be analyzed
		start_date (String): Start date in format of "2001-12-31"
		end_date (String): End date in format of "2001-12-31"
	Returns:
		dataframe (pd.Dataframe): Dataframe with columns date_t-1, X_t and X_t-1
	"""
	dataframe = ""
	return dataframe


# query the news articles for the entire time frame and split by dates 
# returns dictionaries of headlines and dates 
# parse the headlines by date
# key is date and value is headlines 
def query_news_articles(company, start_date, end_date, sources):
	""" Queries news article for a certain time frame and split it by dates
		Note that
	Params:
		company (String): Name of company
		start_date (String): Start date in format of "2001-12-31"
		end_date (String): End date in format of "2001-12-31"
		sources (Array of Strings): Array of different news sources
	Returns:
		company_dic (dictionary): keys are date, values are array of headlines
	"""
	company_dict = {}
	return company_dic

