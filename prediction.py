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
end_date = datetime.datetime.now().isoformat()
start_date = (datetime.datetime.now() - datetime.timedelta(days=2*365)).isoformat()
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
		df_current = initialize_dataframe(company_symb[company], start_date, end_date)

		# Query news articles
		dict_current = query_news_articles(company, start_date, end_date, sources)

		# iterate through dates
		for date in dict_current.keys():
			# when you enter seniment into the dataframe, use the before date not after
			average_sentiment_dict = calculate_sentiment(dict_current[date])

			# Plug this into df current

		# Regression

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
	df = df = pd.DataFrame(columns=['neg','neu','pos','compound'])
	for sentence in sentence_arr:
		sentiment = sia.polarity_scores(sentence)
		df_sentiment = pd.DataFrame([sentiment], columns=sentiment.keys())
		df = df.append(df_sentiment)
	
	avg_sentiment = dict(df.mean())	
	return avg_sentiment 


# return dataframe with price_t and price_t-1
def initialize_dataframe(ticker, start_date, end_date):
	""" Initializes a data frame for a certain ticker
	Params:
		ticker (String): Stock ticker to be analyzed
		start_date (String): Start date in format of "2001-12-31"
		end_date (String): End date in format of "2001-12-31"
	Returns:
		dataframe (pd.Dataframe): Dataframe with index 'Date' ,columns 'X_t' and 'X_t-1'
	"""

	# Convert start_date and end_date into a string
	start_date_day, start_date_hours = start_date.split('T')
	start_date_arr = start_date_day.split('-')
	start_date_string = datetime.date(int(start_date_arr[0]), int(start_date_arr[1]), int(start_date_arr[2]))
	end_date_day, end_date_hours = end_date.split('T')
	end_date_arr = end_date_day.split('-')
	end_date_string = datetime.date(int(end_date_arr[0]), int(end_date_arr[1]), int(end_date_arr[2]))

	# Query quandl for data and make dataframe
	dataframe = quandl.get('EOD/'+ticker, start_date=start_date_string, end_date=end_date_string)['Adj_Close']
	dataframe = pd.Series.to_frame(dataframe)

	# Make columns of dataframe
	dataframe.columns = ['X_t']
	dataframe['X_t-1'] = dataframe['X_t'].shift(1)

	# Remove the first data point because of shift
	dataframe = dataframe.iloc[1:]
	return dataframe


# query the news articles for the entire time frame and split by dates 
# returns dictionaries of headlines and dates 
# parse the headlines by date
# key is date and value is headlines 
def query_news_articles(company, start_date, end_date, sources):
	""" Queries news article for a certain time frame and split it by dates
		Note that we split the date differently based on before and after 4pm EST.
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

initialize_dataframe('MSFT', start_date, end_date)

