#!/usr/bin/env python

from newsapi import NewsApiClient
from pandas_datareader import data
import datetime
import numpy as np
import pandas as pd
import quandl
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def calculate_stock_price_change(ticker, date):
	"""
	Calculated the percent change of the stock based on it's ticker and it's date. Note
	that we use the previous day or next day depending on the hour of release.

	Params:
	ticker: (String) company ticker
	date: (String) Format of 2018-04-13T00:46:59Z (UTC format)

	Returns:
		percent_change (float) Percent change calculated
	"""
	# get info in right format for stock price query 
    stock_after_date, stock_after_time = date.split('T')
	date_arr = stock_after_date.split('-')
	datetime_after = datetime.date(int(date_arr[0]), int(date_arr[1]), int(date_arr[2]))
    time_arr = stock_after_time[:-1].split(':')
    # stock market closes at 4:00 PM EST; if article published after 
    # 16:00:00+4:00:00 = 20:00:00 UTC counts as next "day"
    if int(time_arr[0]) >= 20:
    	datetime_after += datetime.timedelta(1)
	stock_before_date = (datetime_after - datetime.timedelta(1)).isoformat()


	# get stock prices before and after news article 
	# output: Open  High  Low  Close Volume  Dividend  Split  Adj_Open  Adj_High  Adj_Low  Adj_Close  Adj_Volume
	stock_before = quandl.get('EOD/'+ticker, start_date=stock_before_date, end_date=stock_before_date)['Adj_Close'].values[0]
	stock_after = quandl.get('EOD/'+ticker, start_date=stock_after_date, end_date=stock_after_date)['Adj_Close'].values[0]

	percent_change = (stock_after - stock_before) / stock_before
	return percent_change


sia = SentimentIntensityAnalyzer()

# set api keys 
newsapi = NewsApiClient(api_key='cb202441f4bd4a74aa5e5326dc4eb51f')
quandl.ApiConfig.api_key = "REidDLJ8C4pPMRMZGLnA"


# get start and end date for news query (current date and two yeares prev)
current_date = datetime.datetime.now().isoformat()
two_years_ago = (datetime.datetime.now() - datetime.timedelta(days=2*365)).isoformat()
print("current", current_date)
print("2", two_years_ago)

# retrieve s&p 500 companies 
df = pd.read_csv('constituents_csv.csv')
companies = df['Name']
company_symb = {}
# key: company name, val: array of article dic from query
company_dict = {}

# for each company query top news articles with relating information 
for company in companies:
	company_symb[company] = df[df['Name'] == company]['Symbol']
	newsdata = newsapi.get_everything(q=company,
                                      sources='bbc-news,the-verge',
                                      domains='bbc.co.uk,techcrunch.com',
                                      # from_parameter=two_years_ago,
                                      # to=current_date,
                                      language='en',
                                      sort_by='relevancy',
                                      page=2)
	
	articles = newsdata['articles']
	article_dic = {}
	articles_arr = []
	for article in articles:
		headline = article['title']
		article_dic['headline'] = headline
		# format of score is {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
		article_dic['headline_score'] = sia.polarity_scores(headline)
		description = article['description']
		article_dic['description'] = description
		article_dic['description_score'] = sia.polarity_scores(description)
		# format of date is 2018-04-13T00:46:59Z (UTC format)
		article_dic['publishedAt'] = article['publishedAt'] 
		article_dic['source'] = article['source']['name']

		comp_symb = company_symb[company].replace(".", "_").replace("-", "_")


		article_dict['stock_price_change'] = calculate_stock_price_change(comp_symb,article_dic['publishedAt'])
		
		
		# append article dict to array 
		articles_arr.append(article_dic)

		# clear dict 
		article_dic = {}
	
	company_dict[company] = articles_arr

	# for testing purposes 
	if company == 'Apple Inc.':
		for x in company_dict['Apple Inc.']:
			print(x)
		print(company_symb[company])
