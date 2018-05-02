#!/usr/bin/env python

from newsapi import NewsApiClient
from pandas_datareader import data
import datetime
import numpy as np
import pandas as pd
import quandl
from nltk.sentiment.vader import SentimentIntensityAnalyzer


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
		# format of date is 2018-04-13T00:46:59Z
		article_dic['publishedAt'] = article['publishedAt'] 
		article_dic['source'] = article['source']['name']

		
		# get info in right format for stock price query 
		comp_symb = company_symb[company].replace(".", "_").replace("-", "_")
		stock_after_date = article_dic['publishedAt'][:10]
		date_arr = stock_before_date.split('-')
		datetime_after = datetime.date(int(date_arr[0]), int(date_arr[1]), int(date_arr[2]))
		stock_before_date = (datetime_after - datetime.timedelta(1)).isoformat()


		# get stock prices before and after news article 
		# output: Open  High  Low  Close Volume  Dividend  Split  Adj_Open  Adj_High  Adj_Low  Adj_Close  Adj_Volume
		stock_before = quandl.get('EOD/'+comp_symb, start_date=stock_before_date, end_date=stock_before_date)['Close'].values[0]
		stock_after = quandl.get('EOD/'+comp_symb, start_date=stock_after_date, end_date=stock_after_date)['Close'].values[0]
		article_dic['stockPriceChange'] = stock_after-stock_before
		
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
