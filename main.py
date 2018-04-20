#!/usr/bin/env python

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import Lasso
from newsapi import NewsApiClient
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


newsapi = NewsApiClient(api_key='cb202441f4bd4a74aa5e5326dc4eb51f')
current_date = datetime.datetime.now().isoformat()
two_years_ago = (datetime.datetime.now() - datetime.timedelta(days=2*365)).isoformat()
print("current", current_date)
print("2", two_years_ago)

df = pd.read_csv('constituents_csv.csv')
companies = df['Name']
headline_dict = {}

for company in companies:
	newsdata = newsapi.get_everything(q=company,
                                      sources='bbc-news,the-verge',
                                      domains='bbc.co.uk,techcrunch.com',
                                      # from_parameter=two_years_ago,
                                      # to=current_date,
                                      language='en',
                                      sort_by='relevancy',
                                      page=2)
	
	articles = newsdata['articles']
	headlines = []
	for article in articles:
		headlines.append(article['title'])
	headline_dict[company] = headlines


print(headline_dict)