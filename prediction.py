#!/usr/bin/env python

from newsapi import NewsApiClient
from nytimesarticle import articleAPI
import pandas_datareader.data as web
import datetime
import numpy as np
import pandas as pd
import quandl
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Params
# newsapi = NewsApiClient(api_key='cb202441f4bd4a74aa5e5326dc4eb51f')
# extra newsapi keys: 188a7490bd6642efb757f93cddd00d2b, 6b33fffeb4cb463f9908694d4be3a532
api = articleAPI("c91a676aeaef40fd844409c8b0bef485")
quandl.ApiConfig.api_key = "2S7d7eeL5VZrLup9pKg5"
end_date = datetime.datetime.now().isoformat()
start_date = (datetime.datetime.now() - datetime.timedelta(days=2*365)).isoformat()
left_sources = 'the-new-york-times,the-washington-post'
right_sources = 'fox-news,breitbart-news'
center_sources = 'bbc-news,associated-press,the-wall-street-journal,usa-today'
all_sources = left_sources + ',' + right_sources + ',' + center_sources
# domain

def main():
    # retrieve s&p 500 companies 
    df = pd.read_csv('constituents_csv.csv')
    companies = df['Name']
    company_symb = {}

    # Iterate through companies
    # for company in companies:
    for i in range(0,1):

        company = companies.loc[i]
        # get company ticker
        company_symb[company] = df[df['Name'] == company]['Symbol']
        ticker = company_symb[company].values[0]

        try:
            # initialize the dataframe
            df_current = initialize_dataframe(ticker, start_date, end_date)
            
    
            # Add sentiment columns in dataframe
            df_current['Pos_t-1'] = 0
            df_current['Neu_t-1'] = 0
            df_current['Neg_t-1'] = 0
    
            # Query news articles
            trading_dates = df_current.index
            dict_current = query_news_articles(company, start_date, end_date, trading_dates, sources=all_sources)
    
            print(dict_current)
            # iterate through dates
            for date in dict_current.keys():
                # when you enter seniment into the dataframe, use the before date not after
                average_sentiment_dict = calculate_sentiment(dict_current[date])
    
                # Plug this into df current
                df_current.at[date,'Pos_t-1'] = average_sentiment_dict['pos']
                df_current.at[date,'Neu_t-1'] = average_sentiment_dict['neu']
                df_current.at[date,'Neg_t-1'] = average_sentiment_dict['neg']
            print(i, ticker, 'success')
            df_current.to_csv('./data/'+ticker+'.csv')
        except:
            print(i, ticker, 'failed')
            pass




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

    # Query quandl for data and make dataframe
#    dataframe = quandl.get('EOD/'+ticker, start_date=start_date, end_date=end_date)['Adj_Close']
    start_date_arr = start_date[:10].split('-')
    end_date_arr = end_date[:10].split('-')
    
    start_date = datetime.date(int(start_date_arr[0]), int(start_date_arr[1]), int(start_date_arr[2]))
    end_date = datetime.date(int(end_date_arr[0]), int(end_date_arr[1]), int(end_date_arr[2]))
    dataframe = web.DataReader(ticker, 'morningstar', 
                               start_date, end_date)['Close']
    
    dataframe = pd.Series.to_frame(dataframe)
    dataframe.reset_index(level=0, drop=True, inplace=True)

    # Make columns of dataframe
    dataframe.columns = ['X_t']
    dataframe['X_t-1'] = dataframe['X_t'].shift(1)

    # Remove the first data point because of shift
    dataframe = dataframe.iloc[1:]

    return dataframe

# query the news articles for the entire time frame and split by dates 
# returns dictionary of key-value pairs where key is trading date in 
# format YYYY-MM-DD and value is array of all headlines affecting the stock
# price on key date
def query_news_articles(company, start_date, end_date, trading_dates, sources):
    """ Queries news article for a certain time frame and split it by dates
        Note that
    Params:
        company (String): Name of company
        start_date (String): Start date in format of "2001-12-31"
        end_date (String): End date in format of "2001-12-31"
         trading_dates (Array of Strings): Array of dates when the market was open
                 dates in format of "2001-12-31"
        sources (Array of Strings): Array of different news sources
    Returns:
        company_dic (dictionary): keys are date, values are array of headlines
    """
    company_dict = {k: [] for k in trading_dates.date}
    start_date = start_date.replace("-","")
    end_date = end_date.replace("-","")
    for page_num in range(1,11):
        newsdata = api.search(q=company,
                              fq = {'headline':company, 'source':['Reuters','AP', 'The New York Times']},
                              begin_date = start_date,
                              end_date = end_date, 
                              facet_filter = True,
                              page = page_num)
        print(newsdata)
        articles = newsdata['articles']
        for article in articles:
            headline = article['title']
            # description = article['description']
            # format of date is 2018-04-13T00:46:59Z (UTC format)
            publish_date = article['publishedAt'] 
            # adjust date for trading day
            publish_date, publish_time = publish_date.split('T')
            date_arr = publish_date.split('-')
            publish_datetime = datetime.date(int(date_arr[0]), int(date_arr[1]), int(date_arr[2]))
            time_arr = publish_time[:-1].split(':')
            # stock market closes at 4:00 PM EST; if article published after 
            # 16:00:00+4:00:00 = 20:00:00 UTC headline affects next trading day;
            # otherwise affects current trading day
            trading_datetime = publish_datetime
            if int(time_arr[0]) >= 20:
                trading_datetime += datetime.timedelta(1)
            
            # if given trading_date invalid (ie if article published on Friday 
            # after market close, Saturday, or Sunday before 4 pm est) push trading_date
            # to the following Monday (ie first valid trading_date)
            while trading_datetime not in trading_dates:
                trading_datetime += datetime.timedelta(1)
            company_dict[trading_datetime].append(headline)
    return company_dict


# query the news articles for the entire time frame and split by dates 
# returns dictionary of key-value pairs where key is trading date in 
# format YYYY-MM-DD and value is array of all headlines affecting the stock
# price on key date
# def query_news_articles(company, start_date, end_date, trading_dates, sources):
#     """ Queries news article for a certain time frame and split it by dates
#         Note that
#     Params:
#         company (String): Name of company
#         start_date (String): Start date in format of "2001-12-31"
#         end_date (String): End date in format of "2001-12-31"
#          trading_dates (Array of Strings): Array of dates when the market was open
#                  dates in format of "2001-12-31"
#         sources (Array of Strings): Array of different news sources
#     Returns:
#         company_dic (dictionary): keys are date, values are array of headlines
#     """
#     company_dict = {k: [] for k in trading_dates.date}
#     for page_num in range(1,11):
#         newsdata = newsapi.get_everything(q=company,
#                                           sources=sources,
#                                           from_param=start_date,
#                                           to=end_date,
#                                           language='en',
#                                           sort_by='relevancy',
#                                           page_size=100,
#                                           page=page_num)
    
#         articles = newsdata['articles']
#         for article in articles:
#             headline = article['title']
#             # description = article['description']
#             # format of date is 2018-04-13T00:46:59Z (UTC format)
#             publish_date = article['publishedAt'] 
#             # adjust date for trading day
#             publish_date, publish_time = publish_date.split('T')
#             date_arr = publish_date.split('-')
#             publish_datetime = datetime.date(int(date_arr[0]), int(date_arr[1]), int(date_arr[2]))
#             time_arr = publish_time[:-1].split(':')
#             # stock market closes at 4:00 PM EST; if article published after 
#             # 16:00:00+4:00:00 = 20:00:00 UTC headline affects next trading day;
#             # otherwise affects current trading day
#             trading_datetime = publish_datetime
#             if int(time_arr[0]) >= 20:
#                 trading_datetime += datetime.timedelta(1)
            
#             # if given trading_date invalid (ie if article published on Friday 
#             # after market close, Saturday, or Sunday before 4 pm est) push trading_date
#             # to the following Monday (ie first valid trading_date)
#             while trading_datetime not in trading_dates:
#                 trading_datetime += datetime.timedelta(1)
#             company_dict[trading_datetime].append(headline)
#     return company_dict