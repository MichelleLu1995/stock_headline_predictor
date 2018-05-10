from flask import Flask, render_template, request
from newsapi import NewsApiClient
from nytimesarticle import articleAPI
import pandas_datareader.data as web
import datetime
import dateutil
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quandl
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pandas.plotting import autocorrelation_plot
from pandas.tools.plotting import lag_plot
from statsmodels.tsa.ar_model import AR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
app.config["DEBUG"] = True

# Params
# extra API keys, comment out as necessary
#api = articleAPI("b23351c6f9314694bfe4f4929a2b72c5") 
#api = articleAPI("787bd4db8e704bbf9cebe8b7941827e0") 
#api = articleAPI("f8b402f42ed14b249fd5accc95a050dd") 
#api = articleAPI("c91a676aeaef40fd844409c8b0bef485")
#api = articleAPI("c43133d654134109868299ff505e7c55")
#api = articleAPI("eb427ebc2336423ead4d350cfa4e900b")
#api = articleAPI("b538de93f1a9459da22b150d7b53cb6f")
api = articleAPI("88f587ed149d4478b4490168d61ed9dc")

quandl.ApiConfig.api_key = "2S7d7eeL5VZrLup9pKg5"
end_date = (datetime.datetime.now() - datetime.timedelta(days=3)).isoformat()
start_date = (datetime.datetime.now() - datetime.timedelta(days=2*365)).isoformat()
left_sources = 'The New York Times'
right_sources = 'Fox News'
center_sources = 'Reuters AP The Wall Street Journal'
all_sources = left_sources + ' ' + right_sources + ' ' + center_sources
replace_list = ['Corp', 'Inc.', 'Inc', '.com', 'plc', ',', 'Co.']

# get companies
df = pd.read_csv('constituents_csv.csv')
companies = df['Name'].values
companies = companies.tolist()

@app.route('/')
def main():
	return render_template("main.html", company_arr=companies)

@app.route('/company', methods=['GET', 'POST'])
def searchCompany():
	company = request.form['company']
	actual = 30.2
	predict = 44
	if predict > actual:
		suggestion = "Buy"
	else:
		suggestion = "Sell"
	# createRegModel("Amazon.com Inc.")
	legend = 'Predicted'
	labels = ["January", "February", "March", "April", "May", "June", "July", "August"]
	values = [10, 9, 8, 7, 6, 4, 7, 8]
	labels_bar = [1,2,3,4,5,6]
	values_bar = [123,214,556,343,134,234]
	return render_template("search.html", company=company, actual=actual, predict=predict, suggestion=suggestion, legend=legend, labels=labels, values=values, labels_bar=labels_bar, values_bar=values_bar)

def createModel(company):
	# get company ticker
	ticker = df[df['Name'] == company]['Symbol'].values[0]
	# get rid of suffixes from company name
	for word in replace_list:
		company = company.replace(word, '')

	# initialize the dataframe
	lag = get_lag_period(ticker, start_date, end_date)
	df_current = initialize_dataframe(ticker, start_date, end_date, lag=lag)
	
	# Add sentiment columns in dataframe
	df_current['Pos_t-1'] = 0
	df_current['Neu_t-1'] = 0
	df_current['Neg_t-1'] = 0

	# Query news articles
	trading_dates = df_current.index
	dict_current = query_news_articles(company, start_date, end_date, trading_dates, sources=all_sources)

	# iterate through dates
	for date in dict_current.keys():
		# when you enter seniment into the dataframe, use the before date not after
		average_sentiment_dict = calculate_sentiment(dict_current[date])

		# Plug this into df current
		df_current.at[date,'Pos_t-1'] = average_sentiment_dict['pos']
		df_current.at[date,'Neu_t-1'] = average_sentiment_dict['neu']
		df_current.at[date,'Neg_t-1'] = average_sentiment_dict['neg']
			
		# df_current.to_csv('./data/'+ticker+'.csv')
		#df_current = df_current.reset_index()

		# Try normal linear regression using lag period lags
	lag = get_lag_period(ticker, '2018-02-07', end_date)
	print('Number of lags:',lag)
	df_test = initialize_dataframe(ticker, '2018-02-07', end_date, lag)
	df_test.head()

	Y = df_test['X_t']
	X = df_test.loc[:, df_test.columns !='X_t']
	X_train, X_test = X[:int(len(X)*(0.8))], X[int(len(X)*(0.8)):]
	Y_train, Y_test = Y[:int(len(Y)*(0.8))], Y[int(len(Y)*(0.8)):]

	LinReg = LinearRegression(normalize=True)
	LinReg.fit(X_train,Y_train)
	Y_pred = LinReg.predict(X_test)
	print("R^2 Value: %.2f" %LinReg.score(X_test, Y_test))
	print("Mean squared error: %.2f"
		  % mean_squared_error(Y_test, Y_pred))
	print("Coefficients for the Regression are: " ,LinReg.coef_)

	Y_plot = Y_test.copy(deep=True)
	Y_plot = Y_plot.to_frame()
	Y_plot['Predicted'] = pd.Series(Y_pred, index=Y_plot.index)
	Y_plot.columns = ['Actual', 'Predicted']
	# print(Y_plot.head())

	# plt.figure(figsize=(10,10))
	# plt.plot(Y_plot.index, Y_plot['Actual'], label='Actual')
	# plt.plot(Y_plot.index, Y_plot['Predicted'],label='Predicted')
	# plt.legend()

	# plt.show()
	return Y_plot.index, Y_plot

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
	start_date = int(start_date.replace("-","").split('T')[0])
	ending_date = (dateutil.parser.parse(end_date)-datetime.timedelta(days=1)).isoformat()
	end_date = int(ending_date.replace("-","").split('T')[0])
	newsdata = api.search(q=company, begin_date = start_date,
						   end_date = end_date,
						  fq='headline:('+company+ ') OR body:('+company+') AND source:(' + sources + ')',
						  page = 0,
						  facet_filter = True)
							   

	#print(newsdata) # newsdata is full HTTP response
	number_of_hits = newsdata['response']['meta']['hits']
	number_of_pages = (number_of_hits // 10) + 1
	
	time.sleep(1)
	# page through results and add headlines to companY_dict
	for i in range(0, min(number_of_pages,200)):
		# print('page', i)
		newsdata = api.search(q=company, begin_date = start_date,
						   end_date = end_date,
						  fq='headline:('+company+ ') OR body:('+company+') AND source:(' + sources + ')',
						  page = i,
						  facet_filter = True)
		articles = newsdata['response']['docs']
		for article in articles:
			relevance = article['score']
			if relevance >= 0.005: 
				headline = article['headline']['main']
				blurb = article['snippet']
				# print(article['pub_date'], '\t', article['headline']['main'])
			
				# description = article['description']
				# format of date is 2018-04-13T00:46:59Z (UTC format)
				publish_date = article['pub_date'] 
				# print(publish_date)
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
					trading_datetime += datetime.timedelta(days=1)
				
				# if given trading_date invalid (ie if article published on Friday 
				# after market close, Saturday, or Sunday before 4 pm est) push trading_date
				# to the following Monday (ie first valid trading_date)
				while trading_datetime not in trading_dates:
					trading_datetime += datetime.timedelta(1)
				company_dict[trading_datetime].append(headline)
				# company_dict[trading_datetime].append(blurb) include 'snippet' in sentiment analysis
		time.sleep(1)
		
	return company_dict

def get_lag_period(ticker, start_date, end_date):
	""" Finds the optimal lab period given a ticker and the start date and end date
	*** Eventually integrate this into initialize dataframe ***
	Params:
		ticker (String): Stock ticker to be analyzed
		start_date (String): Start date in format of "2001-12-31"
		end_date (String): End date in format of "2001-12-31"
	Returns:
		lag (int): Number of lag periods
	"""
	# Get the data in a dataframe
	dataframe = web.DataReader(ticker, 'morningstar', start_date, end_date)['Close']
	dataframe = pd.Series.to_frame(dataframe)
	dataframe.reset_index(level=0, drop=True, inplace=True)
	dataframe.columns = ['X_t']

	# Fit the model and find optimal lab
	X = dataframe['X_t']
	train, test = X[:int(len(X)*(0.8))], X[int(len(X)*(0.8)):]
	model = AR(train)
	model_fit = model.fit()
	lag = model_fit.k_ar
	
	return lag
 
 # return dataframe with price_t and price_t-1
def initialize_dataframe(ticker, start_date, end_date, lag=1):
	""" Initializes a data frame for a certain ticker
	Params:
		ticker (String): Stock ticker to be analyzed
		start_date (String): Start date in format of "2001-12-31"
		end_date (String): End date in format of "2001-12-31"
		lag (int): Number of lag periods
	Returns:
		dataframe (pd.Dataframe): Dataframe with index 'Date' ,columns 'X_t' and 'X_t-1'
	"""

	# Query quandl for data and make dataframe
	#dataframe = quandl.get('EOD/'+ticker, start_date=start_date, end_date=end_date)['Adj_Close']
	dataframe = web.DataReader(ticker, 'morningstar', 
							   start_date, end_date)['Close']
	
	dataframe = pd.Series.to_frame(dataframe)
	dataframe.reset_index(level=0, drop=True, inplace=True)
	
	# Make columns of dataframe
	dataframe.columns = ['X_t']
	for i in range(lag):
		dataframe['X_t-' + str(i+1)] = dataframe['X_t'].shift(i+1)

	# Remove the first data point because of shift
	dataframe = dataframe.iloc[i+1:]

	return dataframe


if __name__ == "__main__":
  app.run(debug=True)

# def test():
# 	createModel("Amazon.com Inc.")