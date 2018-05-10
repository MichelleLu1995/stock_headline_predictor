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
api = articleAPI("f8b402f42ed14b249fd5accc95a050dd") 
#api = articleAPI("c91a676aeaef40fd844409c8b0bef485")
#api = articleAPI("c43133d654134109868299ff505e7c55")
#api = articleAPI("eb427ebc2336423ead4d350cfa4e900b")
# api = articleAPI("b538de93f1a9459da22b150d7b53cb6f")
# api = articleAPI("88f587ed149d4478b4490168d61ed9dc")

quandl.ApiConfig.api_key = "2S7d7eeL5VZrLup9pKg5"
end_date = (datetime.datetime.now() - datetime.timedelta(days=3)).isoformat()
start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).isoformat()
left_sources = 'The New York Times'
right_sources = 'Fox News'
center_sources = 'Reuters AP The Wall Street Journal'
all_sources = left_sources + ' ' + right_sources + ' ' + center_sources
replace_list = ['Corp', 'Inc.', 'Inc', '.com', 'plc', ',', 'Co.']

# get companies
df = pd.read_csv('constituents_csv.csv')
companies = ['Amazon.com', 'Facebook', 'Netflix', 'Google', 'Twitter', 'Apple', 'Twenty-First Century Fox']
ticker_dict = {'Amazon.com': 'AMZN', 'Facebook': 'FB', 'Netflix': 'NFLX', 'Google': 'GOOGL', 'Twitter': 'TWTR', 'Apple': 'AAPL', 'Twenty-First Century Fox':'FOXA'} 

# companies = df['Name'].values
# companies = companies.tolist()

@app.route('/')
def main():
    return render_template("main.html", company_arr=companies)

@app.route('/company', methods=['GET', 'POST'])
def searchCompany():
    # get company
    company = request.form['company']
    ticker = ticker_dict[company]
    prev_date = (datetime.datetime.now() - datetime.timedelta(days=3)).isoformat()
    curr_date = datetime.datetime.now().isoformat()

    # create model 
    plot_dict, predict = createModel(company)
    # print(str(curr_date)[:10])
    # get today's stock price
    curr_price = web.DataReader(ticker, 'morningstar', 
                               prev_date, curr_date)['Close'].iloc[-1]

    suggestion = {}
    if predict['ADL'] > curr_price:
        suggestion['ADL'] = "Buy"
    else:
        suggestion['ADL'] = "Sell"

    if predict['AR'] > curr_price:
        suggestion['AR'] = "Buy"
    else:
        suggestion['AR'] = "Sell"



    return render_template("search.html", company=company, plot_dict=plot_dict, curr_price=curr_price, predict=predict, suggestion=suggestion)

def createModel(company):
    # query the past day's news 
    # news_dict = query_news_articles(company, prev_date, curr_date, trading_dates, all_sources)
      
    # get company ticker
    # ticker = df[df['Name'] == company]['Symbol'].values[0]
    ticker = ticker_dict[company]

    # create model
    MSE_list_AR, MSE_list_ADL, intercept_AR, intercept_ADL, coef_AR, coef_ADL,\
               best_AR_train_index, best_AR_test_index, best_ADL_train_index, best_ADL_test_index = main_read_in_csv(ticker)
    # AR model
    model_AR = LinearRegression(normalize=True)
    model_AR.intercept_ = intercept_AR
    model_AR.coef_ = coef_AR

    # ADL model
    model_ADL = LinearRegression(normalize=True)
    model_ADL.intercept_ = intercept_ADL
    model_ADL.coef_ = coef_ADL

    # predict values for tomorrow  
    prediction = {}
    prediction['AR'] = predict_next_value(ticker, company, model_AR, is_ADL=False)
    prediction['ADL'] = predict_next_value(ticker, company,model_ADL, is_ADL=True)

    plot_AR = plot_AR_model(ticker, best_ADL_train_index, best_ADL_test_index, best_AR_train_index, best_AR_test_index)
    plot_ADL = plot_ADL_model(ticker, best_ADL_train_index, best_ADL_test_index, best_AR_train_index, best_AR_test_index)

    # plot dict
    plot_dict = {}
    plot_dict['MSE_labels'] = [1,2,3,4,5,6,7,8]
    plot_dict['MSE_AR_values'] = MSE_list_AR
    plot_dict['MSE_ADL_values'] = MSE_list_ADL
    plot_dict['comp_AR_label'] = plot_AR['x_val']
    plot_dict['comp_ADL_label'] = plot_ADL['x_val']
    plot_dict['comp_AR_actual'] = plot_AR['y_actual']
    plot_dict['comp_AR_predict'] = plot_AR['y_predict']
    plot_dict['comp_ADL_actual'] = plot_ADL['y_actual']
    plot_dict['comp_ADL_predict'] = plot_ADL['y_predict']

    return plot_dict, prediction

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

def query_news_articles(company, start_date, end_date, trading_dates, sources, pred=False):
    """ Queries news article for a certain time frame and split it by dates
        Note that
    Params:
        company (String): Name of company
        start_date (String): Start date in format of "2001-12-31"
        end_date (String): End date in format of "2001-12-31"
         trading_dates (Array of Strings): Array of dates when the market was open
                 dates in format of "2001-12-31"
        sources (Array of Strings): Array of different news sources
        pred (Boolean): Whether we are querying for predictions or not
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
    for i in range(0, min(number_of_pages,100)):
        print('page', i)
        newsdata = api.search(q=company, begin_date = start_date,
                           end_date = end_date,
                          fq='headline:('+company+ ') OR body:('+company+') AND source:(' + sources + ')',
                          page = i,
                          facet_filter = True)
        articles = newsdata['response']['docs']
        for article in articles:
            relevance = article['score']
            if relevance >= 0.005 or pred: 
                headline = article['headline']['main']
                blurb = article['snippet']
                # print(article['pub_date'], '\t', article['headline']['main'])
            
                # description = article['description']
                # format of date is 2018-04-13T00:46:59Z (UTC format)
                publish_date = article['pub_date'] 
                print(publish_date)
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


def  get_lag_period(X_train):
    """ Finds the optimal lab period given a ticker and the start date and end date
    Params:
        X_train (df): Training data used to calculate lag back
    Returns:
        lag (int): Number of lag periods
    """
    model = AR(X_train)
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

def main_read_in_csv(ticker):
    df_master = pd.read_csv('../data/'+ticker+'.csv',index_col='Date')
    
    ###### Per Fold do cross validation ######
    MSE_list_AR, coef_AR, intercept_AR, best_AR_train_index, best_AR_test_index = cross_validate_AR(df_master)   
    # print('Mean Squared Error List:', MSE_list_AR)
    # print('Best Coefficient',coef_AR)
    # print('Best Intercept', intercept_AR)
    MSE_list_ADL, coef_ADL, intercept_ADL, best_ADL_train_index, best_ADL_test_index = cross_validate_ADL(df_master)
    # print('Mean Squared Error List:', MSE_list_ADL)
    # print('Best Coefficients:',coef_ADL)
    # print('Best Intercept', intercept_ADL)
    
    return MSE_list_AR, MSE_list_ADL, intercept_AR, intercept_ADL, coef_AR, coef_ADL,\
           best_AR_train_index, best_AR_test_index, best_ADL_train_index, best_ADL_test_index

def cross_validate_AR(df):
    """ Runs the backtest with k fold cross validation for AR model
    Params:
        df (pd.Dataframe): Dataframe with X_t, lags, and sentiment score
    Returns:
        MSE_list (list): List of MSE per fold of cross-validation
        coef_list (list): List of coef per fold of cross-validation
        best_train_index (list): Indices of the train index with best MSE
        best_test_index (list): Indicies of test index with best MSE
    """
    tscv = TimeSeriesSplit(n_splits=8)
    X = df.loc[:, df.columns !='X_t']
    Y = df['X_t']
    
    MSE_list = []
    coef_list = []
        
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        Y_train, Y_test = Y.iloc[train_index], Y[test_index]
        
        # find optimal lag period
        lag = get_lag_period(Y_train)

        # filter the lags and sentiment values
        X_train = X_train.iloc[:,range(lag-1)]
        X_test = X_test.iloc[:,range(lag-1)]

        MSE, coef, intercept = fit_and_run_regression(X_train, Y_train, X_test, Y_test)
        
        # Check to see if this MSE is the best or not
        if all(i >= MSE for i in MSE_list):
            coef_best = coef
            best_train_index = train_index
            best_test_index = test_index
        MSE_list.append(MSE)
        
    return MSE_list, coef_best, intercept, best_train_index, best_test_index

def cross_validate_ADL(df):
    """ Runs the backtest with k fold cross validation for ADL model
    Params:
        df (pd.Dataframe): Dataframe with X_t, lags, and sentiment score
    Returns:
        MSE_list (list): List of MSE per fold of cross-validation
        coef_best (list): List of coef per fold of cross-validation
        best_train_index (list): Indices of the train index with best MSE
        best_test_index (list): Indicies of test index with best MSE
        
    """
    tscv = TimeSeriesSplit(n_splits=8)
    X = df.loc[:, df.columns !='X_t']
    Y = df['X_t']
    
    MSE_list = []
    coef_list = []
        
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        Y_train, Y_test = Y.iloc[train_index], Y[test_index]
        
        # find optimal lag period
        lag = get_lag_period(Y_train)

        # filter the lags and sentiment values
        X_train = X_train.iloc[:,range(lag-1)]
        X_test = X_test.iloc[:,range(lag-1)]
        
        # Add in the sentiment values
        X_train['Pos_t-1'] = df.iloc[train_index,-3]
        X_train['Neu_t-1'] = df.iloc[train_index,-2]
        X_train['Neg_t-1'] = df.iloc[train_index,-1]
        
        X_test['Pos_t-1'] = df.iloc[test_index,-3]
        X_test['Neu_t-1'] = df.iloc[test_index,-2]
        X_test['Neg_t-1'] = df.iloc[test_index,-1]
        
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)

        MSE, coef, intercept = fit_and_run_regression(X_train, Y_train, X_test, Y_test)
        
        # Check to see if this MSE is the best or not
        if all(i >= MSE for i in MSE_list):
            coef_best = coef
            best_train_index = train_index
            best_test_index = test_index
        MSE_list.append(MSE)

        
    return MSE_list, coef_best, intercept, best_train_index, best_test_index

def fit_and_run_regression(X_train, Y_train, X_test, Y_test, doprint=False):
    """ Fits the model with train and test data
    Params:
        X_train (pd.Dataframe): X training data
        Y_train (pd.Dataframe): Y training data
    Returns:
        MSE (float): Mean Squared Error 
    """
    LinReg = LinearRegression(normalize=True)
    LinReg.fit(X_train,Y_train)
    Y_pred = LinReg.predict(X_test)
    MSE = mean_squared_error(Y_test, Y_pred)
    coefficients = LinReg.coef_
    intercept = LinReg.intercept_
    if doprint:
        print("R^2 Value: %.2f" %LinReg.score(X_test, Y_test))
        print("Mean squared error: %.2f "% MSE)
        print("Coefficients for the Regression are: " ,coefficients)
    
    return MSE, coefficients, intercept

def plot_AR_model(ticker, best_ADL_train_index, best_ADL_test_index, best_AR_train_index, best_AR_test_index):
    # Plot for AR model
    df = pd.read_csv('../data/'+ticker+'.csv',index_col='Date')
    X = df.loc[:, df.columns !='X_t']
    Y = df['X_t']
    X_train, X_test = X.iloc[best_AR_train_index,:], X.iloc[best_AR_test_index,:]
    Y_train, Y_test = Y.iloc[best_AR_train_index], Y[best_AR_test_index]

    # find optimal lag period
    lag = get_lag_period(Y_train)

    # filter the lags and sentiment values
    X_train = X_train.iloc[:,range(lag-1)]
    X_test = X_test.iloc[:,range(lag-1)]

    # Fit the model
    LinReg = LinearRegression(normalize=True)
    LinReg.fit(X_train,Y_train)
    Y_pred = LinReg.predict(X_test)

    # Plot
    Y_plot = Y_test.copy(deep=True)
    Y_plot = Y_plot.to_frame()
    Y_plot['Predicted'] = pd.Series(Y_pred, index=Y_plot.index)
    Y_plot.columns = ['Actual', 'Predicted']

    plot_data = {}
    plot_data['x_val'] = Y_plot.index
    plot_data['y_actual'] = Y_plot['Actual']
    plot_data['y_predict'] = Y_plot['Predicted']
    return plot_data

def plot_ADL_model(ticker, best_ADL_train_index, best_ADL_test_index, best_AR_train_index, best_AR_test_index):
    # Plot for ADL model
    df = pd.read_csv('../data/'+ticker+'.csv',index_col='Date')
    X = df.loc[:, df.columns !='X_t']
    Y = df['X_t']
    X_train, X_test = X.iloc[best_ADL_train_index,:], X.iloc[best_ADL_test_index,:]
    Y_train, Y_test = Y.iloc[best_ADL_train_index], Y[best_ADL_test_index]

    # find optimal lag period
    lag = get_lag_period(Y_train)

    # filter the lags and sentiment values
    X_train = X_train.iloc[:,range(lag-1)]
    X_test = X_test.iloc[:,range(lag-1)]

    # Add in the sentiment values
    X_train['Pos_t-1'] = df.iloc[best_ADL_train_index,-3]
    X_train['Neu_t-1'] = df.iloc[best_ADL_train_index,-2]
    X_train['Neg_t-1'] = df.iloc[best_ADL_train_index,-1]

    X_test['Pos_t-1'] = df.iloc[best_ADL_test_index,-3]
    X_test['Neu_t-1'] = df.iloc[best_ADL_test_index,-2]
    X_test['Neg_t-1'] = df.iloc[best_ADL_test_index,-1]

    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # Fit the model
    LinReg = LinearRegression(normalize=True)
    LinReg.fit(X_train,Y_train)
    Y_pred = LinReg.predict(X_test)

    # Plot
    Y_plot = Y_test.copy(deep=True)
    Y_plot = Y_plot.to_frame()
    Y_plot['Predicted'] = pd.Series(Y_pred, index=Y_plot.index)
    Y_plot.columns = ['Actual', 'Predicted']

    plot_data = {}
    plot_data['x_val'] = Y_plot.index
    plot_data['y_actual'] = Y_plot['Actual']
    plot_data['y_predict'] = Y_plot['Predicted']

    return plot_data

def predict_next_value(ticker, company, model, is_ADL):
    """ Predicts the next value given a model and ticker
    Params:
        ticker (String): ticker of company trying to predict
        model (sklearn.linear_model.LinearRegression): model used to predict the data
        is_ADL (boolean): boolean to see if is ADL or not
    Returns:
        predicted_value (float): Predicted value
    """
    # Read in however many lags
    if is_ADL:
        number_lags_back = len(model.coef_) - 3
    else:
        number_lags_back = len(model.coef_)
    
    # Read in 50 days back of data
    pred_start = (datetime.datetime.now() - datetime.timedelta(days=50)).isoformat()
    pred_end = datetime.datetime.now().isoformat()
    dataframe = web.DataReader(ticker, 'morningstar', pred_start, pred_end)['Close']
    dataframe = pd.Series.to_frame(dataframe)
    dataframe.reset_index(level=0, drop=True, inplace=True)
    dataframe.columns = ['X_t-1']
    
    # Add the lags
    for i in range(number_lags_back-1):
        dataframe['X_t-' + str(i+2)] = dataframe['X_t-1'].shift((i+1))
        
    # Add the sentiment values if ADL
    if is_ADL:
        dataframe['Pos_t-1'] = 0
        dataframe['Neu_t-1'] = 0
        dataframe['Neg_t-1'] = 0
        trading_dates = dataframe.index
        query_start = (datetime.datetime.now()-datetime.timedelta(days=3)).isoformat()
        query_end = datetime.datetime.now().isoformat()
        dict_master = query_news_articles(company, query_start, query_end, trading_dates, sources=all_sources, pred=True)
        # iterate through dates
        for date in dict_master.keys():
            # when you enter seniment into the dataframe, use the before date not after
            average_sentiment_dict = calculate_sentiment(dict_master[date])

            # Plug this into df
            dataframe.at[date,'Pos_t-1'] = average_sentiment_dict['pos']
            dataframe.at[date,'Neu_t-1'] = average_sentiment_dict['neu']
            dataframe.at[date,'Neg_t-1'] = average_sentiment_dict['neg']
        dataframe = dataframe.fillna(0)
    # Predict the values    
    # print(dataframe.tail(1))
    predicted_value = model.predict(dataframe.tail(1).values.reshape(1, -1))[0]
    predicted_value = round(predicted_value, 2)
    
    return predicted_value

if __name__ == "__main__":
  app.run(debug=True)

def test():
    createModel("Amazon.com Inc.")