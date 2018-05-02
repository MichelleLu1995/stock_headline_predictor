# testing quandl api

from pandas_datareader import data
import numpy as np
import pandas as pd
import quandl
import datetime

quandl.ApiConfig.api_key = "REidDLJ8C4pPMRMZGLnA"

comp_symb = 'MSFT'

table = quandl.get('EOD/' + comp_symb, start_date='2018-05-01', end_date='2018-05-01')

# output: Open   High    Low  Close      Volume  Dividend  Split  Adj_Open  Adj_High  Adj_Low  Adj_Close  Adj_Volume
# print(table)
# print(table['Close'].values[0])

stock_after_date = "2018-04-13T00:46:59Z"[:10]
date_arr = stock_after_date.split('-')
datetime_after = datetime.date(int(date_arr[0]), int(date_arr[1]), int(date_arr[2]))
stock_before_date = (datetime_after - datetime.timedelta(1)).isoformat()

# print(stock_after_date)
# print(stock_before_date)

stock_before = quandl.get('EOD/'+comp_symb, start_date=stock_before_date, end_date=stock_before_date)['Close'].values[0]
stock_after = quandl.get('EOD/'+comp_symb, start_date=stock_after_date, end_date=stock_after_date)['Close'].values[0]
# print(quandl.get('EOD/'+comp_symb, start_date='2018-04-11', end_date='2018-04-11'))
print(stock_before)
print(stock_after)
print(stock_after-stock_before)

