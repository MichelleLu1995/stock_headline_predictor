

import pandas_datareader.data as web

import datetime

end_date = datetime.datetime.now()
start_date = (datetime.datetime.now() - datetime.timedelta(days=2*365))

f = web.DataReader('ABT', 'morningstar', start_date, end_date)
print(f)