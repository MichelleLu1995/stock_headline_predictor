from nytimesarticle import articleAPI
api = articleAPI("c91a676aeaef40fd844409c8b0bef485")
company = "Amazon"
start_date = "20160506"
end_date = "20180506"
page_num = 2

# company_dict = {k: [] for k in trading_dates.date}
start_date = start_date.replace("-","")
end_date = end_date.replace("-","")
newsdata = api.search(q=company, 
							  # begin_date = start_date,
							  # end_date = end_date,
							  # fq = "'source':['Reuters','AP', 'The New York Times']", 
							  fq = "'headline:' + company", 
							  # fl = 'headline',
							  # page = 2,
							  facet_filter = True)

articles = newsdata['response']['docs']

for article in articles:
	print(article['headline'])
	print(type(article['pub_date']))
# 	print(article['snippet'])

# 	headline = article['headline']

# 	# description = article['description']
# 	# format of date is 2018-04-13T00:46:59Z (UTC format)
# 	publish_date = article['pub_date'] 
# 	# adjust date for trading day
# 	publish_date, publish_time = publish_date.split('T')
# 	date_arr = publish_date.split('-')
# 	publish_datetime = datetime.date(int(date_arr[0]), int(date_arr[1]), int(date_arr[2]))
# 	time_arr = publish_time[:-1].split(':')
# 	# stock market closes at 4:00 PM EST; if article published after
# 	# 16:00:00+4:00:00 = 20:00:00 UTC headline affects next trading day;
# 	# otherwise affects current trading day
# 	trading_datetime = publish_datetime
# 	if int(time_arr[0]) >= 20:
# 		trading_datetime += datetime.timedelta(1)
	
# 	# if given trading_date invalid (ie if article published on Friday 
# 	# after market close, Saturday, or Sunday before 4 pm est) push trading_date
# 	# to the following Monday (ie first valid trading_date)
# 	while trading_datetime not in trading_dates:
# 		trading_datetime += datetime.timedelta(1)
# 	company_dict[trading_datetime].append(headline)

# print(company_dict)	

