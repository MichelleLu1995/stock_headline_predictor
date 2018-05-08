from nytimesarticle import articleAPI
import datetime

api = articleAPI("c91a676aeaef40fd844409c8b0bef485") 
company = "Amazon"
start_date = 20160506
start_date_dt = datetime.date(2016,5,6)
end_date = 20180506
end_date_dt = datetime.date(2018,5,6)
page_num = 2


# this initialization of trading_dates/company_dict only exists because we are not
# using the stock api in this script - for testing purposes only, do not migrate!! 
trading_dates = [end_date_dt - datetime.timedelta(days=x) for x in range(0, (end_date_dt - start_date_dt).days)]
company_dict = {k: [] for k in trading_dates}

#company_dict = {k: [] for k in trading_dates.date}
#start_date = start_date.replace("-","")
#end_date = end_date.replace("-","")
newsdata = api.search(q='3M',
                               begin_date = start_date,
                               end_date = end_date,
                              # fq = "'source':['Reuters','AP', 'The New York Times']", 
                              #fq = "'headline:' + company", 
                              fq='headline:(3M) and source:(Reuters AP The New York Times)',
#                               fl = "['headline','pub_date','snippet']",
#                              fl='pub_date',
                              page = 0,
                              facet_filter = True
                                )

#print(newsdata) # newsdata is full HTTP response
number_of_hits = newsdata['response']['meta']['hits']
number_of_pages = (number_of_hits // 10) + 1

# page through results and add headlines to comapny_dict
for i in range(0, min(number_of_pages, 100)):
    newsdata = api.search(q='3M',
                       begin_date = start_date,
                       end_date = end_date,
                      # fq = "'source':['Reuters','AP', 'The New York Times']", 
                      #fq = "'headline:' + company", 
                      fq='headline:(3M) AND source:(Reuters AP The New York Times)',
#                               fl = "['headline','pub_date','snippet']",
#                              fl='pub_date',
                      page = i,
                      facet_filter = True)
    articles = newsdata['response']['docs']
    for article in articles:
        headline = article['headline']['main']
        print(article['pub_date'], '\t', article['headline']['main'])
    #    print(article['headline'])
    #    print(article['snippet'])
    #    print(article['keywords'])
    #    print(article['web_url'])
        print()
    
    #    headline = article['headline']
    #
    #    # description = article['description']
        # format of date is 2018-04-13T00:46:59Z (UTC format)
        publish_date = article['pub_date'] 
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
        
#
#print(company_dict)    

