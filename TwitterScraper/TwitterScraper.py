
#Scrape tweets from Twitter using Twint
import twint
import nest_asyncio
nest_asyncio.apply()

c = twint.Config()
Pandas_clean = True
c.Profile_full = True # for when getting the tweets from a profile

c.Pandas = True
c.Search = "#vaccine #covid19"
c.Show_hashtags = True
c.Lang = 'en' #english tweets
c.Until = '2021-02-26'
c.Since = '2021-01-01'
c.Count = True #show the number of tweets fetched
c.Retweets = True
c.Popular_tweets = True
c.Store_csv = True
c.Output =  "covid19_vaccine.csv"
twint.run.Search(c)
