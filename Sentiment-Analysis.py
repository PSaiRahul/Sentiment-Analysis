import tweepy
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

#Enter your credentials
consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

tweets = api.search('Machine Learning', count=100)


data = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])

print(data.head(10))


print(tweets[0].created_at)


ana = SentimentIntensityAnalyzer()


li = []

for index, row in data.iterrows():
  ss = ana.polarity_scores(row["Tweets"])
  li.append(ss)
  
se = pd.Series(li)
data['polarity'] = se.values

print(data.head(100))