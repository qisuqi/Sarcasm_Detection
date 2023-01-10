import tweepy
import Credential
import pandas as pd

auth = tweepy.OAuthHandler(Credential.Consumer_key, Credential.Consumer_secret)
auth.set_access_token(Credential.Access_token, Credential.Access_token_secret)
api = tweepy.API(auth)

file = pd.read_csv('sarcasm-annos-emnlp13.csv')
file.columns = ['id', 'result']

Tweets = []
for id in file['id']:
    try:
        tweets = api.get_status(id).text
    except tweepy.error.TweepError:
        tweets = 'nan'
    Tweets.append(tweets)

file.insert(2,'tweets',Tweets)
file.to_csv('Riloff_tweets.csv')


