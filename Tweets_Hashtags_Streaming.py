from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import time
import csv
import sys
import Credential

tweets_to_capture = 1000

class StdOutListener(StreamListener):

    def __init__(self, api=None):
        self.api = api
        self.num_tweets = 0
        self.filename = 'data' + '_' + time.strftime('%Y%m%d-%H%M%S') + '.csv'
        csv_file = open(self.filename, 'w')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['text',
                            'created_at',
                            'lang',
                            'id'])

    def on_status(self, status):
        csv_file = open(self.filename, 'a')
        csv_writer = csv.writer(csv_file)
        self.num_tweets += 1

        if not 'RT @' in status.text:
            try:
                csv_writer.writerow([status.text,
                                     status.created_at,
                                     status.lang,
                                     status.id])
            except Exception as e:
                print(e)
                pass

        if self.num_tweets <= tweets_to_capture:
            if self.num_tweets % 100 == 0:
                print('Number of tweets captured: {}'.format(self.num_tweets))
            return True
        else:
            return False

        csv_file.close()

        return

    def on_error(self, status_code):
        print('Encountered with status code:', status_code)

        if status_code == 401:
            return False

    def on_delete(self, status_id, user_id):
        print('Delete Notice')

        return
    def on_limit(self, track):
        print('Rate Limited, continuing')

        return True

    def on_timeout(self):
        print(sys.stderr, 'Timeout...')

        time.sleep(10)

        return

def StartMining(queries):
    auth = OAuthHandler(Credential.Consumer_key, Credential.Consumer_secret)
    auth.set_access_token(Credential.Access_token, Credential.Access_token_secret)

    l = StdOutListener()

    stream = Stream(auth, l)

    stream.filter(track=queries)

StartMining(['#sarcasm'])




