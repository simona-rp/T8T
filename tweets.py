import pandas as pd
from sqlalchemy import create_engine
import json

# To see all columns
pd.set_option('display.max_columns', None)

# Read json file
# Reading a json file transforms it into a different object from a list/dictionary
data = pd.read_json('tweets_ws23_v1.json')

# Load json file
# Loading a json file keeps the list/dictionary quality, and we can use json_normalize on it
with open('tweets_ws23_v1.json') as f:
    data = json.load(f)

# Extract level 0 data
data_level0 = pd.json_normalize(data)
# Remove topic and sentiment data
data_level0 = data_level0.loc[:, data_level0.columns.drop(['labels.topic', 'labels.sentiment'])]
data_level0 = data_level0.rename(columns={'source_created_at': 'created'})

# Extract topic data
data_labels_topic = pd.json_normalize(data, ['labels', 'topic'])
data_labels_topic = data_labels_topic.rename(columns={'tweet_id': 'tweet_id_topic',
                                                      'id': 'id_topic',
                                                      'user_id': 'user_id_topic',
                                                      'ground_truth': 'ground_truth_topic'})

# Extract sentiment data
data_labels_sentiment = pd.json_normalize(data, ['labels', 'sentiment'])
data_labels_sentiment = data_labels_sentiment.rename(columns={'tweet_id': 'tweet_id_sentiment',
                                                              'id': 'id_sentiment',
                                                              'user_id': 'user_id_sentiment',
                                                              'ground_truth': 'ground_truth_sentiment'})

# Merge extracted data
tweets = pd.concat([data_level0, data_labels_topic, data_labels_sentiment], axis=1, join='inner')

# Check for duplicates
print(tweets['id'].equals(tweets['tweet_id_topic']))
print(tweets['id'].equals(tweets['tweet_id_sentiment']))
print(tweets['tweet_id_topic'].equals(tweets['tweet_id_topic']))
# id, tweet_id_topic and tweet_id_sentiment are duplicates, keep one as  tweet_id
print(tweets['user_id_topic'].equals(tweets['user_id_sentiment']))
# user_id_topic and user_id_sentiment are duplicates, keep one as user_id
print(tweets['id_topic'].equals(tweets['id_sentiment']))
# not duplicates, keep both
print(tweets['ground_truth_topic'].equals(tweets['ground_truth_sentiment']))
# ground_truth_topic and ground_truth_sentiment are duplicates, keep one as ground_truth

tweets = tweets.loc[:, tweets.columns.drop(['tweet_id_topic',
                                            'tweet_id_sentiment',
                                            'user_id_sentiment',
                                            'ground_truth_sentiment'])].rename(columns={'id': 'tweet_id',
                                                                                        'user_id_topic': 'user_id',
                                                                                        'ground_truth_topic': 'ground_truth'})

# Check variable types
types = tweets.dtypes
print(types)
# object -> source_created_at, author_id, text, source, language, tweet_id,
# object -> source_id, user_id, topic, id_topic, sentiment, id_sentiment
# float64 = numeric -> longitude, latitude
# bool = relevant, ground_truth
# Transformation needed

# Convert source_created_at to datetime
tweets['created'] = pd.to_datetime(tweets['created'], format='mixed').apply(lambda x: x.replace(microsecond=0))
print(tweets['created'])
print(type(tweets['created']))

# Setup connection to MySQL database
engine = create_engine('mysql+pymysql://HTW:mysql@localhost/tweets')
# Import data into MySQL database
tweets.to_sql(name='tweets', con=engine, if_exists='replace')

# Data Understanding
# Summary statistics of the dataset
print(tweets.describe(include='all'))
# Time period 16.01.2019-01.12.2020
# Unique author_id = 7.139 > multiple reports by the same authors, author_id = 2589703207 has 715 tweets
# Unique text = 15.749 > duplicate tweets
# Unique tweet_id = 15.749 > duplicate tweets
# Unique source_id = 14.464 > duplicate sources? source_id is the user_id of the subject user
# 1 unique user_id = Z003XDCS > id of Thameslink?
# all tweets are relevant
# 23 topics, topic = delay most frequent with 9.023 tweets
# all tweets are true
# 3 sentiments
# id_topic and id_sentiment?

# Sentiment
sent = tweets.groupby(['sentiment'])['sentiment'].count()
print(sent)
# negative = 10.628 negative, neutral = 6.079 neutral, positive = 242

# Source
sour = tweets.groupby(['source'])['source'].count()
print(sour)
# brandwatch = 15.548, sprinkl = 1.401

# Language
lang = tweets.groupby(['language'])['language'].count()
print(lang)
# en-GB = 10.051, en = 6.898
# Only English

# Authors
auth = tweets.groupby(['author_id'])['author_id'].count().sort_values(ascending=False)
print(auth)

# Created
# By year
created_y = tweets.groupby(tweets['created'].dt.year)['tweet_id'].count()
# By month and year
created_my = tweets.groupby(tweets['created'].dt.strftime('%m.%Y'))['tweet_id'].count()
# By week
created_w = tweets.groupby(tweets['created'].dt.isocalendar().week)['tweet_id'].count()
# By weekday
created_d = tweets.groupby(tweets['created'].dt.isocalendar().day)['tweet_id'].count()
print(created_y, created_my, created_w, created_d)
