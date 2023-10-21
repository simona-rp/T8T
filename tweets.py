import pandas as pd
import json

# To see all columns
pd.set_option('display.max_columns', None)

# Read json file
# Reading a json file transforms it into a different object from a list/dictionary
data = pd.read_json('tweets_ws23_v1.json')

# Load json file
# Loading a json file keeps the list/dictionary quality and we can use json_normalize on it
with open('tweets_ws23_v1.json') as f:
    data = json.load(f)

# Extract level 0 data
data_level0 = pd.json_normalize(data)
# Remove topic and sentiment data and rename id to match tweet_id
data_level0 = data_level0.loc[:,data_level0.columns.drop(['labels.topic','labels.sentiment'])]

# Extract topic data
data_labels_topic = pd.json_normalize(data,['labels', 'topic'])
data_labels_topic = data_labels_topic.rename(columns={'tweet_id':'tweet_id_topic',
                                                      'id':'id_topic',
                                                      'user_id':'user_id_topic',
                                                      'ground_truth':'ground_truth_topic'})

# Extract sentiment data
data_labels_sentiment = pd.json_normalize(data,['labels','sentiment'])
data_labels_sentiment = data_labels_sentiment.rename(columns={'tweet_id':'tweet_id_sentiment',
                                                              'id':'id_sentiment',
                                                              'user_id':'user_id_sentiment',
                                                              'ground_truth':'ground_truth_sentiment'})

# Merge extracted data
tweets = pd.concat([data_level0, data_labels_topic, data_labels_sentiment], axis = 1, join ='inner')

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
                                            'ground_truth_sentiment'])].rename(columns={'id':'tweet_id',
                                                                                        'user_id_topic':'user_id',
                                                                                        'ground_truth_topic':'ground_truth'})
print(tweets.head())

