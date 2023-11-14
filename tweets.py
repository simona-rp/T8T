import pandas as pd, matplotlib.pyplot as plt, seaborn as sns, numpy as np
from sqlalchemy import create_engine
from datetime import datetime
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import json, nltk, re, warnings
# Ignore warnings for future function changes
warnings.simplefilter(action='ignore', category=FutureWarning)

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
# Check for duplicates
tweets['id'].equals(tweets['tweet_id_topic'])
tweets['id'].equals(tweets['tweet_id_sentiment'])
tweets['tweet_id_topic'].equals(tweets['tweet_id_topic'])
# id, tweet_id_topic and tweet_id_sentiment are duplicates, keep one as  tweet_id
tweets['user_id_topic'].equals(tweets['user_id_sentiment'])
# user_id_topic and user_id_sentiment are duplicates, keep one as user_id
tweets['id_topic'].equals(tweets['id_sentiment'])
# not duplicates, keep both
tweets['ground_truth_topic'].equals(tweets['ground_truth_sentiment'])
# ground_truth_topic and ground_truth_sentiment are duplicates, keep one as ground_truth

tweets = tweets.loc[:, tweets.columns.drop(['tweet_id_topic',
                                            'tweet_id_sentiment',
                                            'user_id_sentiment',
                                            'ground_truth_sentiment'])].rename(columns={'id': 'tweet_id',
                                                                                        'user_id_topic': 'user_id',
                                                                                        'ground_truth_topic': 'ground_truth'})

# Check variable types
types = tweets.dtypes
# object -> source_created_at, author_id, text, source, language, tweet_id,
# object -> source_id, user_id, topic, id_topic, sentiment, id_sentiment
# float64 = numeric -> longitude, latitude
# bool = relevant, ground_truth
# Transformation needed

# Convert source_created_at to datetime
tweets['created'] = pd.to_datetime(tweets['created'], format='mixed').apply(lambda x: x.replace(microsecond=0))

# Setup connection to MySQL database
engine = create_engine('mysql+pymysql://HTW:mysql@localhost/tweets')
# Import data into MySQL database
tweets.to_sql(name='tweets', con=engine, if_exists='replace')

# Data Understanding
# Summary statistics of the dataset
tweets.describe(include='all')
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

# count_missing function
def count_missing(df, feature_name):

    # Count number of missing value in the feature from a specific dataframe.
    missing_count = df[feature_name].isnull().sum()
    return print(f"The feature '{feature_name}' has ", missing_count, " missing values.") 

# top_duplicate function
def top_duplicate(df, feature_name):

    # Count values of each duplicate  
    duplicate_count = df[feature_name].value_counts()
    # Output the top 10 duplicates and their counts
    top_duplicates = duplicate_count.head(10)

    return print("The top 10 values with most duplicates:\n"
    ,top_duplicates)

# Sentiment
sent = tweets.groupby(['sentiment'])['sentiment'].count()
# negative = 10.628 negative, neutral = 6.079 neutral, positive = 242

# Source
sour = tweets.groupby(['source'])['source'].count()
# brandwatch = 15.548, sprinkl = 1.401

# Language
lang = tweets.groupby(['language'])['language'].count()
# en-GB = 10.051, en = 6.898 > only English

# Source_id
source_id = tweets.groupby(['source_id'])['source_id'].count().sort_values(ascending=False)
source_id_desc = tweets['source_id'].nunique()
# 14464 unique values -> 2484 duplicates, no significant duplicate
count_missing(tweets,'source_id')
# 1401 missing values
# source_id is an identifier for 'brandwatch' social media analytics tool
filtered_brandwatch = tweets[tweets['source'] == 'brandwatch']
count_missing(filtered_brandwatch,'source_id')
# no missing value for source_id, when source is brandwatch.

# Relevant
relevant = tweets.groupby(['relevant'])['relevant'].count()
# There is only one value "True" in this feature.
count_missing(tweets,'relevant')
# no missing value

# User_id
user = tweets.groupby(['user_id'])['user_id'].count()
# There is only one value "Z003XDCS" in this feature.
# Possibly this is a user_id of Thameslink admin.
count_missing(tweets,'user_id')
# no missing value

# Ground_truth
ground = tweets.groupby(['ground_truth'])['ground_truth'].count()
print(ground)
# There is only one value "True" in this feature.
count_missing(tweets,'ground_truth')
# no missing value

# Id_topic
id_topic = tweets.groupby(['id_topic'])['id_topic'].count().sort_values(ascending=False)
id_topic_desc = tweets['id_topic'].nunique()
# 16711 unique values -> 238 duplicates, no significant duplicate
# One tweet can have several 'topics' associated.
# One id_topic represents one unique topic from one tweet with unique tweet_id
# 16711 unique topics from 15749 tweets with unique tweet_id
count_missing(tweets,'id_topic')
# no missing value

# Id_sentiment
id_sentiment = tweets.groupby(['id_sentiment'])['id_sentiment'].count().sort_values(ascending=False)
id_sentiment_desc = tweets['id_sentiment'].nunique()
print(id_sentiment_desc)
# 15781 unique values -> 1168 duplicates, no significant duplicate
# One tweet can have several 'sentiment' associated.
# Several topics in the same tweet can have the same sentiment and possess the same id_sentiment
# For example, complaint about table, door, ACc have negative sentiment, so all of them have the same id_sentiment
count_missing(tweets,'id_sentiment')
# no missing value

# Created
created_desc = tweets['created'].describe(percentiles = [])
created_na = tweets['created'].isnull().sum()
# No missing values
# range from 16.01.2019 to 01.12.2020

# Break datetime down to wanted granularity
months = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
weekdays = ["Monday", "Tuesday", "Wednesday","Thursday", "Friday","Saturday","Sunday"]
tweets['month'] = pd.Categorical(tweets.created.dt.strftime('%B'), categories = months, ordered = True)
tweets['year'] = pd.Categorical(tweets.created.dt.strftime('%Y'), ordered = True)
tweets['week'] = pd.Categorical(tweets.created.dt.isocalendar().week, ordered = True)
tweets['weekday'] = pd.Categorical(tweets.created.dt.strftime('%A'), categories = weekdays, ordered = True)

# Tweet IDs
# Check counts of tweet_id
# By year
created_y = tweets.groupby(['year'])['tweet_id'].count().to_frame().reset_index().rename(columns={'tweet_id':'count'})
barchart_y = sns.barplot(data=created_y, x='year', y='count', hue='year', palette='dark:#E21185', legend = False)
barchart_y.set(xlabel='Year created', ylabel='Number of tweets', title='Number of tweets per year')
# By month and year
created_my = tweets.groupby(['year', 'month'])['tweet_id'].count().to_frame().reset_index().rename(columns={'tweet_id':'count'})
# barchart_my = sns.barplot(data=created_my, x='month', y='count', hue='year', palette='dark:#E21185', legend = True)
# barchart_my.set(xlabel='Month created', ylabel='Number of tweets', title='Number of tweets per month')
# By calendar week
created_wy = tweets.groupby(['year','week'])['tweet_id'].count().to_frame().reset_index().rename(columns={'tweet_id':'count'})
# barchart_wy = sns.barplot(data=created_wy, x='week', y='count', hue='year', palette='dark:#E21185', legend = True)
# barchart_wy.set(xlabel='Week created', ylabel='Number of tweets', title='Number of tweets per week')
# By weekday
created_dy = tweets.groupby(['year','weekday'])['tweet_id'].count().to_frame().reset_index().rename(columns={'tweet_id':'count'})
# barchart_dy = sns.barplot(data=created_dy, x='weekday', y='count', hue='year', palette='dark:#E21185', legend = True)
# barchart_dy.set(xlabel='Day created', ylabel='Number of tweets', title='Number of tweets per weekday')

# Duplicate tweets
tweet_id_desc = tweets['tweet_id'].nunique()
# 15749 unique tweet_ids
tweet_id_na = tweets['tweet_id'].isnull().sum()
# No missing values
tweet_id_dups = tweets.groupby(['tweet_id'])['tweet_id'].count().to_frame().rename(columns={'tweet_id':'count'}).reset_index()
tweet_id_dups['bins'] = np.digitize(tweet_id_dups['count'], bins = [2,3,4,5,6,7])
tweet_id_bincount = np.bincount(tweet_id_dups['bins'])
tweet_id_bins = ['1', '2', '3', '4', '5', '6', '>6']
# barplot_tweet = sns.barplot(x = tweet_id_bins, y = tweet_id_bincount, color='#E21185')
# barplot_tweet.set(xlabel='Number of duplicate tweets', ylabel='Count of number of duplicate tweets', title='Count of the number of duplicate tweets')
# barplot_tweet.bar_label(barplot_tweet.containers[0])

# Authors
# Number of authors
auth_desc = tweets['author_id'].describe()
auth_na = tweets['author_id'].isnull().sum()
# 7139 unique authors
# No missing values

# Number of authors per year
created_auth_y = tweets.groupby(['year'])['author_id'].count().to_frame().reset_index().rename(columns={'author_id':'count'})
# barchart_auth_y = sns.barplot(data=created_auth_y, x='year', y='count', hue='year', palette='dark:#E21185', legend = False)
# barchart_auth_y.set(xlabel='Year created', ylabel='Number of authors', title='Number of authors per year')
# By month and year
created_auth_my = tweets.groupby(['year', 'month'])['author_id'].count().to_frame().reset_index().rename(columns={'author_id':'count'})
# barchart_auth_my = sns.barplot(data=created_auth_my, x='month', y='count', hue='year', palette='dark:#E21185', legend = True)
# barchart_auth_my.set(xlabel='Month created', ylabel='Number of authors', title='Number of authors per month')
# By calendar week
created_auth_wy = tweets.groupby(['year','week'])['author_id'].count().to_frame().reset_index().rename(columns={'author_id':'count'})
# barchart_auth_wy = sns.barplot(data=created_wy, x='week', y='count', hue='year', palette='dark:#E21185', legend = True)
# barchart_auth_wy.set(xlabel='Week created', ylabel='Number of authors', title='Number of authors per week')
# By weekday
created_auth_dy = tweets.groupby(['year','weekday'])['author_id'].count().to_frame().reset_index().rename(columns={'author_id':'count'})
# barchart_auth_dy = sns.barplot(data=created_auth_dy, x='weekday', y='count', hue='year', palette='dark:#E21185', legend = True)
# barchart_auth_dy.set(xlabel='Day created', ylabel='Number of authors', title='Number of authors per day')

# Number of tweets per author all time
auth_all = tweets.groupby(['author_id'])['tweet_id'].count().sort_values (ascending=False).to_frame().reset_index().rename(columns={'tweet_id':'count'})
auth_all['bins'] = np.digitize(auth_all['count'], bins = [2,3,4,5,6,7,8,9,10,11])
auth_bincount = np.bincount(auth_all['bins'])
bins = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '>10']
# barplot_auth_all = sns.barplot(x = bins, y = auth_bincount, color='#E21185')
# barplot_auth_all.set(xlabel='Number of tweets per author', ylabel='Count of number of tweets', title='Count of the number of tweets per author')
# barplot_auth_all.bar_label(barplot_auth_all.containers[0])

tweet_count_desc = auth_all['count'].describe()
# Average number of tweets per author = 2.3741
# Standard deviation = 13.8693
# Max = 715 tweets by author_id = 2589703207 > Thameslink customer service
# 496 tweets by author_id = 963564801195741184 > Thameslink update
# 457 tweets by author_id = 2589703207.0 > Thameslink customer service
# 294 tweets by author_id = 9.635648011957412e+17 > Thameslink update
# 283 tweets by author_id = 1099408261781032960 > Thameslink update + customer service
# Assumption: abnormal amount of tweets linked to Thameslink account
#auth_all_boxplot = sns.boxplot(auth_all['count'])
auth_upper = auth_all['count'].mean()+3*auth_all['count'].std()
auth_all_o = auth_all[auth_all['count']<=auth_upper]
tweet_count_desc2 = auth_all_o['count'].describe()
# Average number of tweets per author all time = 1.8391

# Number of tweets per author per year
auth_y = tweets.groupby(['year','author_id'])['tweet_id'].count().to_frame().reset_index().rename(columns={'tweet_id':'count'})
auth_y['bins'] = np.digitize(auth_y['count'], bins = [1,2,3,4,5,6,7,8,9,10,11])
bins_y = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '>10']
auth_y_bincount = np.append(arr = np.bincount(auth_y[auth_y['year'] == '2019']['bins']),
                            values = np.bincount(auth_y[auth_y['year'] == '2020']['bins']))
auth_y_year = np.append(arr = [2019]*len(bins_y), values = [2020]*len(bins_y))
auth_y_2 = pd.DataFrame({
    'bincount': auth_y_bincount,
    'year': auth_y_year,
    'bins': bins_y*2})
# barplot_auth_y = sns.barplot(data = auth_y_2, x = 'bins', y = 'bincount', hue='year', color='#E21185')
# barplot_auth_y.set(xlabel='Number of tweets per author per', ylabel='Count of number of tweets', title='Count of the number of tweets per author per year')
# for i in barplot_auth_y.containers:
#     barplot_auth_y.bar_label(i,)

# Average number of tweets per author per year
auth_y = tweets.groupby(['year','author_id'])['tweet_id'].count().to_frame().reset_index().rename(columns={'tweet_id':'count'})
upper_y = auth_y['count'].mean()+3*auth_y['count'].std()
auth_y = auth_y[(auth_y['count']>0) & (auth_y['count']<=upper_y)]
auth_mean_y = auth_y.groupby(['year']).mean('count').reset_index()
# barchart_auth_mean_y = sns.barplot(data=auth_mean_y, x='year', y='count', hue='year', palette='dark:#E21185', legend = True)
# barchart_auth_mean_y.set(xlabel='Year created', ylabel='Average number of tweets per author', title='Average number of tweets per author per year')
# Average number of tweets per author per month and year
auth_my = tweets.groupby(['year','month','author_id'])['tweet_id'].count().to_frame().reset_index().rename(columns={'tweet_id':'count'})
upper_my = auth_my['count'].mean()+3*auth_my['count'].std()
auth_my = auth_my[(auth_my['count']>0) & (auth_my['count']<=upper_my)]
auth_mean_my = auth_my.groupby(['year','month']).mean('count').reset_index()
# barchart_auth_mean_my = sns.barplot(data=auth_mean_my, x='month', y='count', hue='year', palette='dark:#E21185', legend = True)
# barchart_auth_mean_my.set(xlabel='Month created', ylabel='Average number of tweets per author', title='Average number of tweets per author per month')
# By calendar week
auth_wy = tweets.groupby(['year', 'week', 'author_id'])['tweet_id'].count().to_frame().reset_index().rename(columns={'tweet_id': 'count'})
auth_wy = auth_wy[(auth_wy['count'] > 0)]
auth_mean_wy = auth_wy.groupby(['year', 'week']).mean('count').reset_index()
# barchart_auth_mean_wy = sns.barplot(data=auth_mean_wy, x='week', y='count', hue='year', palette='dark:#E21185', legend = True)
# barchart_auth_mean_wy.set(xlabel='Week created', ylabel='Average number of tweets per author', title='Average number of tweets per author per week')
# By weekday
auth_dy = tweets.groupby(['year', 'weekday', 'author_id'])['tweet_id'].count().to_frame().reset_index().rename(columns={'tweet_id': 'count'})
auth_dy = auth_dy[(auth_dy['count'] > 0)]
auth_mean_dy = auth_dy.groupby(['year', 'weekday']).mean('count').reset_index()
# barchart_auth_mean_dy = sns.barplot(data=auth_mean_dy, x='weekday', y='count', hue='year', palette='dark:#E21185', legend = True)
# barchart_auth_mean_dy.set(xlabel='Day created', ylabel='Average number of tweets per author', title='Average number of tweets per weekday')

# GPS data
# Longitutde
longi_desc = tweets['longitude'].describe()
# Number of observations = 1.425
longi_na = tweets['longitude'].isnull().sum()
# 15.524 missing values

# Latitude
lati_desc = tweets['latitude'].describe()
# Number of observations = 1.425
lati_na = tweets['latitude'].isnull().sum()
# 15.524 missing values

# GPS data not reliable for identifying issue location

# Text
# Twitter channel
conditions = [tweets['text'].str.contains('@GatwickExpress'),
              tweets['text'].str.contains('@GNRailUK'),
              tweets['text'].str.contains('@SouthernRailUK'),
              tweets['text'].str.contains('@TLRailUK')]

flag_names = dict(flag_id = [1,2,3,4],
                  flag = ['@GatwickExpress','@GNRailUK','@SouthernRailUK','@TLRailUK'])

tweets['flag_channel'] = np.select(conditions, flag_names['flag_id'], default = 0)
tweets_channel_desc = tweets.groupby(['flag_channel'])['tweet_id'].count()
# No specific channel = 5.641
# GatwickExpress = 215
# GNRailUK = 398
# SouthernRailUK = 644
# TLRailUK = 10.051 > most used channel

thameslink_routes = pd.read_csv('thameslink_routes.csv')
thameslink_routes = pd.DataFrame(thameslink_routes)
thameslink_lines = pd.read_csv('thameslink_lines.csv')
thameslink_lines = pd.DataFrame(thameslink_lines)

tweets['flag_station'] = tweets['text'].apply(lambda x: next((word for word in thameslink_routes['station'] if word.lower() in x.lower()), 'NaN'))
station_count = tweets.groupby(['flag_station'])['flag_station'].count()
# Missing values = 11.734
pd.set_option('display.max_rows', 1000)

tweets['flag_line'] = tweets['text'].apply(lambda x: next((word for word in thameslink_lines['line'] if word.lower() in x.lower()), 'NaN'))
tweets['flag_line_start'] = tweets['text'].apply(lambda x: next((word for word in thameslink_lines['start'] if word.lower() in x.lower()), 'NaN'))
tweets['flag_line_end'] = tweets['text'].apply(lambda x: next((word for word in thameslink_lines['end'] if word.lower() in x.lower()), 'NaN'))
line_count = tweets.groupby(['flag_line'])['flag_line'].count()
# Missing values = 16.941 > only 3 lines Bedford-Brighton (4), Cambridge-Brighton (2), Luton-Rainham (2)
line_count2 = tweets.groupby(['flag_line_start'])['flag_line'].count()
# Missing values = 13.717
line_count3 = tweets.groupby(['flag_line_end'])['flag_line'].count()
# Missing values =14.298

tweets = tweets.astype({'flag_station': 'str',
                        'flag_line': 'str',
                        'flag_line_start': 'str',
                        'flag_line_end': 'str'})
# Check overall conditions of GPS data
tweets_loc = tweets.loc[(tweets['flag_station'].str.len() > 0) |
                        (tweets['flag_line'].str.len() > 0) |
                        (tweets['flag_line_start'].str.len() > 0) |
                        (tweets['flag_line_end'].str.len() > 0) |
                        ((tweets['longitude'].notnull()) & (tweets['latitude'].notnull()))]
# 8.053 observations with location data

# Tokenize text into words & determine word frequency
# Download needed packages
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download("stopwords")
# Separate text
text = pd.DataFrame(tweets[['text', 'topic', 'sentiment']])
text['text'] = text['text'].astype(str)
# Make all text lowercase to identify stopwords
text['text'] = text['text'].str.lower()
# Remove non-alphanumeric characters from text
text['text'] = text['text'].str.replace(r'\W+', ' ', regex=True)
text['topic'] = text['topic'].str.replace(r'\W+', ' ', regex=True)
# In the top 100 words, some numbers were included so run the following to remove numeric characters
text['text'] = text['text'].str.replace(r'\d', ' ', regex=True)
# Define stopwords in the English language
sw = nltk.corpus.stopwords.words('english')
# Remove stopwords from text
text['text_cleaned'] = text['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (sw)]))
# Tokenize text into separate words
text['tokens'] = text.apply(lambda row: nltk.word_tokenize(row['text_cleaned']), axis=1)
words = text[['tokens', 'topic', 'sentiment']]
words = words.explode('tokens').reset_index()
# Check word counts
# Overall
words_gen = words.groupby(['tokens'])['tokens'].count().sort_values(ascending=False).to_frame(name='count').reset_index()
# Top 100 words
words_gen_100 = words_gen[words_gen.index <= 100]
# Without numeric characters
words_gen_desc = words_gen['count'].describe()
# Average = 16.27, Standard deviation = 157.80
upper_w = words_gen['count'].mean()+3*words_gen['count'].std()
words_out = words_gen[words_gen['count']>upper_w]
# The outlier words match the top 100 words
# Create a WordCloud to visualize word frequencies
cloud_gen_text = words_gen['tokens'].to_string()
# wordcloud = WordCloud(background_color='white', colormap = 'magma', max_words = 100, width = 800, height = 400).generate(cloud_gen_text)
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# Per topic & negative sentiment
words_top = words[words['sentiment']=='negative'].groupby(['topic', 'tokens'])['tokens'].count().sort_values(ascending=False).to_frame(name='count').reset_index()
words_top5 = words_top.groupby(['topic']).head(5).sort_values(['topic', 'count'], ascending = False)
cloud_top_text = words[words['sentiment']=='negative'][['topic','tokens']]
# for x in cloud_top_text.topic.unique():
#     wc = WordCloud(background_color="white", colormap = 'magma', max_words=10, width = 800, height = 400)
#     wc.generate(cloud_top_text[(cloud_top_text.topic == x)].tokens.to_string())
#     plt.imshow(wc)
#     plt.title(x)
#     plt.axis("off")
#     plt.savefig('C:/Users/SimonaPaskaleva/Desktop/Tweets/T8T/wc'+str(x)+'.png')

# Topic
topic = tweets.groupby(['topic'])['topic'].count().sort_values(ascending=False)
topic_desc = tweets['topic'].nunique()
# 23 Unique topics (delays: 9023, none: 2304, service: 884, station: 754, ..., brakes: 44, roof: 15, handrails: 2)
count_missing(tweets,'topic')
# no missing value

# Most frequent topic by year
created_topic_y = tweets.groupby(['year', 'topic']).size().reset_index(name='count').sort_values(by=['year', 'count'], ascending=[True, False])
# 'Delays' is the most frequent in both years by far, followed by 'none'. 'Covid' is 3rd in 2020.
# Most frequent topic by month and year
created_topic_my = tweets.groupby(['year', 'month', 'topic']).size().reset_index(name='count')\
    .sort_values(by=['year', 'month', 'count'], ascending=[True, True, False]).groupby(['year', 'month']).head(1)
# 'Delays' is the most prominent topic throughout 2019. In 2020, 'Delays' and 'Covid' are most frequent.
# Most frequent topic by weekday
created_topic_dy = tweets.groupby(['year','weekday', 'topic']).size().reset_index(name='count')\
    .sort_values(by=['year','weekday', 'count'], ascending=[True, True, False]).groupby(['year', 'weekday']).head(1)
# Most frequent weekday by topic
created_topic_d = tweets.groupby(['topic', 'weekday']).size().reset_index(name='count').sort_values(by=['topic', 'count'],
    ascending=[False, False]).groupby(['topic']).head(1).sort_values(by='count', ascending=False)
# Delays are reported the most on Wednesday, All topics are most reported during weekdays.

# Number of topics per tweet
topic_tweet = tweets.groupby(['tweet_id'])['topic'].nunique().reset_index(name='count').sort_values(by=['count'], ascending=[False])
# Most number of topics for one tweet is 4.
topic_tweet_sum = topic_tweet.groupby(['count']).size().reset_index(name='sum').sort_values(by='sum', ascending=True)
# Tweets with 4 topics = 3 (0.02%), Tweets with 3 topics = 52 (0.3%), Tweets with 2 topics = 849 (5.5%), 96% has 1 topic per tweet.
#barchart_y = sns.barplot(data=topic_tweet_sum, x='count', y='sum', hue='year', palette='dark:#E21185')
#barchart_y.set(xlabel='Number of topics per tweet', ylabel='Number of tweets', title='Number of topics per tweet')


# Sentiment
sentiment = tweets.groupby(['sentiment'])['topic'].count().sort_values(ascending=False)
sentiment_desc = tweets['sentiment'].nunique()
# 3 Sentiments ; Negative: 10628 (64.4%) , Neutral: 6079 (36.9%) , Positive: 272 (1.6%)
count_missing(tweets,'sentiment')
# no missing value

# Number of sentiments per tweet/topic
sentiment_tweet = tweets.groupby(['tweet_id'])['sentiment'].nunique().reset_index(name='count').sort_values(by=['count'], ascending=[False])
sentiment_tweet_sum = sentiment_tweet.groupby(['count']).size().reset_index(name='sum').sort_values(by='sum', ascending=True)
# 32 Tweets with 2 sentiments, other than that one sentiment per tweet.
sentiment_topic = tweets.groupby(['id_topic'])['sentiment'].nunique().reset_index(name='count').sort_values(by=['count'], ascending=[False])
sentiment_topic_sum = sentiment_topic.groupby(['count']).size().reset_index(name='sum').sort_values(by='sum', ascending=True)
# 20 Unique topics with 2 sentiments, other than that one sentiment per unique topic.

# Sentiment associated with topics
sentimental_topic = tweets.groupby(['topic', 'sentiment'])['sentiment'].count().sort_values(ascending=False)
# Delays: negative: 5851, neutral: 3156
# None: neutral: 1206, negative: 1053, positive: 45
# Service: negative: 605, neutral: 249, ...

