 import pandas as pd, matplotlib.pyplot as plt, seaborn as sns, numpy as np
from datetime import datetime
import json, nltk, re, warnings
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
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

# Text
# Twitter channel
conditions = [tweets['text'].str.contains('@GatwickExpress'),
              tweets['text'].str.contains('@GNRailUK'),
              tweets['text'].str.contains('@SouthernRailUK'),
              tweets['text'].str.contains('@TLRailUK')]

flag_names = dict(flag_id = [1,2,3,4],
                  flag = ['@GatwickExpress','@GNRailUK','@SouthernRailUK','@TLRailUK'])

tweets['flag_channel'] = np.select(conditions, flag_names['flag'], default = 'None')
tweets_channel_desc = tweets.groupby(['flag_channel'])['tweet_id'].count().to_frame().reset_index().rename(columns={'tweet_id': 'count'})
# No specific channel = 5.641
# GatwickExpress = 215
# GNRailUK = 398
# SouthernRailUK = 644
# TLRailUK = 10.051 > most used channel
# barchart_chan = sns.barplot(data=tweets_channel_desc, x='flag_channel', y='count', color='#E21185')
# barchart_chan.set(xlabel='Twitter channel', ylabel='Number of tweets', title='Tweets per twitter channel')


thameslink_routes = pd.read_csv('thameslink_routes.csv')
thameslink_routes = pd.DataFrame(thameslink_routes)
thameslink_lines = pd.read_csv('thameslink_lines.csv')
thameslink_lines = pd.DataFrame(thameslink_lines)

tweets['flag_station'] = tweets['text'].apply(lambda x: next((word for word in thameslink_routes['station'] if word.lower() in x.lower()), ''))
station_count = tweets.groupby(['flag_station'])['tweet_id'].count().sort_values(ascending=False).to_frame().reset_index().rename(columns={'tweet_id': 'count'})
# barchart_chan = sns.barplot(data=station_count, x='flag_station', y='count', color='#E21185')
# barchart_chan.set(xlabel='Station', ylabel='Number of tweets', title='Tweets per station')
# Missing values = 11.734

tweets['flag_line'] = tweets['text'].apply(lambda x: next((word for word in thameslink_lines['line'] if word.lower() in x.lower()), ''))
tweets['flag_line_start'] = tweets['text'].apply(lambda x: next((word for word in thameslink_lines['start'] if word.lower() in x.lower()), ''))
tweets['flag_line_end'] = tweets['text'].apply(lambda x: next((word for word in thameslink_lines['end'] if word.lower() in x.lower()), ''))
line_count = tweets.groupby(['flag_line'])['flag_line'].count()
# Missing values = 16.941 > only 3 lines Bedford-Brighton (4), Cambridge-Brighton (2), Luton-Rainham (2)
line_count2 = tweets[tweets['flag_line_start'].str.len()>0].groupby(['flag_line_start'])['tweet_id'].count().sort_values(ascending=False).to_frame().reset_index().rename(columns={'tweet_id': 'count'}).head(5)
# barchart_c2 = sns.barplot(data=line_count2, x='flag_line_start', y='count', color='#E21185')
# barchart_2.set(xlabel='Line start', ylabel='Number of tweets', title='Tweets per line start')
# Missing values = 13.717
line_count3 = tweets[tweets['flag_line_end'].str.len()>0].groupby(['flag_line_end'])['tweet_id'].count().sort_values(ascending=False).to_frame().reset_index().rename(columns={'tweet_id': 'count'}).head(5)
# barchart_c3 = sns.barplot(data=line_count3, x='flag_line_end', y='count', color='#E21185')
# barchart_c3.set(xlabel='Line end', ylabel='Number of tweets', title='Tweets per line end')
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

# Data Preparation
# Merge location columns into one
source_col_loc = tweets.columns.get_loc('flag_station')
tweets['location'] = tweets.iloc[:, source_col_loc+1:source_col_loc+3].apply(lambda x: ','.join(x[x.str.len()>0].astype(str)), axis = 1)
# Extract location data in a separate dataframe
tweets_loc = tweets[['tweet_id', 'longitude', 'latitude', 'location']]
# Remove unwanted columns to reduce data processing time
tweets = tweets.drop(['source', 'source_id', 'language', 'relevant', 'ground_truth','user_id',
                      'longitude', 'latitude', 'location',
                      'flag_station', 'flag_line', 'flag_line', 'flag_line_start', 'flag_line_end'], axis = 1)
# Transform sentiment to two-class feature with negative and non-negative
tweets['sentiment'].replace(['negative', 'neutral', 'positive'],
                            [0, 1, 1], inplace = True)
# Rename/merge topics
tweets['topic'].replace(['air conditioning', 'announcements', 'train_general', 'station', 'tickets/seat_reservations', 'covid'],
                        ['hvac', 'service', 'service', 'service', 'service', 'service'], inplace = True)

# Convert source created_at to date
tweets['created2'] = pd.to_datetime(tweets['created'], format='mixed').apply(lambda x: x.replace(microsecond=0)).dt.date
tweets['year'] = pd.Categorical(tweets.created.dt.strftime('%Y'), ordered = True)
# Check average number of tweets per day
tweets_pd = tweets.groupby(['created2'])['tweet_id'].count().to_frame().rename(columns={'tweet_id':'count'}).reset_index()
import statistics as st
st.mean(tweets_pd['count'])
# Average number of tweets per day = 31.21
# Average number of issue tweets per day
tweets_pd_issue = tweets[tweets['topic'] != 'service'].groupby(['created2'])['tweet_id'].count().to_frame().rename(columns={'tweet_id':'count'}).reset_index()
st.mean(tweets_pd_issue['count'])
# 27.81 issue tweets per day
tweets_pdy = tweets.groupby(['year', 'created2'])['tweet_id'].count().to_frame().rename(columns={'tweet_id':'count'}).reset_index()
tweets_pdy = tweets_pdy.groupby(['year'])['count'].mean()
# Average number of tweets per day in 2019 = 21.64 and 2020 = 9.54
tweets_pdy_issue = tweets[tweets['topic'] != 'service'].groupby(['year', 'created2'])['tweet_id'].count().to_frame().rename(columns={'tweet_id':'count'}).reset_index()
tweets_pdy_issue = tweets_pdy_issue.groupby(['year'])['count'].mean()
# Average number of issue tweets per day in 2019 = 19.75 and 2020 = 8.07

import random
random.seed(450390)
sample = tweets[tweets['topic']!= 'service'].sample(15)
sample2 = tweets[tweets['topic'] == 'service'].sample(5)
samples = [sample, sample2]
sample = pd.concat(samples)
# writer = pd.ExcelWriter('sample.xlsx')
# sample.to_excel(writer,'Sheet1')
# writer.close()

# Prepare text
tweets['text_org'] = tweets['text']
tweets['text'] = tweets['text'].astype(str)
tweets['text'] = tweets['text'].str.lower()
tweets['text'] = tweets['text'].replace(r'\W+', ' ', regex=True)
tweets['text'] = tweets['text'].replace(r'\d', ' ', regex=True)
# nltk.download('averaged_perceptron_tagger')
tweets['tokens'] = tweets.apply(lambda row: nltk.word_tokenize(row['text']), axis=1)
tweets['tokens_tagged'] = tweets['tokens'].apply(nltk.pos_tag)

# Function to update POS tags in each list
def update_pos_tags(row):
    lemmatizer = WordNetLemmatizer()
    updated_list = [(token, get_updated_pos(pos)) for token, pos in row]
    return updated_list

# Function to map POS tags to accepted values for lemmatize function
def get_updated_pos(pos):
    if pos.startswith('N'):
        return 'n'  # Noun
    elif pos.startswith('V'):
        return 'v'  # Verb
    elif pos.startswith('R'):
        return 'r'  # Adverb
    elif pos.startswith('J'):
        return 'a'  # Adjective
    else:
        return None  # Return None for other cases

# Apply the update_pos_tags function to the 'tokens_tagged' column
tweets['tokens_tagged'] = tweets['tokens_tagged'].apply(update_pos_tags)

def lemmatize_tokens(tokens_tagged):
    lemmatizer = WordNetLemmatizer()
    lemmatized_list = [(lemmatizer.lemmatize(token, pos=pos_tag) if pos_tag is not None else token) for token, pos_tag in tokens_tagged]
    return lemmatized_list

# Apply the lemmatize_tokens function to the 'token_pos_list' column
tweets['tokens_lem'] = tweets['tokens_tagged'].apply(lemmatize_tokens)
def remove_stopwords(tokens_lem):
    sw = nltk.corpus.stopwords.words('english')
    filtered_tokens = [token for token in tokens_lem if token not in sw]
    return filtered_tokens
tweets['filtered_tokens'] = tweets['tokens_lem'].apply(remove_stopwords)
tweets = tweets.drop(['text', 'tokens', 'tokens_tagged', 'tokens_lem'], axis = 1)

def tokens_to_string(filtered_tokens):
    return ' '.join([token for token in filtered_tokens])
tweets['text'] = tweets['filtered_tokens'].apply(tokens_to_string)

# Function to measure the performance of a model to show accuracy score, confusion matrix and ROC-AUC Curve
def model_Evaluate(model, test_data):
    # Predict values for Test dataset
    y_pred = model.predict(test_data)
    # Print the evaluation metrics for the dataset.
    print(classification_report(sentiments_test, y_pred))
    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(sentiments_test, y_pred)
    categories = ['Negative','Positive']
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f'{v1}n{v2}' for v1, v2 in zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '',
                xticklabels = categories, yticklabels = categories)
    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values" , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)

# Generate action recommendations
# Set aside 20% of the data for KPI generation
random.seed(54992)
# Setting the seed for numpy-generated random numbers
np.random.seed(394)
kpi_data = tweets.sample(round(0.2*len(tweets)))
kpi_data = kpi_data.drop(columns=['topic', 'sentiment', 'id_topic', 'id_sentiment'])
# 3390 observations for KPI generation
# Remove the KPI data from tweets
index_red = kpi_data.index
tweets_red = tweets.drop(index_red)

# Split data for topic classification
topics = tweets_red['topic']
texts = tweets_red['text']
kpi_texts = kpi_data['text']
# Vectorize texts into numeric values
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df = 10)
vecs = vectorizer.fit_transform(texts)
num_features = vecs.shape[1]
# New data has to have the same number of features as training data
vectorizer_new = TfidfVectorizer(vocabulary = None, max_features = num_features)
kpi_vecs = vectorizer_new.fit_transform(kpi_texts)
# Split data into training and testing
# Setting the seed for python random numbers
random.seed(13747)
# Setting the seed for numpy-generated random numbers
np.random.seed(37)
vec_train, vec_test, topics_train, topics_test = train_test_split(vecs, topics, test_size = 0.2, random_state= 42, stratify=topics)

# SVM
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
# Model building
model_svc = LinearSVC()
# Define the parameter grid to search
param_grid = {'C': [0.1, 1, 10, 100], 'penalty': ['l2'], 'dual':[False]}
# Initialize GridSearchCV
grid_search_svc = GridSearchCV(model_svc, param_grid, cv=5, scoring='accuracy')
# Perform grid search
grid_search_svc.fit(vec_train, topics_train)
# Get the best parameters
best_params = grid_search_svc.best_params_
# Get the best model
best_model_svc = grid_search_svc.best_estimator_
# Training the model with the training data
best_model_svc.fit(vec_train, topics_train)
topics_pred_svc = best_model_svc.predict(vec_test)
print(classification_report(topics_test, topics_pred_svc))

# Predict KPI data
kpi_topics_pred = best_model_svc.predict(kpi_vecs)
kpi_topics_pred = pd.DataFrame(kpi_topics_pred, columns = ['topic_pred'])

# Split data for sentiment analysis
sentiments = tweets_red['sentiment']
# Split data into training and testing
vec_train2, vec_test2, sentiments_train, sentiments_test = train_test_split(vecs, sentiments, test_size = 0.2, random_state= 42, stratify=sentiments)
# Define evaluation
from sklearn.svm import SVC     
SVCmodel = SVC(kernel='poly')
SVCmodel.fit(vec_train2, sentiments_train)
model_Evaluate(SVCmodel, vec_test2)
sentiments_pred2 = SVCmodel.predict(vec_test2)
# Precision: 0.75, Recall: 0.73, F1 Score: 0.69


# Define the parameter grid to search
param_grid = {'C': [0.1, 1, 10, 100]}
SVCmodel = SVC(kernel='poly')
# Initialize GridSearchCV
grid_search_svc = GridSearchCV(SVCmodel, param_grid, cv=5, scoring='accuracy')
# Perform grid search
grid_search_svc.fit(vec_train2, sentiments_train)
# Get the best parameters
best_params = grid_search_svc.best_params_
print("Best Parameters:", grid_search_svc.best_params_)
# Best Parameters: {'C': 10}
# Get the best model
best_model_svc = grid_search_svc.best_estimator_

# Training the model with the training data
best_model_svc.fit(vec_train2, sentiments_train)
model_Evaluate(best_model_svc, vec_test2)
sentiments_pred2 = best_model_svc.predict(vec_test2)
# Precision: 0.74, Recall: 0.73, F1 Score: 0.71
# True negatives: 58.59%
# False positive: 4.06%

# ROC-AUC Curve for Model-2
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(sentiments_test, sentiments_pred2)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE')
plt.legend(loc="lower right")
plt.show()
# ROC Curve Area: 0.64


# Predict KPI data
kpi_sentiment_pred = best_model_svc.predict(kpi_vecs)
kpi_sentiment_pred = pd.DataFrame(kpi_sentiment_pred, columns = ['sentiment_pred'])

# Merge results of sentiment analysis and topic analysis
results = pd.merge(left = kpi_sentiment_pred, left_index = True,
                   right = kpi_topics_pred, right_index = True,
                   how = 'inner')
results.index = index_red
result_loc = tweets_loc.loc[index_red][['location','longitude','latitude']]
dfs = [results, result_loc, kpi_data]
from functools import reduce
results_all =  reduce(lambda left,right: pd.merge(left,right,left_index = True, right_index = True), dfs)

# Extract maintenance issues
issues = results_all[results_all['sentiment_pred']==0]
issues_main = issues[(issues['topic_pred'] != 'service') & (issues['topic_pred'] != 'none')]

# Check for duplicates
dups = issues_main.drop(columns=['filtered_tokens'], axis = 1).duplicated('text_org')
# 46 duplicates
dups = issues_main[dups]
# 1. Check for non-verified issues
issues_main_ver = issues_main[issues_main['topic_pred'] != 'delays']
len(issues_main_ver)
# 82 issues total
# Filter out non-Thameslink twitter channels
issues_main_ver = issues_main_ver[issues_main_ver['flag_channel'] == '@TLRailUK']
# 51 Thameslink issues

# 2. Check for verified issues (# of issue > 1 per day)
issues_main_ver1 = issues_main[(issues_main['topic_pred'] != 'delays')].groupby(['created2', 'topic_pred'])['topic_pred'].count().to_frame().rename(columns={'topic_pred':'count'})
issues_main_ver1[issues_main_ver1['count']>1]
# 6 issues
# 3 of them wifi
# Filter out non-Thameslink twitter channels
issues_main_ver1 = issues_main[(issues_main['topic_pred'] != 'delays')&(issues_main['flag_channel']=='@TLRailUK')].groupby(['created2', 'topic_pred'])['topic_pred'].count().to_frame().rename(columns={'topic_pred':'count'})
issues_main_ver1[issues_main_ver1['count']>1]

# 3. Check verified issues based on combination of issue and delay per day
issues_main_ver2 = issues_main.groupby(['created2','topic_pred','location'])['topic_pred'].count().to_frame().rename(columns={'topic_pred':'count'}).reset_index()
issues_main_ver2 = issues_main_ver2[issues_main_ver2.groupby(['created2', 'location'])['topic_pred'].transform('count') > 1]
# Multiple occurrences of issues in combination with delays
issues_main_ver2_count = issues_main_ver2.pivot(index=['created2', 'location'], columns='topic_pred', values='count').fillna(0).astype(int)
issues_main_ver2_count = issues_main_ver2_count[issues_main_ver2_count['delays']>0]
issues_main_ver2_count['combination'] = issues_main_ver2_count.apply(lambda row: ', '.join(col for col in issues_main_ver2_count.columns if row[col] > 0), axis=1)
len(issues_main_ver2_count)
# Merge verified issues with rest of data
merged = pd.merge(issues_main_ver2, issues_main, on = ['created2', 'topic_pred'], how = 'inner')
merged = merged.drop(columns=['location_x','count'], axis = 1)
# Filter our delays
merged = merged[merged['topic_pred']!= 'delays']
# Filter out non-Thameslink twitter channels
merged = merged[merged['flag_channel'] == '@TLRailUK']
len(merged)
# Total of 45 verified issues
len(merged[(merged['location_y'].str.len() > 0) |
       ((merged['longitude'].notnull()) & (merged['latitude'].notnull()))])
# 4 with location or coordinates

# KPIs
# Number of issues per day on average
issues_pd = issues.groupby(['created2'])['topic_pred'].count().to_frame().rename(columns={'topic_pred':'count'}).reset_index()
st.mean(issues_pd['count'])
# Average number of issues = 7.37

# Number of maintenance issues per day on average
issues_main_pd = issues_main.groupby(['created2'])['topic_pred'].count().to_frame().rename(columns={'topic_pred':'count'}).reset_index()
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
import matplotlib.dates as mdates
barchart_im = sns.barplot(data=issues_main_pd, x='created2', y='count', color='#E21185')
barchart_im.set(xlabel='Date created', ylabel='Number of issues', title='Number of maintenance issues per day')
# show every 12th tick on x axes
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=45, fontweight='light',  fontsize='x-small')
plt.rcParams["figure.dpi"] = 500
plt.show()
st.mean(issues_main_pd['count'])
# Average number of maintenance issues = 5.23

# Number of verified maintenance issues per day on average
# First verification method (number of specific issue > 1 per day)
st.mean(issues_main_ver1['count'])
# Average number of verified issues = 1.04
# However small amount of data, no conclusion possible
# Second verification method (delay + issue)
issues_ver_pd = merged.groupby(['created2',['topic_pred']])['topic_pred'].count().to_frame().rename(columns={'topic_pred':'count'}).reset_index()
barchart_imv = sns.barplot(data=issues_ver_pd, x='created2', y='count', color='#E21185')
barchart_imv.set(xlabel='Date created', ylabel='Number of issues', title='Number of verified maintenance issues per day')
# show every 12th tick on x axes
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval = 2))
plt.xticks(rotation=45, fontweight='light',  fontsize='x-small')
plt.rcParams["figure.dpi"] = 500
plt.show()
st.mean(issues_ver_pd['count'])
# Average number of verified issues = 1.13

# Reoccurring issues
# General issues
issues_rec_pd = issues.groupby(['topic_pred'])['topic_pred'].count().to_frame().rename(columns={'topic_pred':'count'}).reset_index()
issues_rec_pd.sort_values(by = 'count', ascending=False)
# Most recurrent issues are delays (1.995), none (708), service (593), wifi (27) and hvac (21)

# Maintenance issues
issues_main_rec_pd = issues_main[issues_main['topic_pred']!= "delays"].groupby(['topic_pred'])['topic_pred'].count().to_frame().rename(columns={'topic_pred':'count'}).reset_index()
issues_main_rec_pd.sort_values(by = 'count', ascending=False)
# Most recurrent issues are wifi (27), hvac (21), brakes (11), vandalism (7) and plugs (6)
barchart_imrec = sns.barplot(data=issues_main_rec_pd, x='topic_pred', y='count', color='#E21185')
barchart_imrec.set(xlabel='Maintenance topics', ylabel='Number of issues', title='Number of reoccuring issues per day')
# show every 12th tick on x axes
plt.xticks(rotation=45)
plt.rcParams["figure.dpi"] = 500
plt.show()

# Verified maintenance issues
# First verification method
issues_main_ver1 = issues_main[(issues_main['topic_pred'] != 'delays')].groupby(['created2', 'topic_pred'])['topic_pred'].count().to_frame().rename(columns={'topic_pred':'count'})
issues_main_ver1[issues_main_ver1['count']>1]
# Most recurrent issues are wifi (16), hvac (4), toilets (2)

# Second verification method
issues_rec_ver2 = merged.groupby(['topic_pred'])['topic_pred'].count().to_frame().rename(columns={'topic_pred':'count'}).reset_index()
issues_rec_ver2.sort_values(by = 'count', ascending=False)
# Most recurrent issues are hvac (12), brakes (9), wifi (9), plugs (5) and vandalism (4)
barchart_imrecver = sns.barplot(data=issues_rec_ver2, x='topic_pred', y='count', color='#E21185')
barchart_imrecver.set(xlabel='Maintenance topics', ylabel='Number of issues', title='Number of reoccuring verified issues per day')
# show every 12th tick on x axes
plt.xticks(rotation=45)
plt.rcParams["figure.dpi"] = 500
plt.show()
# Number of duplicated reports
issues_dub = dups.groupby(['topic_pred', 'text'])['topic_pred'].count().to_frame().rename(columns={'topic_pred':'count'}).reset_index()
issues_dub[issues_dub['count']>1].sort_values(by = 'count', ascending=False)
# Only duplicates concern delays
# Number of reporters
issues_auth = issues_main.groupby(['author_id'])['author_id'].count().to_frame().rename(columns={'author_id':'count'}).reset_index()
len(issues_auth)
# 1340 reporters
# Number of reporters per author per day
issues_auth_pd = issues_main.groupby(['created2','author_id'])['tweet_id'].count().to_frame().rename(columns={'tweet_id':'count'}).reset_index()
issues_auth_pd['bins'] = np.digitize(issues_auth_pd['count'], bins = [2,3,4,5,6])
auth_bincount = np.bincount(issues_auth_pd['bins'])
bins = ['1', '2', '3', '4', '5','>5']
barplot_auth_all = sns.barplot(x = bins, y = auth_bincount, color='#E21185')
barplot_auth_all.set(xlabel='Number of reports per author', ylabel='Count of number of reports', title='Count of the number of reports per author')
barplot_auth_all.bar_label(barplot_auth_all.containers[0])
# Reporters who report more than once
issues_auth_pd[issues_auth_pd['count']>1]
# 170 authors reported more than once
issues_auth_pd[issues_auth_pd['count']>1].sort_values(by='count', ascending = False)
st.mean(issues_auth_pd['count'])
# On average authors tweet about 1.16 times
# Number of reporters per week
issues_main['week'] = pd.Categorical(issues_main.created.dt.isocalendar().week, ordered = True)
issues_auth_wy = issues_main.groupby(['year', 'week'])['author_id'].count().to_frame().rename(columns={'author_id':'count'}).reset_index()
barchart_auth_wy = sns.barplot(data=issues_auth_wy, x='week', y='count', hue='year', palette='dark:#E21185')
barchart_auth_wy.set(xlabel='Week created', ylabel='Number of reporters', title='Number of reporters per week')
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval = 3))
plt.xticks(rotation=45)
st.mean(issues_auth_wy['count'])
# Average number of reports ber week = 20.35
weekdays = ["Monday", "Tuesday", "Wednesday","Thursday", "Friday","Saturday","Sunday"]
issues_main['weekday'] = pd.Categorical(issues_main.created.dt.strftime('%A'), categories = weekdays, ordered = True)
issues_auth_wd = issues_main.groupby(['year', 'weekday'])['author_id'].count().to_frame().rename(columns={'author_id':'count'}).reset_index()
barchart_auth_wd = sns.barplot(data=issues_auth_wd, x='weekday', y='count', hue='year', palette='dark:#E21185')
barchart_auth_wd.set(xlabel='Weekday created', ylabel='Number of reporters', title='Number of reporters per weekday')
plt.xticks(rotation=20)
st.mean(issues_auth_wy['count'])

# # Generate sample
# # Setting the seed for python random numbers
# random.seed(54992)
# # Setting the seed for numpy-generated random numbers
# np.random.seed(394)
# issues_sam = issues_main.sample(20)

# # Setup openai
# from openai import OpenAI
# client = OpenAI(
#     # defaults to os.environ.get("OPENAI_API_KEY")
#     api_key="your-api-key")
# prompt_action = [f"--Input--\nText: {issues_sam['text_org']}"
#                  f"\nTopic: {issues_sam['topic_pred']}"
#                  f"\nDate: {issues_sam['created']}"
#                  f"\nLongitude: {issues_sam['longitude']}"
#                  f"\nLatitude: {issues_sam['latitude']}"
#                  f"\nLocation: {issues_sam['location']}"
#                  "\n--Output--\n\nTicket for Maintenance Staff:"
#                  "\n\nSummary: [Brief summary of the issue with context]"
#                  "\n\n --End--" for i in issues_sam]

# # Get response
# response = client.completions.create(
#     model="gpt-3.5-turbo-instruct",
#     prompt=prompt_action,
#     temperature=0.2,
#     stop=None,
#     max_tokens = 10000,
# )
# # Format response
# # Topic
# response_choices = pd.DataFrame(response.choices, columns=[['col1', 'col2', 'col3', 'action']])
# response_choices['action'] = response_choices['action'].astype(str)
# print(response_choices['action'])


