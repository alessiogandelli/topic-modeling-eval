
# # %%
# from flask import Flask, render_template, request, jsonify,url_for
# import pandas as pd
# import ast
# import csv   
# import os




# path ='/Users/alessiogandelli/data/cop22/cache/tm/all-MiniLM-L6-v2'

# human_labeled = path + '/human_labeled.csv'

# #create file
# if not os.path.exists(human_labeled):
#     with open(human_labeled, 'w') as f:
#         writer = csv.writer(f)
#         writer.writerow(['tweet_id', 'label'])


# def get_data(path):

#     df = pd.read_csv(path+'/topics_cop22.csv' )

#     #df.set_index('Topic', inplace=True)

#     #remove unamed column
#     df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

#     #rename representative docs to docs 
#     df = df.rename(columns={'Representative_Docs': 'docs'})

#     # Convert DataFrame to a dictionary where each row is a key-value pair
#     # with the first column as the key
#     topic_data = {row['Topic']: row.drop('Topic').to_dict() for _, row in df.iterrows()}

#     for topic in topic_data:
#         topic_data[topic]['docs'] = ast.literal_eval(topic_data[topic]['docs'])

#     return topic_data, df

# def get_random_tweet(topic):
#     # should not be already labeled 
#     topic = int(topic)

#     try: 
#         print(topic, type(topic))
#         tweets = df_tweets[df_tweets['topic'] == topic]
#         tweets = tweets[~tweets['reviewed']]
#         tweet = tweets.sample(1)
#         tweet = { 'id' : tweet.index.tolist()[0], 'text': tweet['text'].values[0] }
#     except ValueError:
#         tweet = {'id': None, 'text': None}

#     print(tweet)

#     return tweet




# df_tweets = pd.read_pickle(path + '/tweets_cop22_labeled.pkl')
# df_tweets['reviewed'] = False
# df_tweets['label'] = None

# topic_data,  df_topic= get_data(path)

# #%%
# app = Flask(__name__)


# @app.route('/')
# def index():
#     # Get the list of topics
#         # Get the list of topics
#     global path 


#     topic_data, df_topic = get_data(path)


#     topics = list(topic_data.keys())
#     print(topics)
#     return render_template('index.html', topics=topics, url_for=url_for)

# @app.route('/get_topics', methods=['GET'])
# def get_topics():
#     df_topic.set_index('Topic', inplace=True)
#     topics = df_topic['Name'].to_dict()
#     return jsonify(topics)

# @app.route('/get_topic_data', methods=['GET'])
# def get_topic_data():

#     topic = request.args.get('topic')
#     global_topic_data = topic_data.get(int(topic)) # Renamed to avoid naming conflict
#     return jsonify(global_topic_data)

# @app.route('/get_tweet', methods=['GET'])
# def get_tweet():
#     topic = request.args.get('topic')
#     print('topic', topic)
#     tweets = get_random_tweet(topic)
#     return jsonify(tweets)

# @app.route('/label_tweet', methods=['POST'])
# def label_tweet():
#     tweet_id = request.json.get('tweet')
#     label = request.json.get('label')
    
#     # Find the tweet in df_tweets and update its label
    
#     #append to a csv file tweet_id, label
#     df_tweets.loc[tweet_id, 'reviewed'] = True

#     with open(human_labeled, 'a') as f:
#         writer = csv.writer(f)
#         writer.writerow([tweet_id, label])

#     print('Tweet labeled:', tweet_id, label)


#     return jsonify({'status': 'success'})

# if __name__ == '__main__':
    # app.run(debug=True)
# %%
import os
import csv
import ast
import pandas as pd
from flask import Flask, render_template, request, jsonify, url_for
import logging

logging.basicConfig(level=logging.INFO)

path ='/Users/alessiogandelli/data/cop22/cache/tm/all-MiniLM-L6-v2'
human_labeled = os.path.join(path, 'human_labeled.csv')

if not os.path.exists(human_labeled):
    with open(human_labeled, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['tweet_id', 'label'])

def get_data(path):
    """Reads data from a CSV file and returns a dictionary and a DataFrame."""
    df = pd.read_csv(os.path.join(path, 'topics_cop22.csv'))
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.rename(columns={'Representative_Docs': 'docs'})
    topic_data = {row['Topic']: row.drop('Topic').to_dict() for _, row in df.iterrows()}

    for topic in topic_data:
        topic_data[topic]['docs'] = ast.literal_eval(topic_data[topic]['docs'])

    return topic_data, df

def get_random_tweet(df_tweets, topic):
    if topic is None:
        return {'id': None, 'text': None}
    try: 
        tweets = df_tweets[df_tweets['topic'] == int(topic)]
        tweets = tweets[~tweets['reviewed']]
        tweet = tweets.sample(1)
        return {'id' : tweet.index.tolist()[0], 'text': tweet['text'].values[0]}
    except ValueError:
        return {'id': None, 'text': None}

df_tweets = pd.read_pickle(os.path.join(path, 'tweets_cop22_labeled.pkl'))
df_tweets['reviewed'] = False
df_tweets['label'] = None

topic_data, df_topic = get_data(path)

app = Flask(__name__)

@app.route('/')
def index():
    """Renders the index page with a list of topics."""
    topics = list(topic_data.keys())
    return render_template('index.html', topics=topics)

@app.route('/get_topics', methods=['GET'])
def get_topics():
    """Returns a JSON object containing all topics."""
    df_topic_copy = df_topic.copy()
    df_topic_copy.set_index('Topic', inplace=True)
    return jsonify(df_topic_copy['Name'].to_dict())

@app.route('/get_topic_data', methods=['GET'])
def get_topic_data():
    """Returns a JSON object containing data for a specific topic."""
    topic = request.args.get('topic')
    return jsonify(topic_data.get(int(topic)))

@app.route('/get_tweet', methods=['GET'])
def get_tweet():
    topic = request.args.get('topic')
    if topic is None:
        return jsonify({'error': 'The "topic" parameter is required.'}), 400
    return jsonify(get_random_tweet(df_tweets, topic))

@app.route('/label_tweet', methods=['POST'])
def label_tweet():
    """Updates the status of a tweet to 'reviewed' and appends the tweet's ID and label to a CSV file."""
    tweet_id = request.json.get('tweet')
    label = request.json.get('label')
    df_tweets.loc[tweet_id, 'reviewed'] = True

    with open(human_labeled, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([tweet_id, label])

    logging.info('Tweet labeled: %s, %s', tweet_id, label)
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)