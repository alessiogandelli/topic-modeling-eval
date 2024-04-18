
# %%
from flask import Flask, render_template, request, jsonify,url_for
import pandas as pd
import ast


app = Flask(__name__)

path ='/Users/alessiogandelli/data/cop22/cache/tm/BAAI/bge-base-en-v1.5'



def get_topic_data(path):

    df = pd.read_csv(path+'/topics_cop22.csv' )

    #df.set_index('Topic', inplace=True)

    #remove unamed column
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    #rename representative docs to docs 
    df = df.rename(columns={'Representative_Docs': 'docs'})

    # Convert DataFrame to a dictionary where each row is a key-value pair
    # with the first column as the key
    topic_data = {row['Topic']: row.drop('Topic').to_dict() for _, row in df.iterrows()}

    for topic in topic_data:
        topic_data[topic]['docs'] = ast.literal_eval(topic_data[topic]['docs'])

    return topic_data, df

def get_random_tweet(topic):
    # should not be already labeled 
    topic = int(topic)

    try: 
        print(topic, type(topic))
        tweets = df_tweets[df_tweets['topic'] == topic]
        tweets = tweets[~tweets['reviewed']]
        tweet = tweets.sample(1)
        tweet = { 'id' : tweet.index.tolist()[0], 'text': tweet['text'].values[0] }
    except ValueError:
        tweet = {'id': None, 'text': None}

    return tweet




df_tweets = pd.read_pickle(path + '/tweets_cop22_labeled.pkl')

df_tweets['reviewed'] = False
df_tweets['label'] = None

#%%


from flask import Flask, g
import pandas as pd



app = Flask(__name__)


topic_data,  df_topic= get_topic_data(path)

@app.route('/')
def index():
    # Get the list of topics
    topics = list(topic_data.keys())
    print(topics)
    return render_template('index.html', topics=topics, url_for=url_for)

@app.route('/get_topics', methods=['GET'])
def get_topics():
    df_topic.set_index('Topic', inplace=True)
    topics = df_topic['Name'].to_dict()
    return jsonify(topics)

@app.route('/get_topic_data', methods=['GET'])
def get_topic_data():

    topic = request.args.get('topic')
    global_topic_data = topic_data.get(int(topic)) # Renamed to avoid naming conflict
    return jsonify(global_topic_data)

@app.route('/get_tweet', methods=['GET'])
def get_tweet():
    topic = request.args.get('topic')
    tweets = get_random_tweet(topic)
    return jsonify(tweets)

@app.route('/label_tweet', methods=['POST'])
def label_tweet():
    tweet_id = request.json.get('tweet')
    label = request.json.get('label')
    
    # Find the tweet in df_tweets and update its label
    #df_tweets.loc[794242273462616066, 'label'] = label

    print('Tweet labeled:', tweet_id, label)

    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)
# %%
