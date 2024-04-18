#%%
import pandas as pd 
from dotenv import load_dotenv
import os

from tweets_to_topic_network.data import Data_processor 
from tweets_to_topic_network.topic import Topic_modeler
from tweets_to_topic_network.network import Network_creator

n_cop = 'cop22'

file_tweets = '/Users/alessiogandelli/data/' + n_cop + '/' + n_cop + '.json'
file_user = '/Users/alessiogandelli/data/' + n_cop + '/users_'+ n_cop+'.json'

# file_tweets = '/Users/alessiogandelli/dev/uni/tweets-to-topic-network/data/toy.json'
# file_user = '/Users/alessiogandelli/dev/uni/tweets-to-topic-network/data/toy_users.json'

#%%
data = Data_processor(file_tweets=file_tweets, file_user=file_user, n_cop='22')
data.process_json()

tm = Topic_modeler(data.df_original, name = data.name + 'bai', embedder_name='BAAI/bge-base-en-v1.5', path_cache = data.path_cache)
df_labeled = tm.get_topics()

#%%

tm = Topic_modeler(data.df_original, name = data.name + 'multilingual_paraphrase', embedder_name='paraphrase-multilingual-MiniLM-L12-v2', path_cache = data.path_cache)
df_labeled = tm.get_topics()






#%%

tm.model.visualize_topics()



#%%

tm.model.visualize_documents(tm.df['text'], embeddings=tm.embeddings)

























# %%
