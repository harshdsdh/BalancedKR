from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from numpy import dot
from numpy.linalg import norm 
import gensim.downloader as api
from gensim.models import Word2Vec
import pandas as pd

df = pd.read_csv('../main_df_1.csv')

temp_data = [s.split(' ') for s in df['text']]
temp_data

emb = Word2Vec(temp_data, min_count=1)
# summarize the loaded model
print(emb)
# summarize vocabulary
words = list(emb.wv.vocab)
print(words)
# access vector for one word
print(emb['riots'])
# save model
emb.save('model_1.bin')
# load model
emb = Word2Vec.load('model_1.bin')
print(emb)