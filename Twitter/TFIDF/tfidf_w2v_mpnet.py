# glove emb+twitter data
import pandas as pd
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
nltk.download("wordnet")
from nltk.stem import WordNetLemmatizer

import numpy as np

# Get the interactive Tools for Matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')


from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from numpy import dot
from numpy.linalg import norm
from gensim import models
from gensim import similarities
from gensim.corpora import Dictionary


emb_word2vec = KeyedVectors.load("model.bin", mmap='r')
from sentence_transformers import SentenceTransformer

emb = np.load('embeddings_mpnet_base_v2.npy')
#model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import matplotlib.pyplot as plt

# load processed data
main_df = pd.read_csv('completeTwitterData2023.csv')
democrats_df1 =  pd.read_csv('completeTwitterLeftData2023.csv')
republicans_df1 = pd.read_csv('completeTwitterRightData2023.csv')

main_df = main_df.dropna(subset = ['text']).reset_index(drop=True)
democrats_df1 = democrats_df1.dropna(subset = ['text']).reset_index(drop=True)
republicans_df1 = republicans_df1.dropna(subset = ['text']).reset_index(drop=True)

dictionary = Dictionary.load('2023_dictionary.dict')
dems_dictionary = Dictionary.load('2023_dems_dictionary.dict')
reps_dictionary = Dictionary.load('2023_reps_dictionary.dict')

tfidf_vectorizer = TfidfVectorizer()
arr = [line for line in main_df['text']]
tfidf_weights_matrix = tfidf_vectorizer.fit_transform(arr)

#query pre processing
def cutom_tokenizer(s):
    s = s.lower()
    s = s.translate(str.maketrans("","", string.punctuation))
    tokens = nltk.word_tokenize(s)
    remove_stopwords = list(filter(lambda token: token not in stopwords.words("english"),tokens))
    tokenize_words = [word for word in remove_stopwords]
    return tokenize_words

def preprocessed_query(orig_query):
    temp = cutom_tokenizer(orig_query)
    query = " ".join(temp)
    return [temp, query]

# get set of recommended keywords
def get_recommended_queries(query):
    if query in emb_word2vec:
        ans = emb_word2vec.most_similar(query)[:10]
        return ans

def cfs(word):
    if word in dems_dictionary.token2id or word in reps_dictionary.token2id: 
        return True
    return False

def get_recommended_set(query_arr, query_str):
    recommended_set = set()
    for word in query_arr:
        if word in emb_word2vec:
            set_words = get_recommended_queries(word)
            list_words = [key for key,dist in set_words]
            for w in list_words:
                if cfs(w):
                    recommended_set.add(w)
    recommended_set.add(query_str)
    return recommended_set

def analyze(main_df, sim):
    query_results = set()
    for sent in sim:
        politics  = main_df.loc[sent]['bias']
        query_results.add((sent, politics))
    return query_results

def tf_idf(search_keys):
    search_query_weights = tfidf_vectorizer.transform([search_keys])
    return search_query_weights

def cos_similarity(search_query_weights, tfidf_weights_matrix):
    cosine_distance = cosine_similarity(search_query_weights, tfidf_weights_matrix)
    similarity_list = cosine_distance[0]
    return similarity_list

def most_similar(similarity_list, min_talks=1):

    most_similar= []
  
    while min_talks > 0:
        tmp_index = np.argmax(similarity_list)
        if tmp_index not in most_similar and tmp_index!=0:
            most_similar.append(tmp_index)
            similarity_list[tmp_index] = 0
        min_talks -= 1

    return most_similar

def cosine_search_engine(main_df, query):
    search_query_weights = tfidf_vectorizer.transform([query])
    l = cos_similarity(search_query_weights, tfidf_weights_matrix)
    result = most_similar(l,20)    
    
    return result

# get scores

def get_precision(orig_index, recommended_word_indexes):
    precision_set = []
    for doc_index in recommended_word_indexes:
        #sent1 = [w for w in doc.split(' ') if w in emb_glove.vocab]
        max_precision_val = 0
        for orig_doc_index in orig_index:
            #sent = [w for w in orig_doc.split(' ') if w in emb_glove.vocab]
            if doc_index and orig_doc_index:
                max_precision_val = max(max_precision_val, cosine_similarity([emb[doc_index]], [emb[orig_doc_index]])[0][0])
        precision_set.append(max_precision_val)
    #precision_set = precision_set.flatten()
    if len(precision_set):    
        avg_precision = round(sum(precision_set)/len(precision_set),2)
        return avg_precision
    else:
        return 0

def get_recall(orig_index, recommended_word_indexes):
    recall_set = []
    for orig_doc_index in orig_index:
        max_recall_val = 0
        for doc_index in recommended_word_indexes:
            if orig_doc_index and doc_index:
                max_recall_val = max(max_recall_val, cosine_similarity([emb[orig_doc_index]], [emb[doc_index]])[0][0])
        recall_set.append(max_recall_val)
    if len(recall_set):
        avg_recall = round(sum(recall_set)/len(recall_set),2)
        return avg_recall
    else:
        return 0

def get_bias(docs_bias_set):
    val_bias = [val for key,val in docs_bias_set]
    return round(sum(val_bias)/len(val_bias),2)

def get_scores(main_df, recommended_set, orig_query_str, search_engine_result_indexes):
    result_indexes = cosine_search_engine(main_df, orig_query_str) 
    orig_docsIndex_bias = analyze(main_df,search_engine_result_indexes)
    precision = 0
    recall = 0
    bias = 0
    measure = 0
    metrics_obj = {}
    for word in recommended_set:
        recommended_set_indexes = cosine_search_engine(main_df, word)
        if len(recommended_set_indexes):
            recommended_docsIndex_bias = analyze(main_df,recommended_set_indexes)
            precision = get_precision(result_indexes, recommended_set_indexes)
            recall = get_recall(result_indexes, recommended_set_indexes)
            bias = get_bias(recommended_docsIndex_bias)
            if precision!=0 or recall!=0:
                relevance = round((2*precision*recall/(precision+recall)),2)
            else:
                relevance=0
            metrics_obj[word] = {
                'precision':precision,
                'recall': recall,
                'bias': bias,
                'relevance': relevance
            }
    return metrics_obj

# create pareto front
def get_rel_bias_scores(m):
    scores_arr =[]
    for key, val  in m.items():
        scores_arr.append([val['relevance'], val['bias']])
    return  scores_arr

def build_paretofront(sorted_list,bias,maxY=True):
    pareto_front = [sorted_list[0]]
    for pair in sorted_list[1:]:
        if maxY:
            if pair[1] > pareto_front[-1][1] and pair not in pareto_front and abs(pair[0])<=abs(bias):
                pareto_front.append(pair)
        else:
            if pair[1] <= pareto_front[-1][1] and pair not in pareto_front:
                pareto_front.append(pair)
    return pareto_front

def add_to_excel(m, query_str, filename):
    word_graph = []
    res_arr_graph = []
    bias_arr_graph = []
    precision_graph = []
    recall_graph = [] 
    original_bias = float('-inf')
    for k, v in m.items():
        if k == query_str:
            k ='query: '+ k
            original_bias = v['bias']
        word_graph.append(k)
        res_arr_graph.append(v['relevance'])
        bias_arr_graph.append(v['bias'])
        precision_graph.append(v['precision'])
        recall_graph.append(v['recall'])
    graph_data={}
    graph_data['word'] = word_graph
    graph_data['bias']=bias_arr_graph
    graph_data['relevance'] = res_arr_graph
    graph_data['precision'] = precision_graph
    graph_data['recall'] = recall_graph
    graph_data = pd.DataFrame(graph_data)
    graph_data.to_csv(filename+'.csv',mode='a', index=False, header=False)
    graph_data.to_csv(filename+'Headers.csv',mode='a', index=False, header=True)
    return original_bias
    
    
def get_pareto_result(Xs, Ys, original_bias,m, maxX=True, maxY=True):
    sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))])
    positive_bias=[]
    negative_bias=[]
    min_rel=float('inf')
    for bias,rel in sorted_list:
        if bias>=0:
            positive_bias.append([bias, rel])
        if bias<0:
            negative_bias.append([bias, rel])
        min_rel = min(min_rel, rel)

    negative_bias.sort(key= lambda x:x[0], reverse=True)
    if len(positive_bias)>0:
        pf_positive_bias = build_paretofront(positive_bias,original_bias, True)
    else:
        pf_positive_bias=[]
    if len(negative_bias):
        pf_negative_bias  = build_paretofront(negative_bias,original_bias, True)
    else:
        pf_negative_bias=[]
    
    
    pareto_front = pf_positive_bias+pf_negative_bias
    ans = {}
    for s in pareto_front:
        for k, v in m.items():
            if s==[v['bias'], v['relevance']]:
                ans[k] = {
                'bias' : v['bias'],
                'relevance' : v['relevance'],
                'precision' : v['precision'],
                'recall' : v['recall']
                }
                
    return ans

def main():
    q_s = pd.read_excel('queries.xlsx')
    set_queries = set()
    for row in range(len(q_s)):
        if q_s.loc[row]['FairKR relevant']==1:
            set_queries.add(q_s.loc[row]['query'])
    
    for row in set_queries:
        query_arr, query_str = preprocessed_query(row)
        recommended_set = get_recommended_set(query_arr, query_str)
        
        search_engine_result_indexes = cosine_search_engine(main_df, query_str)
        search_engine_result_sents = analyze(main_df,search_engine_result_indexes)
        metrics = get_scores(main_df, recommended_set, query_str, search_engine_result_indexes)
        
        scores = get_rel_bias_scores(metrics)
        X = [i[1] for i in scores]
        Y=[i[0] for i in scores]
        original_bias = add_to_excel(metrics, query_str, 'tfidf_w2v_mpnet')
        
        result = get_pareto_result(X,Y, original_bias, metrics)
        add_to_excel(result, query_str, 'paretoFront_tfidf_w2v_mpnet')
        
main()
        