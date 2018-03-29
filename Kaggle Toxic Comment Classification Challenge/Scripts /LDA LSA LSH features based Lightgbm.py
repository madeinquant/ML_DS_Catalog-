# Latent Dirichlet allocation Topic proportion features 

import graphlab
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
import string
import gensim
from gensim import corpora
import pandas as pd
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding('utf8')

def clean_doc(doc):
	stop = set(stopwords.words('english'))
	exclude = set(string.punctuation)
	lemma = WordNetLemmatizer()
	cleaned_doc = " ".join([word for word in doc.lower().split() if word not in stop])
	cleaned_doc = "".join(ch for ch in cleaned_doc if ch not in exclude)
	cleaned_doc = " ".join(lemma.lemmatize(word) for word in cleaned_doc.split())

	return cleaned_doc
path = '/home/santanu/Downloads/' 
comp = 'Toxic Comment/'
#EMBEDDING_FILE=path + comp + 'glove.6B.50d.txt'
TRAIN_DATA_FILE=path + comp + 'train.csv'
TEST_DATA_FILE=path + comp + 'test.csv'

train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values

train['comment_text'] = train['comment_text'].apply(clean_doc)
test['comment_text'] = test['comment_text'].apply(clean_doc)
corpus = graphlab.SArray(train['comment_text'].append(test['comment_text']))
docs = graphlab.text_analytics.count_words(corpus)
model = graphlab.topic_model.create(docs,num_topics=30,initial_topics=None, alpha=None, beta=0.1, num_iterations=100, num_burnin=10, associations=None, verbose=True, print_interval=10, validation_set=None, method='auto')
topics = model.predict(docs,output_type='probabilities')

topics = np.array(topics)
train_lda = topics[:len(train)]
test_lda = topics[len(train):]

del train
del test


#Local Sensitive Hashing Bin features

import numpy as np
import graphlab
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm
from sklearn.metrics.pairwise import pairwise_distances
import time
from copy import copy
import matplotlib.pyplot as plt
%matplotlib inline

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from scipy.special import logit, expit

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train = pd.read_csv('/home/santanu/Downloads/Toxic Comment/train.csv').fillna(' ')
test = pd.read_csv('/home/santanu/Downloads/Toxic Comment/test.csv').fillna(' ')

train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 1),
    max_features=10000)
word_vectorizer.fit(all_text)
word_features = word_vectorizer.transform(all_text)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    ngram_range=(1, 5),
    max_features=50000)
char_vectorizer.fit(all_text)
char_features = char_vectorizer.transform(all_text)

all_features = hstack([char_features,word_features])

all_features = all_features.tocsr()
print 'can terminate now'

def generate_random_vectors(num_vector, dim):
    return np.random.randn(dim, num_vector)

def train_lsh(data, num_vector=16, seed=None):
    
    dim = 60000
    if seed is not None:
        np.random.seed(seed)
    random_vectors = generate_random_vectors(num_vector, dim)
  
    powers_of_two = 1 << np.arange(num_vector-1, -1, -1)
  
    table = {}
    
    # Partition data points into bins
    bin_index_bits = (data.dot(random_vectors) >= 0)
    #print bin_index_bits
  
    # Encode bin index bits into integers
    bin_indices = bin_index_bits.dot(powers_of_two)
    #print bin_indices
    
    # Update `table` so that `table[i]` is the list of document ids with bin index equal to i.
    for data_index, bin_index in enumerate(bin_indices):
        if bin_index not in table:
            # If no list yet exists for this bin, assign the bin an empty list.
            table[bin_index] = [] # YOUR CODE HERE
        # Fetch the list of document ids associated with the bin and add the document id to the end.
        # YOUR CODE HERE
        table[bin_index].append(data_index)
            
    model = {'data': data,
             'bin_index_bits': bin_index_bits,
             'bin_indices': bin_indices,
             'table': table,
             'random_vectors': random_vectors,
             'num_vector': num_vector}
    
    return model

model = train_lsh(all_features, num_vector=32, seed=143)
all_bins = model['bin_index_bits']


#Latent Semantic Analysis features

from sklearn.decomposition import SparsePCA,TruncatedSVD
model_n = TruncatedSVD(n_components=20)
W = model_n.fit_transform(all_features)

# Polarity scores TextBlob


polarity_scores = []
import textblob
for t in all_text.values:
    text_ = textblob.TextBlob(t)
    feat = [text_.sentiment.polarity,text_.sentiment.subjectivity]
    polarity_scores.append(feat)

# Merging all the features together 

topics = np.array(topics)
polarity_scores = np.array(polarity_scores)
all_feat = np.hstack((topics,all_bins,W,polarity_scores))
train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)
X_tr,X_te = all_feat[:len(train),:],all_feat[len(train):,:]


#saving the features so that wont need to regenerate them


pd.DataFrame(X_tr).to_csv('/home/santanu/Downloads/Toxic Comment/train_feat.csv',index=False)
pd.DataFrame(X_te).to_csv('/home/santanu/Downloads/Toxic Comment/test_feat.csv',index=False)

# Final Lightgbm model on the features and prediction

import lightgbm as lgb
from  sklearn.model_selection import StratifiedKFold
train_features = X_tr
test_features = X_te
#train_features = train_features.tocsr()
#test_features = test_features.tocsr()
losses = []
predictions = {'id': test['id']}
pred_out = pd.DataFrame()
for class_name in class_names:
    print class_name 
    train_target = train[class_name]
    k = 0
    kf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    for train_index, test_index in kf.split(train_features,train_target):
        k += 1
                  
        X_train,X_test = train_features[train_index],train_features[test_index]
        y_train, y_test = train_target[train_index],train_target[test_index]
	import lightgbm as lgb
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

        params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0  }

        print('Start training...')
    # train
        gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=500,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=20
                   )

        #model_name = '/home/santanu/Downloads/Churn prediction/' + 'lgbm_model_' + str(k) 
        #gbm.save_model(model_name)
    
        print('Start predicting...')
      # predict
        out = gbm.predict(test_features,num_iteration=gbm.best_iteration)
        out_pred_fld = class_name + '_'  + str(k)
        pred_out[out_pred_fld] = out  
        model_name = "/home/santanu/Downloads/Toxic Comment/" + class_name + 'ldalsh_' + str(k)
        gbm.save_model(model_name)

pred_out.to_csv('/home/santanu/Downloads/lgbm_lda_pol.csv',index=False)    

pred_out['id'] = test['id']
for c in class_names:
    fields = [c + '_1',c + '_2',c + '_3',c + '_4',c + '_5']
    pred_out[c] = np.mean(pred_out[fields],axis=1)
reqd = ['id'] + class_names
pred_out[reqd].to_csv('/home/santanu/Downloads/Toxic Comment/experiment.csv',index=False) 







