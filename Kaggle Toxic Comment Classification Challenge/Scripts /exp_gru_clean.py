import sys, os, re, csv, codecs, numpy as np, pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation,GRU
from keras.layers import Bidirectional, GlobalMaxPool1D,GlobalAvgPool1D,Concatenate
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from sklearn.model_selection import KFold
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, Callback
from keras import callbacks
from keras.layers.merge import concatenate

from keras.layers import merge,BatchNormalization,Activation
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *

from attention_utils import get_activations, get_data_recurrent
from sklearn.metrics import roc_auc_score
#from keras.callbacks import Callback

import numpy as np
np.random.seed(42)
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import re
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from keras import backend as K

import warnings
warnings.filterwarnings('ignore')




# if True, the attention vector is shared across the input_dimensions where the attention is applied.
SINGLE_ATTENTION_VECTOR = False
APPLY_ATTENTION_BEFORE_LSTM = False
TIME_STEPS = 100

def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul


class roc_auc_callback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x, verbose=0)
        roc = roc_auc_score(self.y, y_pred)
        logs['roc_auc'] = roc_auc_score(self.y, y_pred)
        logs['norm_gini'] = ( roc_auc_score(self.y, y_pred) * 2 ) - 1

        y_pred_val = self.model.predict(self.x_val, verbose=0)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        logs['roc_auc_val'] = roc_auc_score(self.y_val, y_pred_val)
        logs['norm_gini_val'] = ( roc_auc_score(self.y_val, y_pred_val) * 2 ) - 1

        print('\rroc_auc: %s - roc_auc_val: %s - norm_gini: %s - norm_gini_val: %s' % (str(round(roc,5)),str(round(roc_val,5)),str(round((roc*2-1),5)),str(round((roc_val*2-1),5))))
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return



#EMBEDDING_FILE = '/home/santanu/Downloads/Toxic Comment/glove.840B.300d.txt'
EMBEDDING_FILE = '/home/santanu/Downloads/Toxic Comment/crawl-300d-2M.vec'


train = pd.read_csv('/home/santanu/Downloads/Toxic Comment/train.csv')
test = pd.read_csv('/home/santanu/Downloads/Toxic Comment/test.csv')
#submission = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')

# PREPROCESSING PART
repl = {
    "&lt;3": " good ",
    ":d": " good ",
    ":dd": " good ",
    ":p": " good ",
    "8)": " good ",
    ":-)": " good ",
    ":)": " good ",
    ";)": " good ",
    "(-:": " good ",
    "(:": " good ",
    "yay!": " good ",
    "yay": " good ",
    "yaay": " good ",
    "yaaay": " good ",
    "yaaaay": " good ",
    "yaaaaay": " good ",
    ":/": " bad ",
    ":&gt;": " sad ",
    ":')": " sad ",
    ":-(": " bad ",
    ":(": " bad ",
    ":s": " bad ",
    ":-s": " bad ",
    "&lt;3": " heart ",
    ":d": " smile ",
    ":p": " smile ",
    ":dd": " smile ",
    "8)": " smile ",
    ":-)": " smile ",
    ":)": " smile ",
    ";)": " smile ",
    "(-:": " smile ",
    "(:": " smile ",
    ":/": " worry ",
    ":&gt;": " angry ",
    ":')": " sad ",
    ":-(": " sad ",
    ":(": " sad ",
    ":s": " sad ",
    ":-s": " sad ",
    r"\br\b": "are",
    r"\bu\b": "you",
    r"\bhaha\b": "ha",
    r"\bhahaha\b": "ha",
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bhasn't\b": "has not",
    r"\bhaven't\b": "have not",
    r"\bhadn't\b": "had not",
    r"\bwon't\b": "will not",
    r"\bwouldn't\b": "would not",
    r"\bcan't\b": "can not",
    r"\bcannot\b": "can not",
    r"\bi'm\b": "i am",
    "m": "am",
    "r": "are",
    "u": "you",
    "haha": "ha",
    "hahaha": "ha",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "won't": "will not",
    "wouldn't": "would not",
    "can't": "can not",
    "cannot": "can not",
    "i'm": "i am",
    "m": "am",
    "i'll" : "i will",
    "its" : "it is",
    "it's" : "it is",
    "'s" : " is",
    "that's" : "that is",
    "weren't" : "were not",
}

keys = [i for i in repl.keys()]

new_train_data = []
new_test_data = []
ltr = train["comment_text"].tolist()
lte = test["comment_text"].tolist()
for i in ltr:
    arr = str(i).split()
    xx = ""
    for j in arr:
        j = str(j).lower()
        if j[:4] == 'http' or j[:3] == 'www':
            continue
        if j in keys:
            # print("inn")
            j = repl[j]
        xx += j + " "
    new_train_data.append(xx)
for i in lte:
    arr = str(i).split()
    xx = ""
    for j in arr:
        j = str(j).lower()
        if j[:4] == 'http' or j[:3] == 'www':
            continue
        if j in keys:
            # print("inn")
            j = repl[j]
        xx += j + " "
    new_test_data.append(xx)
train["new_comment_text"] = new_train_data
test["new_comment_text"] = new_test_data
print("crap removed")
trate = train["new_comment_text"].tolist()
tete = test["new_comment_text"].tolist()
for i, c in enumerate(trate):
    trate[i] = re.sub('[^a-zA-Z ?!]+', '', str(trate[i]).lower())
for i, c in enumerate(tete):
    tete[i] = re.sub('[^a-zA-Z ?!]+', '', tete[i])
train["comment_text"] = trate
test["comment_text"] = tete
print('only alphabets')


X_train = train["comment_text"].fillna("fillna").values
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
X_test = test["comment_text"].fillna("fillna").values

#max_features = 30000
max_features = 30000
maxlen = 100
embed_size = 300

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)
x_tr = pd.read_csv('/home/santanu/Downloads/Toxic Comment/train_feat.csv')
x_te = pd.read_csv('/home/santanu/Downloads/Toxic Comment/test_feat.csv')
x_tr,x_te = x_tr.values,x_te.values
len1 = x_tr.shape[1]

def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


'''
# Fasttext doesn't require this #
all_embs = np.stack(embeddings_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
'''

def model_():
    inp1 = Input(shape=(maxlen, ))
    inp2 = Input(shape=(len1, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp1)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(80, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool,inp2])
    outp = Dense(6, activation="sigmoid")(conc)
    
    model = Model(inputs=[inp1,inp2], outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model



k = 0
batch_size=128
epochs = 2 
kf = KFold(n_splits=20, random_state=0, shuffle=True)

for train_index, test_index in kf.split(x_train):
    k += 1 
    X_train1,X_test1 = x_train[train_index],x_train[test_index]
    X_train2,X_test2 = x_tr[train_index],x_tr[test_index]
    y_train1, y_test1 = y_train[train_index],y_train[test_index]
    model = model_()
  
    #adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(optimizer='adam', loss=["binary_crossentropy"],metrics=['accuracy'])
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.50,
                              patience=3, min_lr=0.000001)
    
    callbacks = [roc_auc_callback(training_data=([X_train1,X_train2], y_train1),validation_data=([X_test1,X_test2], y_test1)),
                EarlyStopping(monitor='roc_auc_val', patience=7, mode='max', verbose=1),
             CSVLogger('keras-5fold-run-01-v1-epochs_ib.log', separator=',', append=False),reduce_lr,
                ModelCheckpoint(
                        'kera1-5fold-run-01-v1-fold-' + str('%02d' % (k + 1)) + '-run-' + str('%02d' % (1 + 1)) + '.check',
                        monitor='roc_auc_val', mode='max', # mode must be set to max or Keras will be confused
                        save_best_only=True,
                        verbose=1)
            ]    
    exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
    steps = int(len(X_train1)/batch_size) * epochs
    lr_init, lr_fin = 0.001, 0.0005
    lr_decay = exp_decay(lr_init, lr_fin, steps)
    K.set_value(model.optimizer.lr, lr_init)
    K.set_value(model.optimizer.decay, lr_decay)
    model.fit([X_train1,X_train2],y_train1,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=([X_test1,X_test2],y_test1),callbacks=callbacks)
    model_name = 'kera1-5fold-run-01-v1-fold-' + str('%02d' % (k + 1)) + '-run-' + str('%02d' % (1 + 1)) + '.check'
    del model
    model  = keras.models.load_model(model_name)
    model_name1 = '/home/santanu/Downloads/Toxic Comment/' + 'gru20_ft_' + str(k) 
    model.save(model_name1)

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
 
for k in xrange(1,21):
    print k
    model_name1 = '/home/santanu/Downloads/Toxic Comment/' + 'gru20_ft_' + str(k) 
    model = keras.models.load_model(model_name1)
    y_test = model.predict([x_test,x_te], batch_size=1024, verbose=1)
    if k == 1:
        y_full = y_test.copy()
    else:
        y_full = y_full + y_test
sample_submission = pd.read_csv('/home/santanu/Downloads/Toxic Comment/sample_submission.csv')
sample_submission[list_classes] = y_full/10.0
sample_submission.to_csv('submission_gru_att_exp_20_ft_lda.csv', index=False)

