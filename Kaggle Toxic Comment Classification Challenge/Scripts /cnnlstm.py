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
from keras.layers.convolutional import Conv1D

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



path = '/home/santanu/Downloads/'
comp = 'Toxic Comment/'
#EMBEDDING_FILE=path + comp + 'glove.6B.200d.txt'
EMBEDDING_FILE=path + comp + 'glove.840B.300d.txt'
TRAIN_DATA_FILE=path + comp + 'train.csv'
TEST_DATA_FILE=path + comp + 'test.csv'


embed_size = 300 #how big is each word vector
max_features = 20000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100# max number of words in a comment to use

train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)

list_sentences_train = train["comment_text"].fillna("_na_").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("_na_").values


tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)


def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
emb_mean,emb_std


word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

def model():

    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    xa = Bidirectional(GRU(100, return_sequences=True, dropout=0.2,recurrent_dropout=0.1))(x)
   # xb = Bidirectional(GRU(100, return_sequences=False, dropout=0.1, recurrent_dropout=0.1))(x)
 
    print x.shape
## Added , let' see 
    #x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x) 
   #x = attention_3d_block(x)
   #x1 = GlobalMaxPool1D()(xa)
   #x2 = GlobalAvgPool1D()(xa)
    xa1 = Conv1D(filters=64, kernel_size=3, strides=1, padding="same")(xa)
    xa1  = Dropout(0.2)(xa1) 
    xa1 = BatchNormalization()(xa1)
    xa1 = Activation("relu")(x)
    xa1 = Conv1D(filters=32, kernel_size=3, strides=1, padding="same")(xa1)
    xa1  = Dropout(0.2)(xa1) 
    xa1 = BatchNormalization()(xa1)
    xa1 = Activation("relu")(xa1)
    xa1 = Conv1D(filters=32, kernel_size=3, strides=1, padding="same")(xa1)
    xa1  = Dropout(0.2)(xa1) 
    xa1 = BatchNormalization()(xa1)
    xa1 = Activation("relu")(xa1)
    xa2 = GlobalMaxPool1D()(xa1)
    xa3 = Dense(6,activation="sigmoid")(xa2)
    model = Model(inputs=inp, outputs=x3)
    return model

k = 0
kf = KFold(n_splits=10, random_state=0, shuffle=True)

for train_index, test_index in kf.split(X_t):
    k += 1 
    X_train1,X_test1 = X_t[train_index],X_t[test_index]
    y_train1, y_test1 = y[train_index],y[test_index]
    model_final = model()
  
    adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model_final.compile(optimizer=adam, loss=["binary_crossentropy"],metrics=['accuracy'])
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.50,
                              patience=3, min_lr=0.000001)
    
    callbacks = [roc_auc_callback(training_data=(X_train1, y_train1),validation_data=(X_test1, y_test1)),
                EarlyStopping(monitor='roc_auc_val', patience=7, mode='max', verbose=1),
             CSVLogger('keras-5fold-run-01-v1-epochs_ib.log', separator=',', append=False),reduce_lr,
                ModelCheckpoint(
                        'kera1-5fold-run-01-v1-fold-' + str('%02d' % (k + 1)) + '-run-' + str('%02d' % (1 + 1)) + '.check',
                        monitor='roc_auc_val', mode='max', # mode must be set to max or Keras will be confused
                        save_best_only=True,
                        verbose=1)
            ]    
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_final.fit(X_train1,y_train1,batch_size=512,epochs=50,verbose=1,validation_data=(X_test1,y_test1),callbacks=callbacks)
    model_name = 'kera1-5fold-run-01-v1-fold-' + str('%02d' % (k + 1)) + '-run-' + str('%02d' % (1 + 1)) + '.check'
    del model_final
    model_final  = keras.models.load_model(model_name)
    model_name1 = '/home/santanu/Downloads/Toxic Comment/' + 'gruc2' + str(k) 
    model_final.save(model_name1)

 
for k in xrange(1,11):
    model_name1 = '/home/santanu/Downloads/Toxic Comment/' + 'gruc2' + str(k) 
    model = keras.models.load_model(model_name1)
    y_test = model.predict([X_te], batch_size=512, verbose=1)
    if k == 1:
        y_full = y_test.copy()
    else:
        y_full = y_full + y_test
sample_submission = pd.read_csv(path + comp + 'sample_submission.csv')
sample_submission[list_classes] = y_full/10.0
sample_submission.to_csv('submission_gru_comb_new.csv', index=False)

