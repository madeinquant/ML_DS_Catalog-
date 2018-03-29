import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from scipy.special import logit, expit
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
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
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    ngram_range=(1, 5),
    max_features=30000)
char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)

train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])
train_features = train_features.tocsr()
test_features = test_features.tocsr()
print np.shape(train_features)
print np.shape(test_features)
losses = []
predictions = {'id': test['id']}
for class_name in class_names:
    print class_name 
    train_target = train[class_name]
    k = 0
    kf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    for train_index, test_index in kf.split(train_features,train_target):
        k += 1
                  
        X_train,X_test = train_features[train_index],train_features[test_index]
        y_train, y_test = train_target[train_index],train_target[test_index]

        DTrain = xgb.DMatrix(X_train,label=y_train)
        DVal = xgb.DMatrix(X_test,label=y_test)
	xgb_params = {
	    'objective': 'binary:logistic',
            'eval_metric':'auc',
	    'eta': 0.07, 
	    'max_depth':4,
	    'lambda':1.3,
	    'alpha': 8,
	    'subsample':0.8,
	    #'colsample_bytree': 1 / F_train.shape[1]**0.5,
	    'colsample_bytree':0.8,
	    'min_child_weight':6,
	    'gamma':8,
	    'silent': 1,
	    'nthread':-1,
	    'seed':0
	    #'scale_pos_weight':1.6
	}
	bst = xgb.train(xgb_params,DTrain,1000,[(DTrain,'train'),((DVal,'val'))],early_stopping_rounds=20,verbose_eval=10)
        model_name = "/home/santanu/Downloads/Toxic Comment/" + class_name + str(k)
        bst.save_model(model_name)

    

 
    
