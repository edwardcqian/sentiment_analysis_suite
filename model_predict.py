import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle, os

from keras.models import Model
from keras.layers import Dense, Embedding, Input, Activation, Conv1D, Flatten, MaxPooling1D, Add, concatenate, SpatialDropout1D
from keras.layers import LSTM, Bidirectional, GlobalMaxPooling1D, Dropout, GRU, GlobalAveragePooling1D
from keras.preprocessing import text, sequence

import lightgbm as lgb

import re
import random
from pandas import Series
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix, hstack
from keras.models import model_from_json

if len(sys.argv) != 4:
    print("usage: cli_model.py <file_path(type: csv)> <x_col_name> <weight_path> <out_file_path>")
    sys.exit()

# load data
path = sys.argv[1]
x_col = sys.argv[2]
data = pd.read_csv(path)

X_data = data[x_col]

out_path = sys.argv[4]

wt_path = sys.argv[3]

if wt_path[-1] != '/':
    wt_path+='/'

# word
print("Frequency Vectorization")
with open(wt_path+"wvec.pkl", "rb") as fin:
    word_vectorizer = pickle.load(fin)
w_vect = word_vectorizer.transform(X_data)

# character
with open(wt_path+"cvec.pkl", "rb") as fin:
    char_vectorizer = pickle.load(fin)
c_vect = char_vectorizer.transform(X_data)

wc_vect = hstack([w_vect, c_vect]).tocsr()

# nbsvm
print("nbsvm prediction")
with open(wt_path+"nbsvm.pkl", "rb") as fin:
    m = pickle.load(fin)
r = np.load(wt_path+"nbsvm_r.npy")
pred_nbsvm = m.predict_proba(wc_vect.multiply(r))

# lstm
print("Tokenizing for LSTM")
with open(wt_path+"lstm_tok.pkl", "rb") as fin:
    lstm_tok,maxlen = pickle.load(fin)
X_lstm = lstm_tok.texts_to_sequences(X_data)
X_lstm = sequence.pad_sequences(X_lstm, maxlen=maxlen)

with open(wt_path+"lstm_model.json", "r") as json_file:
    lstm_model_json = json_file.read()

lstm_model = model_from_json(lstm_model_json)
lstm_model.load_weights(wt_path+"lstm_final.hdf5")

pred_lstm = lstm_model.predict(X_lstm,batch_size=1024,verbose=1)


# gru
print("Tokenizing for GRU")
with open(wt_path+"gru_tok.pkl", "rb") as fin:
    gru_tok,maxlen = pickle.load(fin)
X_gru = gru_tok.texts_to_sequences(X_data)
X_gru = sequence.pad_sequences(X_gru, maxlen=maxlen)

with open(wt_path+"gru_model.json", "r") as json_file:
    gru_model_json = json_file.read()

gru_model = model_from_json(gru_model_json)
gru_model.load_weights(wt_path+"gru_final.hdf5")
pred_gru = gru_model.predict(X_gru,batch_size=1024,verbose=1)


# lgbm
print("lgb prediction")
with open(wt_path+"lgb.pkl", "rb") as fin:
    lgb_model = pickle.load(fin)

pred_lgb = lgb_model.predict(wc_vect)


# all together
pred_ens = (pred_nbsvm + pred_lstm + pred_gru + pred_lgb)/4

pred_class = np.argmax(pred_ens,axis=1)

pred_class -= 1

temp = pd.Series(pred_class)

temp.to_csv(out_path,index=False)

