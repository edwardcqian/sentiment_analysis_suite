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
# parameter values
max_features = 20000
maxlen = 100
batch_size = 32
epochs = 50
num_class = 4
embed_size = 128


print("Tokenizing for LSTM")
with open(wt_path+"lstm_tok.pkl", "rb") as fin:
    lstm_tok = pickle.load(fin)
X_lstm = lstm_tok.texts_to_sequences(X_data)
X_lstm = sequence.pad_sequences(X_lstm, maxlen=maxlen)

# model structure
print("LSTM prediction")
lstm_input = Input(shape=(maxlen, ))
x = Embedding(max_features, embed_size)(lstm_input)
x = Bidirectional(LSTM(50, return_sequences=True))(x)
x = GlobalMaxPooling1D()(x)
x = Dropout(0.1)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
lstm_output = Dense(num_class, activation="sigmoid")(x)
lstm_model = Model(inputs=lstm_input, outputs=lstm_output)
lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

file_path= wt_path+"lstm_final.hdf5"

lstm_model.load_weights(file_path)

pred_lstm = lstm_model.predict(X_lstm,batch_size=1024,verbose=1)


# gru
max_features=50000
maxlen=150
batch_size = 128
epochs = 20
num_class = 4
embed_size=200


print("Tokenizing for GRU")
with open(wt_path+"gru_tok.pkl", "rb") as fin:
    gru_tok = pickle.load(fin)
X_gru = gru_tok.texts_to_sequences(X_data)
X_gru = sequence.pad_sequences(X_gru, maxlen=maxlen)

print("Loading GloVe Embeddings")
embedding_matrix = np.load(wt_path+"glove_embedding.npy")

# model structure
print("GRU prediction")
gru_input = Input(shape=(maxlen, ))
x = Embedding(max_features, embed_size, weights=[embedding_matrix],trainable = False)(gru_input)
x = SpatialDropout1D(0.2)(x)
x = Bidirectional(GRU(128, return_sequences=True, reset_after=True, recurrent_activation='sigmoid'))(x)
x = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(x)
avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
x = concatenate([avg_pool, max_pool]) 
x = Dense(128, activation='relu')(x)
x = Dropout(0.1)(x)
gru_output = Dense(4, activation="sigmoid")(x)
gru_model = Model(gru_input, gru_output)
gru_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

file_path= wt_path+"gru_final.hdf5"

gru_model.load_weights(file_path)
pred_gru = gru_model.predict(X_gru,batch_size=1024,verbose=1)


# lgbm
print("lgb prediction")
with open(wt_path+"lgb.pkl", "rb") as fin:
    lgb_model = pickle.load(fin)

pred_lgb = lgb_model.predict(wc_vect)

pred_ens = (pred_nbsvm + pred_lstm + pred_gru + pred_lgb)/4

pred_class = np.argmax(pred_ens,axis=1)

pred_class -= 1

temp = pd.Series(pred_class)

temp.to_csv(out_path,index=False)
