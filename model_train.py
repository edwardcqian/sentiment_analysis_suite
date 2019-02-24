import sys
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score
import pickle, os

from keras.models import Model
from keras.layers import Dense, Embedding, Input, Activation, Conv1D, Flatten, MaxPooling1D, Add, concatenate, SpatialDropout1D
from keras.layers import LSTM, Bidirectional, GlobalMaxPooling1D, Dropout, CuDNNLSTM, CuDNNGRU, GlobalAveragePooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

import lightgbm as lgb

from utils import clean_text
from utils import re_sample
from utils import pr
from utils import get_mdl
from utils import get_onehot
from utils import pred_cutoff

from scipy.sparse import csr_matrix, hstack

############################### Loading annotated data ###############################
def load_data(args):
    print("Loading Data")
    data = pd.read_csv(args.path_data)
    training_no_none = data.loc[data['Consensus'] != 'NONE']
    # training_no_none = training_no_none.drop_duplicates(subset=['merged'])

    y = training_no_none['Consensus']
    y = y.astype('int64')
    y.value_counts()

    print("Cleaning Data")
    X_data = training_no_none['merged'].apply(clean_text)

    y.reset_index(drop=True, inplace=True)
    X_data.reset_index(drop=True, inplace=True)

    temp = pd.concat([y, X_data], join='outer', axis=1)
    temp.columns = ['tags','text']
    temp = temp.drop_duplicates(subset=['text'])

    y = temp['tags']
    X_data = temp['text']

    tag_dic = {} 
    for tag in y.unique(): 
        if tag not in tag_dic: 
            tag_dic[tag] = len(tag_dic) 

    y = y.apply(lambda x: tag_dic[x])

    X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size= args.val_split, random_state = 1)
    return X_train, X_test, y_train, y_test


############################### frequency vectorization ###############################
def model_tfidf(X_train, X_test, y_train, y_test, args):
        # word
    print("tfidf Vectorization")
    word_vectorizer = TfidfVectorizer(stop_words= 'english', analyzer='word', use_idf = 1, ngram_range = (1,1), max_features = 5000)
    word_vectorizer.fit(X_train)
    wtr_vect = word_vectorizer.transform(X_train)
    wts_vect = word_vectorizer.transform(X_test)
    # character
    char_vectorizer = TfidfVectorizer(stop_words= 'english', analyzer='char', use_idf = 1, ngram_range = (2,6), max_features = 50000)
    char_vectorizer.fit(X_train)
    ctr_vect = char_vectorizer.transform(X_train)
    cts_vect = char_vectorizer.transform(X_test)

    tr_vect = hstack([wtr_vect, ctr_vect]).tocsr()
    ts_vect = hstack([wts_vect, cts_vect]).tocsr()

    return tr_vect, ts_vect

def model_nbsvm(tr_vect, ts_vect, y_train, y_test, args):
    ############################### NB-SVM ###############################
    print("Fitting NB-SVM")
    m,r = get_mdl(tr_vect,y_train)

    print("Saving NB-SVM")
    filename = '/nbsvm.pkl'
    pickle.dump(m, open(args.directory+filename, 'wb'))
    filename = '/nbsvm_r'
    np.save(args.directory+filename, r)

############################### LSTM ###############################
def model_lstm(X_train, X_test, y_train, y_test, args):
# parameter values
    max_features = args.max_feat_lstm
    maxlen = 100
    batch_size = args.batch_size_lstm
    epochs = args.epochs_lstm
    num_class = args.num_classes
    embed_size = args.emb_size_lstm
    rec_size = args.recurrent_size_lstm

    print("Tokenizing for LSTM")
    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(X_train))
    X_tr = tokenizer.texts_to_sequences(X_train)
    X_tr = sequence.pad_sequences(X_tr, maxlen=maxlen)
    X_ts = tokenizer.texts_to_sequences(X_test)
    X_ts = sequence.pad_sequences(X_ts, maxlen=maxlen)

    y_tr_one = get_onehot(y_train, num_class)     
    y_ts_one = get_onehot(y_test, num_class)  

    lstm_input = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size)(lstm_input)
    x = Bidirectional(CuDNNLSTM(rec_size, return_sequences=True))(x)
    x = GlobalMaxPooling1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    lstm_output = Dense(num_class, activation="sigmoid")(x)
    lstm_model = Model(inputs=lstm_input, outputs=lstm_output)
    lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    file_path=args.directory+"/lstm_final.hdf5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=2)
    callbacks_list = [checkpoint, early] #early

    lstm_model.fit(X_tr, y_tr_one, batch_size=batch_size, epochs=epochs, validation_data=(X_ts,y_ts_one), callbacks=callbacks_list)


############################### GRU with Glove ###############################
# path to GloVe
def model_gru(X_train, X_test, y_train, y_test, args):
    EMBEDDING_FILE = args.path_embs

    max_features = args.max_feat_gru
    maxlen = 100
    batch_size = args.batch_size_gru
    epochs = args.epochs_gru
    num_class = args.num_classes
    embed_size = args.emb_size_gru
    rec_size = args.recurrent_size_gru


    print("Tokenizing for GRU")
    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(X_train))
    X_tr = tokenizer.texts_to_sequences(X_train)
    X_tr = sequence.pad_sequences(X_tr, maxlen=maxlen)
    X_ts = tokenizer.texts_to_sequences(X_test)
    X_ts = sequence.pad_sequences(X_ts, maxlen=maxlen)

    print("Loading GloVe Embeddings")
    embeddings_index = {}
    with open(EMBEDDING_FILE,encoding='utf8') as f:
        for line in f:
            values = line.rstrip().rsplit(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    word_index = tokenizer.word_index
    #prepare embedding matrix
    num_words = min(max_features, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    y_tr_one = get_onehot(y_train, num_class)
    y_ts_one = get_onehot(y_test, num_class) 

    gru_input = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix],trainable = False)(gru_input)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(CuDNNGRU(rec_size, return_sequences=True))(x)
    x = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool]) 
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    gru_output = Dense(num_class, activation="sigmoid")(x)
    gru_model = Model(gru_input, gru_output)
    gru_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


    file_path=args.directory+"gru_final.hdf5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=4)
    callbacks_list = [checkpoint, early] #early

    gru_model.fit(X_tr, y_tr_one, batch_size=batch_size, epochs=epochs, validation_data=(X_ts,y_ts_one), callbacks=callbacks_list)


############################### LightGBM ###############################
def model_lgb(tr_vect, ts_vect, y_train, y_test, args):
    d_train = lgb.Dataset(tr_vect, label=y_train)
    d_valid = lgb.Dataset(ts_vect, label=y_test)
    watchlist = [d_train, d_valid]
    params = {'learning_rate': 0.2,
                'application': 'multiclass',
                'num_class': args.num_classes,
                'num_leaves': 31,
                'verbosity': -1,
                'metric': 'auc',
                'data_random_seed': 2,
                'bagging_fraction': 0.8,
                'feature_fraction': 0.6,
                'nthread': 4,
                'lambda_l1': 1,
                'lambda_l2': 1}

    print("Fitting LGB")
    lgb_model = lgb.train(params,
                        train_set=d_train,
                        valid_sets=watchlist,
                        verbose_eval=10)
    print("Saving LGB")
    filename = '/lgb.pkl'
    pickle.dump(lgb_model, open(args.directory+filename, 'wb'))

def main(args):
    """Main function"""
    print('Reading Data')
    X_train, X_test, y_train, y_test = load_data(args)

    tr_vect, ts_vect = model_tfidf(X_train, X_test, y_train, y_test,args)

    model_nbsvm(tr_vect, ts_vect, y_train, y_test,args)

    model_lgb(tr_vect, ts_vect, y_train, y_test,args)

    model_lstm(X_train, X_test, y_train, y_test,args)

    model_gru(X_train, X_test, y_train, y_test,args)

def get_args(parser):
    parser.add_argument('--num_classes', type=int, required=True,
                        help='Number of sentiment classes')
    parser.add_argument('--emb_size_lstm', type=int, default=200,
                        help='Size of the LSTM embedding layer')
    parser.add_argument('--emb_size_gru', type=int, default=200,
                        help='Size of the GRU embedding layer (NOTE: must match word embedding file used)')
    parser.add_argument('--epochs_lstm', type=int, default=1,
                        help='Number of max LSTM epochs')
    parser.add_argument('--epochs_gru', type=int, default=1,
                        help='Number of max GRU epochs')
    parser.add_argument('--recurrent_size_lstm', type=int, default=50,
                        help='Size of the lstm recurrent layers')
    parser.add_argument('--recurrent_size_gru', type=int, default=128,
                        help='Size of the gru recurrent layers')
    parser.add_argument('--batch_size_lstm', type=int, default=32,
                        help='LSTM Batch Size')
    parser.add_argument('--batch_size_gru', type=int, default=128,
                        help='GRU Batch Size')
    parser.add_argument('--max_feat_lstm', type=int, default=20000,
                        help='LSTM maximum features')
    parser.add_argument('--max_feat_gru', type=int, default=50000,
                        help='GRU maximum features')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split')
    parser.add_argument('--path_data', type=str, default='data/climate_data.csv',
                        help='Path to data')
    parser.add_argument('--path_embs', type=str, default='data/glove.twitter.27B.200d.txt',
                        help='Path to embeddings')
    parser.add_argument('--directory', type=str, default='Model',
                        help='Directory to save the model and the log file to')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = get_args(parser)
    main(args)