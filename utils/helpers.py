import re
import random
from pandas import Series
import numpy as np
from sklearn.linear_model import LogisticRegression

# Removes meaningless symbols, and replace api placeholder symbols
def clean_text(raw): 
    # Remove HTML
    # review_text = BeautifulSoup(raw).get_text()
    review_text = raw 
    # Remove non-letters
    review_text = re.sub(r'http\S+', '', review_text)
    review_text = re.sub(r'â€¦', '', review_text)
    review_text = re.sub(r'…', '', review_text)
    review_text = re.sub(r'â€™', "'", review_text)
    review_text = re.sub(r'â€˜', "'", review_text)
    review_text = re.sub(r'\$q\$', "'", review_text)
    review_text = re.sub(r'&amp;', "and", review_text)
    review_text = re.sub('[^A-Za-z0-9#@ ]+', "", review_text)
    # review_text = re.sub(r'RT ', '', review_text) 
    # review_text = re.sub("[^a-zA-Z]", " ", review_text) 
    # review_text = re.sub(r' q ', ' ', review_text) 
        
    # Convert to lower case, split into individual words 
    # words = review_text.lower().split()  
    words = review_text.split()  

    # if remove_stopwords:
    #     stops = set(stopwords.words("english"))
    #     words = [w for w in words if not w in stops]

    #for w in words:
        #w = wordnet_lemmatizer.lemmatize(w)
        # very slow
        # w = spell(w)

    return( " ".join(words))

# resample the given dataset to get even class sizes
def re_sample(X, y):
    random.seed(0)
    rand_set = random.sample(range(0, X.shape[0]), X.shape[0])
    # find lowest class
    lowest_tag = y.value_counts().idxmin()
    lowest_cnt = y.value_counts().min()
    # data declarations
    X_data_bal = []
    y_data_bal = []
    pro_cnt = 0
    other_cnt = 0
    news_cnt = 0
    anti_cnt = 0
    for i in rand_set:
        if y.iloc[i] == -1 and anti_cnt < lowest_cnt:
            X_data_bal.append(X.iloc[i])
            y_data_bal.append(-1)
        elif y.iloc[i] == 1 and pro_cnt < lowest_cnt:
            X_data_bal.append(X.iloc[i])
            y_data_bal.append(1)
            pro_cnt += 1
        elif y.iloc[i] == 0 and other_cnt < lowest_cnt:
            X_data_bal.append(X.iloc[i])
            y_data_bal.append(0)
            other_cnt += 1
        elif y.iloc[i] == 2 and news_cnt < lowest_cnt:
            X_data_bal.append(X.iloc[i])
            y_data_bal.append(2)
            news_cnt += 1

    X_data_bal = Series(X_data_bal)
    y_data_bal = Series(y_data_bal)
    return X_data_bal, y_data_bal


# NB features
def pr(x, y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

# fit nb lr
def get_mdl(x, y):
    y = y.values
    r = np.log(pr(x,1,y) / pr(x,0,y))
    m = LogisticRegression(C=4, dual=True)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r

# create onehot encoding
def get_onehot(arr, num_class):
    return np.eye(num_class)[np.array(arr).reshape(-1)+1]

# predict classes using argmax, 
# provide a cutoff probability for each class
# data points outside of cutoff is given a placeholder class (3)
def pred_cutoff(pred, c0, c1, c2, c3):                                                                                             
    result = np.zeros(pred.shape[0])   
    prob = np.zeros(pred.shape[0])                                                           
    for i in range(0,pred.shape[0]):                                                                                               
        if pred[i].argmax(axis=-1) == 0 and pred[i][0] >= c0:                                    
            result[i] = 0  
            prob[i] = pred[i][0]                                                                                                        
        elif pred[i].argmax(axis=-1) == 1 and pred[i][1] >= c1:                                  
            result[i] = 1  
            prob[i] = pred[i][1]                                                                                                          
        elif pred[i].argmax(axis=-1) == 2 and pred[i][2] >= c2:                                               
            result[i] = 2    
            prob[i] = pred[i][2]                                                                                                        
        elif pred[i].argmax(axis=-1) == 3 and pred[i][3] >= c3:                                                                    
            result[i] = 3   
            prob[i] = pred[i][3]                                                                                    
        else:                                                                                                                      
            result[i] = 4                                                                                                          
    return result, prob