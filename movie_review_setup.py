import pandas as pd
import os, sys
from sklearn.model_selection import train_test_split

if not os.path.exists('test'):
    print('test directory not found, please see [url] for instructions')
    sys.exit()
if not os.path.exists('test/train.tsv'):
    print('test/train.tsv not found, please see [url] for instructions')
    sys.exit()

data = pd.read_csv('test/train.tsv', sep='\t')

train, test = train_test_split(data, test_size=0.1, random_state = 1)

train.to_csv('test/train_data.csv', index=False)

test.to_csv('test/test_data.csv', index=False)