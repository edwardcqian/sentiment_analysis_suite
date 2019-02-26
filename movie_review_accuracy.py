import pandas as pd 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
pred = pd.read_csv('test/pred.csv',header=None)
data = pd.read_csv('test/test_data.csv')

y_pred = data['Sentiment']

print(accuracy_score(y_pred,pred))
print(classification_report(y_pred,pred))
