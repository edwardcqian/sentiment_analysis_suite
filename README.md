# Sentiment Analysis Suite
This repo contains a suite of python script to train your own ensemble model for sentiment analysis. The ensemble model is made up of a SVM with Naive Bayes features (nbsvm), a RNN with LSTM gates(lstm), a RNN with GRU gates and word vector embedding layers(gru), and a Microsoft's Gradient Boosted Tree - LightGBM (lgb).

## Model scripts
### Training
```
python model_train.py --option_arguments
```
Train the 4 individual models using the given training data, saves the model files into the given model folder
#### Training Arguments
##### General Arguments

- `--num_classes`: Number of classes in your dataset 
- `--val_split`: Proportion of data to use for validation during training
- `--path_data`: Path to data in csv format (i.e. `/home/data/my_data/training.csv`)
- `--path_embs`: Path to word embedding to be used in the model
- `--directory`: Directory to save model and log files to 
- `--skip_nbsvm`: Do not train the nbsvm model
- `--skip_lstm`: Do not train the lstm model
- `--skip_gru`: Do not train the gru model
- `--skip_lgb`: Do not train the lgb model

##### Model specific arguments
LSTM

- `--emb_size_lstm`: Size of the LSTM embedding layer
- `--epochs_lstm`: Number of max LSTM epochs to train
- `--recurrent_size_lstm`: Size of the recurrent layer for LSTM
- `--batch_size_lstm`: training batch size for LSTM
- `--max_feat_lstm`: Max number of word features for LSTM

GRU

- `--emb_size_gru`: Size of the GRU embedding layer
- `--epochs_gru`: Number of max GRU epochs to train
- `--recurrent_size_gru`: Size of the recurrent layer for GRU
- `--batch_size_gru`: training batch size for GRU
- `--max_feat_gru`: Max number of word features for GRU