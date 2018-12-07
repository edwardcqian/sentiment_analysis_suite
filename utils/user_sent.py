import os, sys
import pandas as pd
import matplotlib.pyplot as plt
############### loading data ###############
# data = pd.read_csv('')

# check arguments
if len(sys.argv) != 3:
  print('usage: user_sent <file_path> <threshold>')
  print('accepted types: csv')
  sys.exit()

path = sys.argv[1]

# check file
if not os.path.exists(path):
  print('file does not exist')
  sys.exit()
name_ext = os.path.basename(path)
name, ext = os.path.splitext(name_ext)

# check type
if ext != '.csv':
  print ('accepted types: csv')
  sys.exit()

data = pd.read_csv(path, header=0, encoding='utf-8')

# drop unneeded columns
col_list = ['Consensus', 'tweetid', 'createdat', 'screenname']
data = data.reindex(columns=col_list)

data = data.fillna(' ')

# sort column
data = data.sort_values(['screenname', 'createdat'])

# reclass 2 to 0
data['Consensus'][data['Consensus']==2] = 0

# remove 0s
data = data[data['Consensus']!=0]

# reformat dates
data['createdat'] = data['createdat'].apply(lambda d: d[:10])

############### ploting ###############
# save_dir = ''
save_dir = os.path.dirname(os.path.abspath(path))

try:
    os.stat(save_dir+'/'+name+'_plots')
except:
    os.mkdir(save_dir+'/'+name+'_plots') 

# minimum tweets needed to graph (inclusive)
# threshold = 10
threshold = int(sys.argv[2])

name_list = data['screenname'].value_counts()
# subset by threshold
name_list = name_list[name_list > threshold]

for n in name_list.index:
    print(n)
    plt.clf()
    n_data = data[data['screenname']==n]
    y = n_data['Consensus']
    x = range(0,len(y))
    # my_xticks = n_data['createdat']
    # plt.xticks(x, my_xticks, rotation='vertical', fontsize=6)
    plt.yticks([-1,0,1])
    # plot
    plt.plot(x, y)
    plt.xlabel('Date')
    plt.ylabel('Sentiment')
    plt.title('Sentiment overtime for: ' + n)
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(save_dir+'/'+name+'_plots/'+n)


