import numpy as np
import pandas as pd
import re
import sys
import os
import urllib.request as urllib2
from bs4 import BeautifulSoup
import signal
import time
import csv

# timeout stub
def test_request(arg=None):
    """Your http request."""
    time.sleep(2)
    return arg
 
class Timeout():
    """Timeout class using ALARM signal."""
    class Timeout(Exception):
        pass
 
    def __init__(self, sec):
        self.sec = sec
 
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.raise_timeout)
        signal.alarm(self.sec)
 
    def __exit__(self, *args):
        signal.alarm(0)    # disable alarm
 
    def raise_timeout(self, *args):
        raise Timeout.Timeout()

# get path from command line
path = sys.argv[1]
start_index = sys.argv[2]
print(len(sys.argv))
if len(sys.argv) != 3:
  print('usage: get_url <file_path> <start_index>')
  print('accepted types: csv')
  sys.exit()

if not os.path.exists(path):
  print('file does not exist')
  sys.exit()

# get file name
name_ext = os.path.basename(path)
name, ext = os.path.splitext(name_ext)

# read file
if ext != '.csv':
  print ('accepted types: csv')
  sys.exit()
file = pd.read_csv(path, header=0, encoding='utf-8')[start_index:]

# initalize lists for features
url = []
handle = []
tag = []
site_title = []

col = 'message'

for i in range(0, file.shape[0]):
  if i % 10 == 0:
    print('%d of %d' % (i+1, file.shape[0]))
  http = re.findall(r'(https?://[\w|.|/]+)', file[col][i])
  line_handle = ' '.join(re.findall(r'\B@(\w{1,15})\b', file[col][i]))
  line_tag = ' '.join(sorted(re.findall(r'\B#(\w{1,35})\b', file[col][i])))
  # save every 500 tweets
  if i % 500 == 0 and i != 0:
    save = file[:i]
    save['url'] = url
    save['handle'] = handle
    save['tag'] = tag
    save['site_title'] = site_title
    loc = os.path.dirname(os.path.abspath(path))
    save.to_csv(loc+'/'+name+'_w_url_save.csv', index=False, encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
  if len(http) > 0:
    url.append(http[0])
    # get website title
    print(http[0])
    i = 0
    while i < 3:
      i += 1
      try:
        with Timeout(10):
            soup = BeautifulSoup(urllib2.urlopen(http[0], timeout=10))
        if soup and soup.title and soup.title.string:
          title = soup.title.string
        else:
          title = 'ERROR: no title'
      except urllib2.HTTPError:
        title = 'ERROR: 404'
      except urllib2.URLError:
        title = 'ERROR: BAD URL'
      except:
        continue
      break
    
  else:
    url.append("")
    title = 'ERROR: NO URL'
  
  site_title.append(title)

  if len(line_handle) > 0:
    handle.append(line_handle)
  else:
    handle.append("")

  if len(line_tag) > 0:
    tag.append(line_tag)
  else:
    tag.append("")

file['handle'] = np.asarray(handle)
file['tag'] = np.asarray(tag)
file['url'] = np.asarray(url)
file['site_title'] = np.asarray(site_title)

# save new csv
loc = os.path.dirname(os.path.abspath(path))
file.to_csv(loc+'/'+name+'_w_url.csv', index=False, encoding='utf-8')