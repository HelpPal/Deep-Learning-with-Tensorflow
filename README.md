# Deep-Learning-with-Tensorflow

notMINST Data Exploration
The objective of this assignment is to learn about simple data curation practices, and familiarize you with some of the data we'll be reusing later.

This notebook uses the notMNIST dataset to be used with python experiments. This dataset is designed to look like the classic MNIST dataset, while looking a little more like real data: it's a harder task, and the data is a lot less 'clean' than MNIST.

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

# Config the matlotlib backend as plotting inline in IPython
%matplotlib inline
First, we'll download the dataset to our local machine. The data consists of characters rendered in a variety of fonts on a 28x28 image. The labels are limited to 'A' through 'J' (10 classes). The training set has about 500k and the testset 19000 labelled examples. Given these sizes, it should be possible to train models quickly on any machine.

url = 'http://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None

def download_progress_hook(count, blockSize, totalSize):
  """A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 1% change in download progress.
  """
  global last_percent_reported
  percent = int(count * blockSize * 100 / totalSize)

  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()
      
    last_percent_reported = percent
        
def maybe_download(filename, expected_bytes, force=False):
  """Download a file if not present, and make sure it's the right size."""
  if force or not os.path.exists(filename):
    print('Attempting to download:', filename) 
    filename, _ = urlretrieve(url + filename, filename, reporthook=download_progress_hook)
    print('\nDownload Complete!')
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)
Found and verified notMNIST_large.tar.gz
Found and verified notMNIST_small.tar.gz
Extract the dataset from the compressed .tar.gz file. This should give you a set of directories, labelled A through J.

num_classes = 10
np.random.seed(133)

def maybe_extract(filename, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall()
    tar.close()
  data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d))]
  if len(data_folders) != num_classes:
    raise Exception(
      'Expected %d folders, one per class. Found %d instead.' % (
        num_classes, len(data_folders)))
  print(data_folders)
  return data_folders
  
train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)
notMNIST_large already present - Skipping extraction of notMNIST_large.tar.gz.
['notMNIST_large/A', 'notMNIST_large/B', 'notMNIST_large/C', 'notMNIST_large/D', 'notMNIST_large/E', 'notMNIST_large/F', 'notMNIST_large/G', 'notMNIST_large/H', 'notMNIST_large/I', 'notMNIST_large/J']
notMNIST_small already present - Skipping extraction of notMNIST_small.tar.gz.
['notMNIST_small/A', 'notMNIST_small/B', 'notMNIST_small/C', 'notMNIST_small/D', 'notMNIST_small/E', 'notMNIST_small/F', 'notMNIST_small/G', 'notMNIST_small/H', 'notMNIST_small/I', 'notMNIST_small/J']
Problem 1
Let's take a peek at some of the data to make sure it looks sensible. Each exemplar should be an image of a character A through J rendered in a different font. Display a sample of the images that we just downloaded. Hint: you can use the package IPython.display.

# We can display images using Image(filename="")
Image(filename="notMNIST_small/A/Q0NXaWxkV29yZHMtQm9sZEl0YWxpYy50dGY=.png")

# We get all file names
fn = os.listdir("notMNIST_small/A/")
fn
['MDEtMDEtMDAudHRm.png',
 'MDRiXzA4LnR0Zg==.png',
 'MjAwcHJvb2Ztb29uc2hpbmUgcmVtaXgudHRm.png',
 'MlJlYmVsc0RldXgtQmxhY2sub3Rm.png',
 'MlRvb24gU2hhZG93LnR0Zg==.png',
 'MlRvb24yIFNoYWRvdy50dGY=.png',
 'MTAuMTUgU2F0dXJkYXkgTmlnaHQgQlJLLnR0Zg==.png',
 'MTFTMDEgQmxhY2sgVHVlc2RheSBPZmZzZXQudHRm.png',
 'MTggSG9sZXMgQlJLLnR0Zg==.png',
 'MTh0aENlbnR1cnkudHRm.png',
 'MTIgV2FsYmF1bSBJdGFsaWMgMTMyNjMudHRm.png',
 'MTJTYXJ1WWVsbG93Rm9nLnR0Zg==.png',
 'Nng3b2N0IEFsdGVybmF0ZSBFeHRyYUxpZ2h0LnR0Zg==.png',
 'Nng3b2N0IEFsdGVybmF0ZSBSZWd1bGFyLnR0Zg==.png',
 'NXRoR3JhZGVyLnR0Zg==.png',
 'OC1iaXQgTGltaXQgTyBCUksudHRm.png',
 'OEJhbGxTY3JpcHRTQ2Fwc1NTSyBJdGFsaWMudHRm.png',
 'OTExIFBvcnNjaGEgSXRhbGljLnR0Zg==.png',
 'OXNxZ3JkIFRoaW4udHRm.png',
 'Q09ERTNYLnR0Zg==.png',
 'Q0cgT21lZ2EudHRm.png',
 'Q0NDb21pY3JhenktQm9sZEl0YWxpYy50dGY=.png',
 'Q0NTcG9va3l0b290aC1SZWd1bGFyLnR0Zg==.png',
 'Q0NXaWxkV29yZHMtQm9sZEl0YWxpYy50dGY=.png',
 'Q0sgUGluay50dGY=.png',
 'Q1FOLU1vbGVjdWxhci1EeW5hbW8tUmVndWxhci5vdGY=.png',
 'Q290cmFuIERpc3BsYXkgU1NpLnR0Zg==.png',
 'Q291bnRyeXNpZGUgQmxhY2sgU1NpIEV4dHJhIEJvbGQudHRm.png',
 'Q291cmllci1Cb2xkLm90Zg==.png',
 'Q291cmllci1Cb2xkT2JsaXF1ZS5vdGY=.png',
 'Q291cmllciAxMCBQaXRjaCBJdGFsaWMucGZi.png',
 'Q291cmllciBOZXcgQ0UgQm9sZCBJdGFsaWMudHRm.png',
 'Q291cmllcjEwIEJUIEl0YWxpYy50dGY=.png',
 'Q291cmllcjEwIEJUIFJvbWFuLnR0Zg==.png',
 'Q291cmllcjEwUGl0Y2hCVC1JdGFsaWMub3Rm.png',
 'Q291cmllckMtQm9sZC5vdGY=.png',
 'Q291cmllckMtT2JsaXF1ZS5vdGY=.png',
 'Q291Y2hsb3Zlcjk3LnR0Zg==.png',
 'Q292aW5ndG9uIENvbmQgQm9sZCBJdGFsaWMudHRm.png',
 'Q292aW5ndG9uIFNoYWRvdyBJdGFsaWMudHRm.png',
 'Q29hZ3VsYXRlLnR0Zg==.png',
 'Q29iYWx0IE5vcm1hbC50dGY=.png',
 'Q29jaGluIExULnR0Zg==.png',
 'Q29jb24tUmVndWxhci5vdGY=.png',
 'Q29jb2EgVGhpbiBOb3JtYWwudHRm.png',
 'Q29jYWluZSBTYW5zLnR0Zg==.png',
 'Q29lcmNpb25OYWtlZC50dGY=.png',
 'Q29mZmVlIFNob3AudHRm.png',
 'Q29sb255Lm90Zg==.png',
 'Q29sb3NzYWxpcy1CbGFjay5vdGY=.png',
 'Q29sbWNpbGxlTVRTdGQtUmVndWxhci5vdGY=.png',
 'Q29sdW1iaWEtQm9sZC5vdGY=.png',
 'Q29sdW1idXNNVFN0ZC1Cb2xkSXRhbGljLm90Zg==.png',
 'Q29seW1waWEtRGVtaUJvbGQub3Rm.png',
 'Q29seW1waWEtTGlnaHQub3Rm.png',
 'Q29sZGJyaW5nZXIgS0cudHRm.png',
 'Q29taWNCb29rLm90Zg==.png',
 'Q29taXggUmVndWxhcigxKS50dGY=.png',
 'Q29tb25zIEJvbGQudHRm.png',
 'Q29tbWVyY2lhbFNjcmlwdC5vdGY=.png',
 'Q29tbWVyY2UgQ29uZGVuc2VkIFNTaSBDb25kZW5zZWQgSXRhbGljLnR0Zg==.png',
 'Q29tbWVyY2UgQmxhY2sgU1NpIEJvbGQudHRm.png',
 'Q29tbWVyY2UgU1NpIFNlbWkgQm9sZC50dGY=.png',
 'Q29tbXVuaXR5U2VydmljZSBCbGFjay50dGY=.png',
 'Q29tc2F0LU5hdnktVW5pdC5vdGY=.png',
 'Q29tc2F0LVN0YXRpb24ub3Rm.png',
 'Q29tc2F0LVVuaXQub3Rm.png',
 'Q29tcGF0aWxGYWN0IExUIFJlZ3VsYXIgU21hbGwgQ2Fwcy50dGY=.png',
 'Q29tcGF0aWxMZXR0ZXIgTFQgQm9sZCBTbWFsbCBDYXBzLnR0Zg==.png',
 'Q29tcGF0aWxMZXR0ZXIgTFQgUmVndWxhciBTbWFsbCBDYXBzLnR0Zg==.png',
 'Q29tcGF0aWxUZXh0IExUIFJlZ3VsYXIgU21hbGwgQ2Fwcy50dGY=.png',
 'Q29tcGFjdEMtQm9sZEl0YWxpYy5vdGY=.png',
 'Q29tcGFjdGEgQmxhY2sgQlQudHRm.png',
 'Q29tcGFjdGEgQmxhY2sucGZi.png',
 'Q29tcGFjdGEgSUNHIEJvbGRJdGFsaWMudHRm.png',
 'Q29tcGFjdGEucGZi.png',
 'Q29tcGFjdGFFRi1Cb2xkSXRhbGljLm90Zg==.png',
 'Q29tcGFjdGFFRi1MaWdodC5vdGY=.png',
 'Q29tcGFjdGFTdGQtQm9sZC5vdGY=.png',
 'Q29tcGVuZGl1bSBCbGFjayBTU2kgQmxhY2sudHRm.png',
 'Q29tcHJlc3Nvci1TbGFiU2VyaWYub3Rm.png',
 'Q29tcHV0ZXJmb250LnR0Zg==.png',
 'Q29tZW5pdXNCUS1NZWRpdW0ub3Rm.png',
 'Q29uc29sYXMgSXRhbGljLnR0Zg==.png',
 'Q29uc3RhbnRpYS50dGY=.png',
 'Q29uc3RydWN0aXZpc3QgU29saWQudHRm.png',
 'Q29uc3RydWN0YS1UaGluLm90Zg==.png',
 'Q29ucXVpc3RhIFNTaSBJdGFsaWMudHRm.png',
 'Q29udGludXVtIEJvbGQudHRm.png',
 'Q29udGluZW50YWxSYWlsd2F5LnBmYg==.png',
 'Q29udGV4dCBSZXByaXNlIE1lZGl1bSBTU2kgTWVkaXVtLnR0Zg==.png',
 'Q29udGV4dCBSZXByaXNlIENvbmRlbnNlZCBTU2kgQ29uZGVuc2VkLnR0Zg==.png',
 'Q29udGV4dCBSZXByaXNlIFRoaW4gU1NpIFRoaW4udHRm.png',
 'Q29udGV4dCBTU2kgQm9sZCBJdGFsaWMudHRm.png',
 'Q29udGV4dCBTZW1pIENvbmRlbnNlZCBTU2kgU2VtaSBDb25kZW5zZWQudHRm.png',
 'Q29udHJvbCBGcmVhayBPZmZzZXQudHRm.png',
 'Q29uY29yZGlhU1NLIEJvbGQudHRm.png',
 'Q29uY29yZGUgTm92YSAoUikgRXhwZXJ0IEl0YWxpYyBPc0YudHRm.png',
 'Q29uY29yZGVCRS1Cb2xkQ24ub3Rm.png',
 'Q29uY29yZGVFeHBlcnRCUS1JdGFsaWNPc0Yub3Rm.png',
 'Q29uY29yZGVFeHBlcnRCUS1NZWRpdW0ub3Rm.png',
 'Q29uY3Vyc29JdGFsaWFuIEJUTiBCb2xkLnR0Zg==.png',
 'Q29uY3Vyc29JdGFsaWFuIEJUTiBMaW5lZCBPYmxpcXVlLnR0Zg==.png',
 'Q29uY3Vyc29Nb2Rlcm5lIEJUTiBMdC50dGY=.png',
 'Q29uY3Vyc29Nb2Rlcm5lIEJUTiBXaWRlIE9ibGlxdWUudHRm.png',
 'Q29uZHVpdE9TSVRDLUxpZ2h0Lm90Zg==.png',
 'Q29vcEZvcmdlZC5wZmI=.png',
 'Q29vcGVyIEJsayBCVCBCbGFjay50dGY=.png',
 'Q29vcGVyIEl0YWxpYy50dGY=.png',
 'Q29vcGVyLnR0Zg==.png',
 'Q29vcGVyQlQtQmxhY2tIZWFkbGluZS5vdGY=.png',
 'Q29vcGVyQmxhRCBJdGFsaWMudHRm.png',
 'Q29wcGVycGxhdGUgQ29uZGVuc2VkIFNTaSBDb25kZW5zZWQudHRm.png',
 'Q29wcGVycGxhdGUgRXh0cmEgQ29uZGVuc2VkIFNTaSBCb2xkIEV4dHJhIENvbmRlbnNlZC50dGY=.png',
 'Q29wcGVycGxhdGUudHRm.png',
 'Q29wcGVycGxhdGVCUS1IZWF2eUV4dGVuZGVkLm90Zg==.png',
 'Q29wcGVycGxhdGVFRi1MaWdodC5vdGY=.png',
 'Q29wcGVycGxhdGVHb3RoaWNTdGQtMzJCQy5vdGY=.png',
 'Q29wcGVycGxhdGVULUJvbGRDb25kZW5zZWQub3Rm.png',
 'Q29weTA5NjUudHRm.png',
 'Q29wYSBTaGFycCBCVE4gU2hhZG93LnR0Zg==.png',
 'Q29yaW50aGlhbi50dGY=.png',
 'Q29yb25ldC1TZW1pQm9sZC1JdGFsaWMgRXgudHRm.png',
 'Q29yb25ldC50dGY=.png',
 'Q29ybmV0IFNjcmlwdC50dGY=.png',
 'Q29ycG9BLnR0Zg==.png',
 'Q29ycG9TLnR0Zg==.png',
 'Q29ycG9TTGlnIEJvbGQudHRm.png',
 'Q29ycG9yYXRlIEEgRXhwZXJ0IEl0YWxpYyBPc0YudHRm.png',
 'Q29ycG9yYXRlIEEgRXhwZXJ0IExpZ2h0LnR0Zg==.png',
 'Q29ycG9yYXRlIEhRLnR0Zg==.png',
 'Q29ycG9yYXRlIEUga3Vyc2l2IGhhbGJmZXR0LnR0Zg==.png',
 'Q29ycG9yYXRlIFMgRXhwZXJ0IEV4dHJhIEJvbGQgSXRhbGljIE9zRi50dGY=.png',
 'Q29ycG9yYXRlIFMgUmVndWxhci50dGY=.png',
 'Q29ycG9yYXRlQVNDLUJvbGQub3Rm.png',
 'Q29ycG9yYXRlRUV4cGVydEJRLUJvbGRJdGFsaWNPc0Yub3Rm.png',
 'Q29ycG9yYXRlU0V4cGVydEJRLUxpZ2h0Lm90Zg==.png',
 'Q29ycG9yYXRlUy1EZW1pSXRhbGljLm90Zg==.png',
 'Q29ycGlkLUxpZ2h0SXRhbGljLm90Zg==.png',
 'Q29ycGlkQ2RMRi1Cb2xkSXRhbGljLm90Zg==.png',
 'Q29ycGlkQ2RMRi1MaWdodC5vdGY=.png',
 'Q29ydGluLnR0Zg==.png',
 'Q29yYmVpIFVuY2lhbC50dGY=.png',
 'Q29yZGlhIE5ldyBCb2xkIEl0YWxpYy50dGY=.png',
 'Q29yZGlhVVBDLnR0Zg==.png',
 'Q29zbW9zQlEtTGlnaHRJdGFsaWMub3Rm.png',
 'Q29zdGFQdGYtRGVtaS5vdGY=.png',
 'Q29zdGFQdGYtRXh0cmFCb2xkLm90Zg==.png',
 'Q2dFZ2l6aWFub0JsLnR0Zg==.png',
 'Q2dGaWVkbGVyR290aGljLUJvbGQudHRm.png',
 'Q2dGdXR1cmFNYXhpTHQudHRm.png',
 'Q2dNb2Rlcm5Ud2VudHkudHRm.png',
 'Q2dQaGVuaXhBbWVyaWNhbi50dGY=.png',
 'Q2dZZWFyYm9va091dGxpbmUudHRm.png',
 'Q2dZZWFyYm9va0ZpbGxlci50dGY=.png',
 'Q2F0aGFyc2lzIENhcmdvLnR0Zg==.png',
 'Q2F0dWxsIChSKSBNZWRpdSBPc0YudHRm.png',
 'Q2F2ZSBHeXJsLnR0Zg==.png',
 'Q2F4dG9uIEJvbGQgSXRhbGljLnBmYg==.png',
 'Q2F4dG9uIExpZ2h0IEl0YWxpYyBCVC50dGY=.png',
 'Q2F4dG9uLUJvbGRJdGFsaWMub3Rm.png',
 'Q2F4dG9uU3RkLUJvbGRJdGFsaWMub3Rm.png',
 'Q2FiYXJnYUN1cnNJQ0cub3Rm.png',
 'Q2Fjb3Bob255IExvdWQudHRm.png',
 'Q2FlY2lsaWEtSGVhdnlJdGFsaWNPc0Yub3Rm.png',
 'Q2FlY2lsaWEtTGlnaHRJdGFsaWMub3Rm.png',
 'Q2FlY2lsaWFMVFN0ZC1MaWdodC5vdGY=.png',
 'Q2FmZU5vaXJTaGFkb3cudHRm.png',
 'Q2Fpcm9FeHRlbmRlZEl0YWxpYyBJdGFsaWMudHRm.png',
 'Q2FsaWd1bGEgUmVndWxhci50dGY=.png',
 'Q2FsaXBlciBXaWRlLnR0Zg==.png',
 'Q2FsaXMgaW4gUHVwcGV0bGFuZC50dGY=.png',
 'Q2FsbGlncmFwaDgxMCBCVCBSb21hbi50dGY=.png',
 'Q2Fsdmlub0hhbmQudHRm.png',
 'Q2FsdmVydCBNVCBCb2xkLnR0Zg==.png',
 'Q2FsdmVydCBNVCBMaWdodC50dGY=.png',
 'Q2FsdmVydE1ULm90Zg==.png',
 'Q2FsY3VsdXNMQ0RDYW1lby5vdGY=.png',
 'Q2FsYW1pdHkgVGVlbiBCVE4gQm9sZC50dGY=.png',
 'Q2FsZ2FyeS1MaWdodC5vdGY=.png',
 'Q2FtYnJpZGdlLURlbWlCb2xkLm90Zg==.png',
 'Q2FtYnJpZGdlLUxpZ2h0Lm90Zg==.png',
 'Q2FtZWxpYSBSZWd1bGFyLnR0Zg==.png',
 'Q2FtZWxsaWFELnR0Zg==.png',
 'Q2FudG9yaWEgTVQgRXh0cmFCb2xkLnR0Zg==.png',
 'Q2FudG9yaWFNVFN0ZC1Cb2xkLm90Zg==.png',
 'Q2FudXRoLnR0Zg==.png',
 'Q2FuY2VsbGFyZXNjYSBTY3JpcHQudHRm.png',
 'Q2FuY2VsbGFyZXNjYVNjcmlwdFBsYWluLm90Zg==.png',
 'Q2FuYWRpYW5QaG90b2dyYXBoZXIub3Rm.png',
 'Q2FuZGlkYSBCVCBJdGFsaWMudHRm.png',
 'Q2FuZGlkYUJULUl0YWxpYy5vdGY=.png',
 'Q2FuZHkgQ2FuZSBNYXRjaC50dGY=.png',
 'Q2FuZHkgU3F1YXJlIEJUTiBDb25kLnR0Zg==.png',
 'Q2FuZHkgU3RyaXBlIChCUkspLnR0Zg==.png',
 'Q2FwaXRhbHMudHRm.png',
 'Q2FwdGFpbiBTaGluZXIudHRm.png',
 'Q2Fyb2xzQ2h1bmtzLnR0Zg==.png',
 'Q2FybGEgQm9sZC50dGY=.png',
 'Q2FybGlzbGUgUmVndWxhci50dGY=.png',
 'Q2FybmF0aSBTU2kgSXRhbGljLnR0Zg==.png',
 'Q2Fybml2YWwub3Rm.png',
 'Q2FybWluYSBNZCBCVCBNZWRpdW0udHRm.png',
 'Q2FycGFsIFR1bm5lbC50dGY=.png',
 'Q2FycmVOb2lyU3RkLUJvbGRJdGFsaWMub3Rm.png',
 'Q2FydG9vbiBXaWRlLnR0Zg==.png',
 'Q2FydGllckJvb2tTdGQtTWVkaXVtLm90Zg==.png',
 'Q2FyYm9uIEJsb2NrLnR0Zg==.png',
 'Q2FyZ28gVHdvIFNGLnR0Zg==.png',
 'Q2FyZ29ELnR0Zg==.png',
 'Q2FyZGluYWwgUmVndWxhci50dGY=.png',
 'Q2FzaEVGLU1vbm9zcGFjZS5vdGY=.png',
 'Q2FzbG9uIDU0MCBJdGFsaWMucGZi.png',
 'Q2FzbG9uIEJvb2sgQkUgQm9sZC50dGY=.png',
 'Q2FzbG9uIENhbGxpZ3JhcGhpYyBJbml0aWFscy50dGY=.png',
 'Q2FzbG9uMjI0SVRDYnlCVC1Cb2xkSXRhbGljLm90Zg==.png',
 'Q2FzbG9uQm9va0JFLUJvbGRPc0Yub3Rm.png',
 'Q2FzbG9uQm9vay5vdGY=.png',
 'Q2FzbG9uQzM3LUxndEl0bEFsdC5vdGY=.png',
 'Q2FzbG9uSC1TQy1JdGFsaWMub3Rm.png',
 'Q2FzbG9uSUNHLVRpdGxpbmcub3Rm.png',
 'Q2FzbG9uT2xkRmFjZSBCVCBSb21hbi50dGY=.png',
 'Q2FzbG9uT2xkRmFjZUJULVJvbWFuLm90Zg==.png',
 'Q2FzbG9uVHdvVHdlbnR5Rm91ci1CbGFja0l0Lm90Zg==.png',
 'Q2FzbG9uVHdvVHdlbnR5Rm91ci1NZWRpdW0ub3Rm.png',
 'Q2FzbG9uVHdvVHdlbnR5Rm91ckJRLU1lZGl1bS5vdGY=.png',
 'Q2Fzc2FuZHJhLnR0Zg==.png',
 'Q2Fzc2FuZHJhRUYtQm9sZC5vdGY=.png',
 'Q2FzdG9yZ2F0ZSAtIE1lc3NlZC50dGY=.png',
 'Q2FzdG9yZ2F0ZSAtIFJvdWdoLnR0Zg==.png',
 'Q2FzdGxlVC1Cb2xkLm90Zg==.png',
 'Q2FzdWFsLnR0Zg==.png',
 'Q2FzIE9wZW4gRmFjZSBOb3JtYWwudHRm.png',
 'Q2FzYWJsYW4tRXh0cmFCb2xkLm90Zg==.png',
 'Q2h1cmNod2FyZEJydURSZWcudHRm.png',
 'Q2h1YmJ5T3V0bGluZS50dGY=.png',
 'Q2hhbGV0Qm9vayBCb2xkIEl0YWxpYy50dGY=.png',
 'Q2hhbGV0Qm9vayBCb2xkLm90Zg==.png',
 'Q2hhbmNlcnkgU2NyaXB0IE1lZGl1bSBTU2kgTWVkaXVtLnR0Zg==.png',
 'Q2hhbmV5IEV4dGVuZGVkIE5vcm1hbC50dGY=.png',
 'Q2hhbnNvbiBIZWF2eSBTRiBCb2xkIEl0YWxpYy50dGY=.png',
 'Q2hhbnRpbGx5LU1lZGl1bS5vdGY=.png',
 'Q2hhbnRpbGx5LVhsaWdodEl0YS5vdGY=.png',
 'Q2hhbXBhZ25lSXRhbGljLm90Zg==.png',
 'Q2hhc2xpbmUtQm9sZC5vdGY=.png',
 'Q2hhc2xpbmUtT2JsaXF1ZS5vdGY=.png',
 'Q2hhcGFycmFsUHJvLUxpZ2h0SXRDYXB0Lm90Zg==.png',
 'Q2hhcmxpZSdzIEFuZ2xlcyBPdXRHcmFkaWVudC50dGY=.png',
 'Q2hhcmxvdHRlU3RkLUJvb2sub3Rm.png',
 'Q2hhcnJpbmd0b24gV2lkZS50dGY=.png',
 'Q2hhcnRlciBJVEMgQm9sZCBCVC50dGY=.png',
 'Q2hhcnRlciBJVEMgSXRhbGljIEJULnR0Zg==.png',
 'Q2hhcnRlciBJVEMgUm9tYW4gQlQudHRm.png',
 'Q2hhcnRlciBPU0YgQlQgQmxhY2sgSXRhbGljLnR0Zg==.png',
 'Q2hhcnRlcklUQ2J5QlQtQm9sZEl0YWxpYy5vdGY=.png',
 'Q2hhcnRlcklUQy1SZWd1SXRhbC5vdGY=.png',
 'Q2hhcnRlckVGLVJlZ3VsYXJJdGFsaWNPc0Yub3Rm.png',
 'Q2hheiBXaWRlIE5vcm1hbC50dGY=.png',
 'Q2hlbHNleSBCb2xkIEl0YWxpYy50dGY=.png',
 'Q2hlbHNleSBDb25kZW5zZWQgQm9sZCBJdGFsaWMudHRm.png',
 'Q2hlbHNleSBFeHRlbmRlZCBCb2xkIEl0YWxpYy50dGY=.png',
 'Q2hlbHNleSBXaWRlIE5vcm1hbC50dGY=.png',
 'Q2hlbHNlYS1Cb29rLm90Zg==.png',
 'Q2hlbHRlbmhhbS1Cb29rQ29uZEl0YWxpYy5vdGY=.png',
 'Q2hlbHRlbmhhbS1Cb2xkQ29uZEl0YWxpYy5vdGY=.png',
 'Q2hlbHRlbmhhbS1VbHRyYUNvbmQub3Rm.png',
 'Q2hlbHRlbmhhbSBCb2xkIEl0YWxpYyBCVC50dGY=.png',
 'Q2hlbHRlbmhhbSBJdGFsaWMucGZi.png',
 'Q2hlbHRlbmhhbSBJdGMgVCBFRSBCb2xkLnBmYg==.png',
 'Q2hlbHRlbmhhbU9sZFN0eWxlRUYub3Rm.png',
 'Q2hlbHRlbmhhbUFURk9sZHN0eWxlQlEtSXRhbGljLm90Zg==.png',
 'Q2hlbHRlbmhhbUJULUl0YWxpYy5vdGY=.png',
 'Q2hlbHRlbmhhbUlUQ0JRLUJvb2sub3Rm.png',
 'Q2hlbHRlbmhhbUlUQ2J5QlQtQm9va0l0YWxpYy5vdGY=.png',
 'Q2hlbHRlbmhhbVN0ZC1Cb2xkLm90Zg==.png',
 'Q2hlbHRlbmhtIEJUIEJvbGQudHRm.png',
 'Q2hlbHRlbmhtIEJUIEl0YWxpYy50dGY=.png',
 'Q2hlbHRlbmhtIEJUIFJvbWFuLnR0Zg==.png',
 'Q2hlbHRlbmhtIFhCZENuIEJUIEJvbGQudHRm.png',
 'Q2hlbmdhbHVsdS5vdGY=.png',
 'Q2hlbWljYWwgUmVhY3Rpb24gQiAtQlJLLS50dGY=.png',
 'Q2hlc3RlcmZpZWxkQW50RC50dGY=.png',
 'Q2hlcmllIElUQy50dGY=.png',
 'Q2hldmFsaWVyQ2Fwc0JRLVJlZ3VsYXIub3Rm.png',
 'Q2hldmFsaWVyU3RyU0NELnR0Zg==.png',
 'Q2hpbGFkYUlDRy1Vbm8ub3Rm.png',
 'Q2hpbGQncyBQbGF5LnR0Zg==.png',
 'Q2hpbGRzUGxheS1BZ2VOaW5lLm90Zg==.png',
 'Q2hpbmVzZSBCcnVzaC50dGY=.png',
 'Q2hpbnllbiAgTm9ybWFsLnR0Zg==.png',
 'Q2hpc2VsIFdpZGUgQm9sZCBJdGFsaWMudHRm.png',
 'Q2hpc2VsIFdpZGUgSXRhbGljLnR0Zg==.png',
 'Q2hpc2VsIFRoaW4gQm9sZCBJdGFsaWMudHRm.png',
 'Q2hpY29yeS50dGY=.png',
 'Q2hpY2Fnb0xhc2VyIE1lZGl1bS50dGY=.png',
 'Q2hpY2FuZSBSZWd1bGFyLnR0Zg==.png',
 'Q2hpYW50aSBHWCBCVC50dGY=.png',
 'Q2hpYW50aSBYQmRJdCBPU0YgQlQgRXh0cmEgQm9sZCBJdGFsaWMudHRm.png',
 'Q2hvcml6by50dGY=.png',
 'Q2hvcmQtQmxhY2sub3Rm.png',
 'Q2hvcmQtQmxhY2tJdGFsaWMub3Rm.png',
 'Q2hvcmVhIERpc3BsYXkgU1NpIEJsYWNrLnR0Zg==.png',
 'Q2hvd2RlcmhlYWQudHRm.png',
 'Q2hvY0lDRy5vdGY=.png',
 'Q2hvZGEudHRm.png',
 'Q2hyaXN0b3Boc1F1aWxsSVRDU3RkLUJvbGQub3Rm.png',
 'Q2hyaXN0bWFzR2lmdFNjcmlwdEEudHRm.png',
 'Q2hyaXN0bWFzR2lmdFNjcmlwdEJvbGRBLnR0Zg==.png',
 'Q2hyb21lWWVsbG93LnR0Zg==.png',
 'Q2hyb21vc29tZVJldmVyc2VkLUhlYXZ5Lm90Zg==.png',
 'Q2l0aXplbkxpZ2h0IFJlZ3VsYXIudHRm.png',
 'Q2l0aXplbkxpZ2h0Lm90Zg==.png',
 'Q2l0eS1MaWdodC5vdGY=.png',
 'Q2l0eSAoUikgTGlnaHQudHRm.png',
 'Q2l0eURFRU1lZC50dGY=.png',
 'Q2l0YWRlbC1JbmxpbmUub3Rm.png',
 'Q2lyY3VsYXIgRkMudHRm.png',
 'Q2lyY3VsYXRlIChCUkspLnR0Zg==.png',
 'Q2lyY3VtY2lzaW9uLUJvbGQub3Rm.png',
 'Q2lyY3VzIE5vcm1hbC50dGY=.png',
 'Q2VubmVyaWsgUGxhaW4udHRm.png',
 'Q2VudG8gRXh0ZW5kZWQgQm9sZEl0YWxpYy50dGY=.png',
 'Q2VudGF1ci1EZW1pQm9sZC5vdGY=.png',
 'Q2VudGVubmlhbC1CbGFjay5vdGY=.png',
 'Q2VudHVyeS1Cb29rLm90Zg==.png',
 'Q2VudHVyeS1TY2hvb2xib29rLU5vcm1hbC50dGY=.png',
 'Q2VudHVyeS1VbHRyYUl0YWxpYy5vdGY=.png',
 'Q2VudHVyeSA3MjUgQ29uZGVuc2VkIEJULnR0Zg==.png',
 'Q2VudHVyeSA3MzEgQm9sZC5wZmI=.png',
 'Q2VudHVyeSA3NTEgQm9sZCBJdGFsaWMucGZi.png',
 'Q2VudHVyeSBFeHBhbmRlZCBJdGFsaWMucGZi.png',
 'Q2VudHVyeSBHb3RoaWMudHRm.png',
 'Q2VudHVyeSBPbGRzdHlsZS5wZmI=.png',
 'Q2VudHVyeSBSZXRyb3NwZWN0aXZlIExpZ2h0IFNTaSBMaWdodCBJdGFsaWMudHRm.png',
 'Q2VudHVyeSBTY2hvb2xib29rIEdyZWVrIEJULnR0Zg==.png',
 'Q2VudHVyeSBTY2hvb2xib29rIEdyZWVrIEJvbGQgSW5jbGluZWQgQlQudHRm.png',
 'Q2VudHVyeSBTY2hvb2xib29rIEdyZWVrIEluY2xpbmVkIEJULnR0Zg==.png',
 'Q2VudHVyeSBTY2hvb2xib29rIEJvbGQgSXRhbGljLnBmYg==.png',
 'Q2VudHVyeSBTY2hvb2xib29rIEl0YWxpYy5wZmI=.png',
 'Q2VudHVyeSBTY2hvb2xib29rIEl0YWxpYyBCVC50dGY=.png',
 'Q2VudHVyeSBTY2hvb2xib29rIFNTaSBCb2xkIEl0YWxpYy50dGY=.png',
 'Q2VudHVyeSBTY2hvb2xib29rIFQgRUUgUmVndWxhciBJdGFsaWMucGZi.png',
 'Q2VudHVyeTcyNSBCbGsgQlQgQmxhY2sudHRm.png',
 'Q2VudHVyeTcyNSBCZENuIEJUIEJvbGQudHRm.png',
 'Q2VudHVyeTcyNUJULUl0YWxpYy5vdGY=.png',
 'Q2VudHVyeTcyNUJULVJvbWFuQ29uZGVuc2VkLm90Zg==.png',
 'Q2VudHVyeTczMSBCVCBJdGFsaWMudHRm.png',
 'Q2VudHVyeTczMUJULUJvbGRJdGFsaWNBLm90Zg==.png',
 'Q2VudHVyeU9sZFN0eWxlU3RkLVJlZ3VsYXIub3Rm.png',
 'Q2VudHVyeU9TTVRTdGQtQm9sZEl0YWxpYy5vdGY=.png',
 'Q2VudHVyeU9TTVRTdGQtSXRhbGljLm90Zg==.png',
 'Q2VudHVyeUlUQ0JRLUJvb2tJdGFsaWMub3Rm.png',
 'Q2VudHVyeUlUQ0JRLUxpZ2h0Lm90Zg==.png',
 'Q2VudHVyeUlUQ0NvbmRlbnNlZEJRLUJvb2sub3Rm.png',
 'Q2VudHVyeUlUQ0NvbmRlbnNlZEJRLUJvbGQub3Rm.png',
 'Q2VudHVyeUlUQ0NvbmRlbnNlZEJRLUxpZ2h0Lm90Zg==.png',
 'Q2VudHVyeUlUQ2J5QlQtQm9sZENvbmRJdGFsaWMub3Rm.png',
 'Q2VudHVyeUlUQ2J5QlQtQm9sZENvbmRlbnNlZC5vdGY=.png',
 'Q2VudHVyeUlUQ2J5QlQtTGlnaHRDb25kSXRhbGljLm90Zg==.png',
 'Q2VudHVyeUNvbmRCb29rSXRhbGljLm90Zg==.png',
 'Q2VudHVyeVN0ZC1VbHRyYUl0YWxpYy5vdGY=.png',
 'Q2VudHVyeVNjaG9vbC5vdGY=.png',
 'Q2VudHVyeVNjaG9vbGJvb2tCVC1Sb21hbi5vdGY=.png',
 'Q2VudHVyeVNjaG9vbGJvb2tFRi1Cb2wub3Rm.png',
 'Q2VyaWdvRUYtQm9sZEl0YWxpYy5vdGY=.png',
 'Q2xhaXJ2YXV4TFRTdGQub3Rm.png',
 'Q2xhc3NpY2EgT25lLnR0Zg==.png',
 'Q2xhc3NpY2EtSXRhbGljLnR0Zg==.png',
 'Q2xhc3NpY2FsR2FyYW1vbmRCVC1Sb21hbi5vdGY=.png',
 'Q2xhc3NpYyBUcmFzaCAxIEJSSy50dGY=.png',
 'Q2xhc3NpYzEwNjUudHRm.png',
 'Q2xhcml0eSBHb3RoaWMgU0YudHRm.png',
 'Q2xhcmVuZG9uIEh2IEJUIEhlYXZ5LnR0Zg==.png',
 'Q2xhcmVuZG9uIEJsYWNrLnBmYg==.png',
 'Q2xhcmVuZG9uQlEtQmxhY2sub3Rm.png',
 'Q2xhcmVuZG9uTVRTdGQub3Rm.png',
 'Q2xlb3BhdHJhLm90Zg==.png',
 'Q2xlcmljIEJsYWNrIFNTaSBCbGFjayBJdGFsaWMudHRm.png',
 'Q2xlcmZhY2UtQm9sZC5vdGY=.png',
 'Q2xlcmZhY2UtQm9sZEl0YS5vdGY=.png',
 'Q2xlcmZhY2UtRGVtaUJvbGQub3Rm.png',
 'Q2xldmVyIER1a2UgQlROIFNtb290aC50dGY=.png',
 'Q2xlYXJ2aWV3SHd5LTYtQi50dGY=.png',
 'Q2xlYXJmYWNlIElUQyBCbGFjay50dGY=.png',
 'Q2xlYXJmYWNlIElUQyBCbGFjayBJdGFsaWMudHRm.png',
 'Q2xlYXJmYWNlQVRGQlEtQm9sZC5vdGY=.png',
 'Q2xlYXJmYWNlR290aGljTFQtQmxhY2sub3Rm.png',
 'Q2xlYXJmYWNlR290aGljTFRTdGQtTGlnaHQub3Rm.png',
 'Q2xlYXJmYWNlTFQtSGVhdnkub3Rm.png',
 'Q2xlYXJmYWNlTVRTdGQtQm9sZC5vdGY=.png',
 'Q2xlYXJnb3RoaWMtRXh0cmFCb2xkLm90Zg==.png',
 'Q2xlYXJnb3RoaWMtUmVndWxhckl0YS5vdGY=.png',
 'Q2xlYXJseSBHb3RoaWMgSGVhdnkgSXRhbGljLnR0Zg==.png',
 'Q2xpcGUgT3Blbi50dGY=.png',
 'Q2xpcXVlLVNlcmlmQm9sZE9ibGlxdWUub3Rm.png',
 'Q2xpY2hlZUNFLU9ibGlxdWUub3Rm.png',
 'Q2xvaXN0ZXJPcGVuRmFjZSBCVC50dGY=.png',
 'Q2xvaXN0ZXJTdGQtT3BlbkZhY2Uub3Rm.png',
 'Q2xvdmVySVRDLm90Zg==.png',
 'Q2xvY2t3b3JrIFJlZ3VsYXIudHRm.png',
 'Q3J1bmNoeUZheFBob250Lm90Zg==.png',
 'Q3J1c3RpRXN0LnR0Zg==.png',
 'Q3J5c3RhbHNIYW5kIFJlZ3VsYXIudHRm.png',
 'Q3Jha29vbSEudHRm.png',
 'Q3JhenkgR2lybHogQmxvbmQgQlROLnR0Zg==.png',
 'Q3JhY2tkb3duIFIgLUJSSy0udHRm.png',
 'Q3Jld0N1dENhcHMgSXRhbGljLnR0Zg==.png',
 'Q3Jpc3RvTGlraWQgVHJ5b3V0LnR0Zg==.png',
 'Q3JpY2tldC50dGY=.png',
 'Q3JvaXNzYW50RUYub3Rm.png',
 'Q3Jvbm9zUHJvLUJvbGRDYXB0SXQub3Rm.png',
 'Q3Jvbm9zUHJvLUJvbGRTdWJoLm90Zg==.png',
 'Q3Jvbm9zUHJvLUNhcHRJdC5vdGY=.png',
 'Q3Jvbm9zUHJvLVJlZ3VsYXIub3Rm.png',
 'Q3Jvbm9zUHJvLVNlbWlib2xkSXQub3Rm.png',
 'Q3Jvc2J5c0hhbmQudHRm.png',
 'Q3Jvc3NvdmVyIEJvbGRPYmxpcXVlLnR0Zg==.png',
 'Q3Jvc3NvdmVyIEV4dHJhTGlnaHQudHRm.png',
 'Q3liZXJ3aGl6LnR0Zg==.png',
 'Q3ljbGluLnR0Zg==.png',
 'Q3V0dGVyUm9tYW4ub3Rm.png',
 'Q3VybHpNVFN0ZC1SZWd1bGFyLm90Zg==.png',
 'Q3VydmF0dXJlLVJvdW5kZWQub3Rm.png',
 'Q3VydmF0dXJlLVJvdW5kZWRJdGFsaWMub3Rm.png',
 'Q3VzaGluZ0Jvb2tJdGFsaWMub3Rm.png',
 'Q3VzaGluZ0l0Y1RFRUhlYS50dGY=.png',
 'Q3VzaGluZ0VGLUhlYXZ5SXRhbGljLm90Zg==.png',
 'Q3VzaGluZ1N0ZC1Cb29rSXRhbGljLm90Zg==.png',
 'Q3VzaGluZyBIZWF2eSBJdGFsaWMgQlQudHRm.png',
 'QklHQ1NIQUQgTGVmdHkudHRm.png',
 'QkRmYXplci5vdGY=.png',
 'QlJFQUtCRUFULm90Zg==.png',
 'QlZHSGZldHRLdXJzaXYub3Rm.png',
 'QlZHUGljdEVpbnMub3Rm.png',
 'Qm90b24gUmVndWxhci50dGY=.png',
 'Qm90b24tTGlnaHRJdGFsaWMub3Rm.png',
 'Qm90b25CUS1NZWRpdW1JdGFsaWMub3Rm.png',
 'Qm91Z2FuIEJsYWNrIFNTaSBCb2xkLnR0Zg==.png',
 'Qm91Z2FuIEJvb2sgU1NpIEJvb2sudHRm.png',
 'Qm91Z2FuIFNTaSBTZW1pIEJvbGQudHRm.png',
 'Qm94c3ByaW5nQS50dGY=.png',
 'Qm9iYmllc0hhbmQudHRm.png',
 'Qm9kaWUgTUYgRmxhZy50dGY=.png',
 'Qm9kb25pIE1UIEl0YWxpYy50dGY=.png',
 'Qm9kb25pIE9sZCBGYWNlIEJFIE1lZGl1bSBJdGFsaWMgT2xkc3R5bGUgRmlndXJlcy50dGY=.png',
 'Qm9kb25pIFN2dHlUd28gSVRDIFRUIEJvbGQudHRm.png',
 'Qm9kb25pIFNpeCBJVEMgVFQgQm9vay50dGY=.png',
 'Qm9kb25pIFNpeCBPUyBJVEMgVFQgQm9sZC50dGY=.png',
 'Qm9kb25pIFNTaS50dGY=.png',
 'Qm9kb25pLUJvbGRJdGFsaWMub3Rm.png',
 'Qm9kb25pLUl0YWxpYy5vdGY=.png',
 'Qm9kb25pQ2xhc3NpYy1Cb2xkSXRhbGljLm90Zg==.png',
 'Qm9kb25pQkUtTWVkaXVtQ24ub3Rm.png',
 'Qm9kb25pQkUtUmVndWxhci5vdGY=.png',
 'Qm9kb25pQlQtUm9tYW4ub3Rm.png',
 'Qm9kb25pQmF1ZXJCUS1SZWd1bGFyLm90Zg==.png',
 'Qm9kb25pQmVydENvbkJRLUJvbGRJdGFsaWMub3Rm.png',
 'Qm9kb25pQmVydGhvbGRCUS1Cb2xkSXRhbGljLm90Zg==.png',
 'Qm9kb25pQy1Cb2xkSXRhbGljLm90Zg==.png',
 'Qm9kb25pQy1JdGFsaWMub3Rm.png',
 'Qm9kb25pREJsYS50dGY=.png',
 'Qm9kb25pRUYtRGVtaUJvbGQub3Rm.png',
 'Qm9kb25pRUYtRGVtaUJvbGRJdGEub3Rm.png',
 'Qm9kb25pRUYtTGlnaHQub3Rm.png',
 'Qm9kb25pSUNHLm90Zg==.png',
 'Qm9kb25pT2xkRmFjZUJFLVJlZ3VsYXIub3Rm.png',
 'Qm9kb25pT2xkRmFjZUJRLUl0YWxpYy5vdGY=.png',
 'Qm9kb25pT2xkRmFjZUJRLVJlZ3VsYXIub3Rm.png',
 'Qm9kb25pT2xkRmFjZUV4cEJRLU1lZGl1bS5vdGY=.png',
 'Qm9kb25pTVRTdGQtSXRhbGljLm90Zg==.png',
 'Qm9kb25pU2l4SVRDLUJvb2tJdGFsT1Mub3Rm.png',
 'Qm9kb25pU2l4SVRDLUJvbGQub3Rm.png',
 'Qm9kb25pU2l4SVRDLUJvbGRJdGFsT1Mub3Rm.png',
 'Qm9kb25pU2l4SVRDU3RkLUJvbGQub3Rm.png',
 'Qm9kb25pU2V2SVRDLUJvb2tJdGFsLm90Zg==.png',
 'Qm9kb25pU3RkLUl0YWxpYy5vdGY=.png',
 'Qm9kb25pVHdlSVRDLUJvbGQub3Rm.png',
 'Qm9kb3hpLURlbWlCb2xkLm90Zg==.png',
 'Qm9kZWdhU2VyaWYtTWVkaXVtT2xkc3R5bGUub3Rm.png',
 'Qm9ndXNmbG93LnR0Zg==.png',
 'Qm9sZGZhY2VJdGFsaWMtLnR0Zg==.png',
 'Qm9sZGZhY2VJdGFsaWMtU2VtaUJvbGQtSXRhbGljLnR0Zg==.png',
 'Qm9uayBPdXRlcmN1dC50dGY=.png',
 'Qm9uZWhlYWQudHRm.png',
 'Qm9va21hbiBCVCBJdGFsaWMudHRm.png',
 'Qm9va21hbiBCVCBSb21hbi50dGY=.png',
 'Qm9va21hbiBPbGQgU3R5bGUgQm9sZCBJdGFsaWMudHRm.png',
 'Qm9va21hbiBSZWd1bGFyLnR0Zg==.png',
 'Qm9va21hbkJRLURlbWlCb2xkLm90Zg==.png',
 'Qm9va21hbkJRLURlbWlCb2xkSXRhbGljLm90Zg==.png',
 'Qm9va21hbklUQ1N0ZC1MaWdodEl0YWxpYy5vdGY=.png',
 'Qm9va21hblN0ZC1Cb2xkSXRhbGljLm90Zg==.png',
 'Qm9va2VyLm90Zg==.png',
 'Qm9vemxlIERpc3BsYXkgU1NpIEl0YWxpYy50dGY=.png',
 'Qm9yb3dheUJPbGQudHRm.png',
 'Qm9yem9pIE1lZGl1bS50dGY=.png',
 'Qm9yZGVhdXggSUNHLnR0Zg==.png',
 'Qm9yZGVhdXggSXRhbGljIFBsYWluLnR0Zg==.png',
 'Qm9yZGVhdXhNZWRpdW0ub3Rm.png',
 'Qm9yZGVsbG8tU2hhZGVkLm90Zg==.png',
 'Qm9zbmlhIEQgQm9sZC50dGY=.png',
 'Qm9zdG9uIFRyYWZmaWMudHRm.png',
 'QmF0aCBMaWdodEl0YWxpYy50dGY=.png',
 'QmF0YWtDb25kZW5zZWRJVENTdGRCb2xkLm90Zg==.png',
 'QmF1aGF1c0l0Y1RFRU1lZC50dGY=.png',
 'QmF1aGF1c0l0Y1RFRUJvbC50dGY=.png',
 'QmF1aGF1c0lUQ2J5QlQtTGlnaHQub3Rm.png',
 'QmF1aGF1cy1EZW1pLm90Zg==.png',
 'QmF1aGF1cyBCb2xkLnR0Zg==.png',
 'QmF1ZXIgQm9kb25pIEJvbGQgQ29uZGVuc2VkIEJULnR0Zg==.png',
 'QmF1ZXJCb2RvbmlCVC1CbGFjay5vdGY=.png',
 'QmF1ZXJCb2RvbmlCVC1UaXRsaW5nLm90Zg==.png',
 'QmF1ZXJUb3BpYy1CbGQub3Rm.png',
 'QmF5ZXJFeHBlcmltZW50LnR0Zg==.png',
 'QmFieSBLcnVmZnkudHRm.png',
 'QmFja3RhbGtTZXJpZiBCVE4gU0MgQm9sZE9ibGlxdWUudHRm.png',
 'QmFjY2FyYXRXaWRlIFJlZ3VsYXIudHRm.png',
 'QmFjY3VzIFJlZ3VsYXIudHRm.png',
 'QmFkIE1vZm8udHRm.png',
 'QmFkLnR0Zg==.png',
 'QmFpbGV5IFNhbnMgSVRDIEJvb2sudHRm.png',
 'QmFpbGV5U2Fuc0lUQ1N0ZC1Cb2xkLm90Zg==.png',
 'QmFrZXIgU2lnbmV0IEJULnR0Zg==.png',
 'QmFrZXJTaWduZXQgQlQgUm9tYW4udHRm.png',
 'QmFsbGFudGluZXMtRGVtaUJvbGQub3Rm.png',
 'QmFsbGFudGluZXNTY3JpcHRFRi1NZWRpdW0ub3Rm.png',
 'QmFsc2Ftby50dGY=.png',
 'QmFsdHJhR0Qub3Rm.png',
 'QmFsYUN5bnd5ZC50dGY=.png',
 'QmFsYW5jZS1MaWdodC5vdGY=.png',
 'QmFsYW5jZS1MaWdodENhcHNJdGFsaWMub3Rm.png',
 'QmFsYW5jZUxpZ2h0LUl0YWxpYy5vdGY=.png',
 'QmFsYW5jZUxpZ2h0LUNhcHMub3Rm.png',
 'QmFubmVyU3RkLm90Zg==.png',
 'QmFuY29JVENTdGQtSGVhdnkub3Rm.png',
 'QmFuY29JVENTdGQtTGlnaHQub3Rm.png',
 'QmFuZ2xlIENvbmRlbnNlZCBCb2xkIEl0YWxpYy50dGY=.png',
 'QmFycnlzSGFuZCBSZWd1bGFyLnR0Zg==.png',
 'QmFydCBIZWF2eSBJdGFsaWMudHRm.png',
 'QmFydCBIZWF2eSBOb3JtYWwudHRm.png',
 'QmFydCBJdGFsaWMudHRm.png',
 'QmFydCBUaGluIEhlYXZ5IEJvbGQudHRm.png',
 'QmFydCBUaGluIEJvbGRJdGFsaWMudHRm.png',
 'QmFyY29vbC50dGY=.png',
 'QmFyY2Vsb25hIEJvbGQudHRm.png',
 'QmFyY2Vsb25hSVRDU3RkLUJvb2tJdGFsaWMub3Rm.png',
 'QmFyY2Vsb25hSVRDU3RkLUJvbGQub3Rm.png',
 'QmFyY2xheSBPcGVuLnR0Zg==.png',
 'QmFyYmVyUG9sZSBSZWd1bGFyLnR0Zg==.png',
 'QmFza2VydmlsbGUgQm9sZCBJdCBXaW45NUJUKDEpLnR0Zg==.png',
 'QmFza2VydmlsbGUgQmxhY2sgU1NpIEJvbGQgSXRhbGljLnR0Zg==.png',
 'QmFza2VydmlsbGUgU1NpLnR0Zg==.png',
 'QmFza2VydmlsbGUtTm9ybWFsLUl0YWxpYy50dGY=.png',
 'QmFza2VydmlsbGVCUS1JdGFsaWMub3Rm.png',
 'QmFza2VydmlsbGVCUS1NZWRpdW0ub3Rm.png',
 'QmFza2VydmlsbGVCVC1Sb21hbi5vdGY=.png',
 'QmFza2VydmlsbGVFRi1NZWRpdW0ub3Rm.png',
 'QmFza2VydmlsbGVOZXdCUS1Cb2xkSXRhbGljLm90Zg==.png',
 'QmFza2VydmlsbGVULnR0Zg==.png',
 'QmFza2VydmlsbGVULVJlZ3Uub3Rm.png',
 'QmFza2VydmlsbGVULVJlZ3VJdGFsLm90Zg==.png',
 'QmFza2VydmxsZTIgQlQgUm9tYW4udHRm.png',
 'QmFzaWMgU2FucyBIZWF2eSBTRiBCb2xkLnR0Zg==.png',
 'QmFzaWMgU2FucyBTRiBCb2xkLnR0Zg==.png',
 'QmFzc2V0dCBUaGluIEJvbGRJdGFsaWMudHRm.png',
 'QmFzcXVlIFRoaW4gTm9ybWFsLnR0Zg==.png',
 'QmFzZU1vbm9XaWRlVGhpbiBSZWd1bGFyLnR0Zg==.png',
 'QmFzZU5pbmUgQm9sZC50dGY=.png',
 'QmFzZU5pbmVCSS50dGY=.png',
 'QmFzZU5pbmVTbWFsbENhcHMgQm9sZCBJdGFsaWMudHRm.png',
 'QmFzZU5pbmVTQ0IudHRm.png',
 'QmFzZVR3ZWx2ZVNlcmlmU0NCLnR0Zg==.png',
 'Qml0c3RyZWFtIEFtZXJpZ28gQm9sZC5wZmI=.png',
 'Qml0c3RyZWFtIEFtZXJpZ28gQm9sZCBJdGFsaWMucGZi.png',
 'Qml0c3RyZWFtIEFycnVzIEJsYWNrLnBmYg==.png',
 'Qml0c3RyZWFtIElvd2FuIE9sZCBTdHlsZS5wZmI=.png',
 'Qml0c3RyZWFtIENoYXJ0ZXIgQmxhY2sgSXRhbGljIE9TRi5wZmI=.png',
 'Qml0c3RyZWFtIENvb3BlciBMaWdodCBJdGFsaWMucGZi.png',
 'Qml0c3RyZWFtIFZlcmEgU2FucyBNb25vIE9ibGlxdWUudHRm.png',
 'Qml0c3RyZWFtIFZlcmEgU2VyaWYgQm9sZC50dGY=.png',
 'Qml0d2lzZS50dGY=.png',
 'Qml4bGVlLUhlYXZ5LnR0Zg==.png',
 'Qml4bGVlQ25kLUhlYXZ5LnR0Zg==.png',
 'Qmlja2hhbSBTY3JpcHQgVHdvLnR0Zg==.png',
 'QmlndG93bmVCb2xkLnR0Zg==.png',
 'QmlnIEJhY29uIFRyeW91dC50dGY=.png',
 'Qmlqb3V4LUJvbGQub3Rm.png',
 'QmlraW5pLnR0Zg==.png',
 'QmlsbGJvYXJkIDExIENvbmRlbnNlZCBOb3JtYWwudHRm.png',
 'QmlsbHV5LnR0Zg==.png',
 'QmlsYm9EaXNwbGF5IEJvbGQgSXRhbGljLnR0Zg==.png',
 'QmlubmVyRC50dGY=.png',
 'QmluZ28ub3Rm.png',
 'QmlvbmljIENvbWljIEV4cCBJdGFsaWMudHRm.png',
 'QmlvbmljIFR5cGUgQ29uZGVuc2VkLnR0Zg==.png',
 'QmlvbWVjaGFuaWMudHRm.png',
 'QmlyZC50dGY=.png',
 'QmV0b25FRi1FeHRyYUJvbGQub3Rm.png',
 'QmV0dGVyIE9mZiAxLnR0Zg==.png',
 'QmV0dGVyS2FtcCBCb2xkSXRhbGljLnR0Zg==.png',
 'QmV0dGVySW5ncmlhbmEgSXRhbGljLnR0Zg==.png',
 'QmV0dGVyVHlwZVJpZ2h0IEJvbGRJdGFsaWMudHRm.png',
 'QmV0YSBCbG9jay50dGY=.png',
 'QmV0YVNlbWktQm9sZC5vdGY=.png',
 'QmV0YWNhcGl0YWwudHRm.png',
 'QmVhbSBSaWRlciBFeHBhbmRlZC50dGY=.png',
 'QmVhc3QudHRm.png',
 'QmVhckJ1dHRlVCBQbGFpbi50dGY=.png',
 'QmVhdG5pa0hheXNlZWQudHRm.png',
 'QmVhdSBUaGluIEl0YWxpYy50dGY=.png',
 'QmVhdXJlZ2FyZCBEaXNwbGF5IFNTaS50dGY=.png',
 'QmVhY2ggVGhpbiBCb2xkLnR0Zg==.png',
 'QmVjY2FyaWEudHRm.png',
 'QmVla21hbi1TcXVhcmVCb2xkLm90Zg==.png',
 'QmVla21hbi1TcXVhcmVSZWd1bGFyLm90Zg==.png',
 'QmVlc2tuZWVzIFNDIElUQyBUVC50dGY=.png',
 'QmVlc2tuZWVzQy5vdGY=.png',
 'QmVldGhhbS1MaWdodC5vdGY=.png',
 'QmVsaXppb0JvbGQub3Rm.png',
 'QmVsbEdvdGhpY1N0ZC1CbGFjay5vdGY=.png',
 'QmVsbENlbnQgU3ViQ2FwIEJUIFN1Yi1DYXB0aW9uLnR0Zg==.png',
 'QmVsbENlbnRlbm5pYWwtTmFtZUFuZE51bWJlci5vdGY=.png',
 'QmVsbGFteUV4dHJhQm9sZC5vdGY=.png',
 'QmVsbGhvcE5GLnR0Zg==.png',
 'QmVsd2VFRi1Cb2xkLm90Zg==.png',
 'QmVsd2VFRi1Cb2xkQ29uZGVuc2VkLm90Zg==.png',
 'QmVsd2VTdGQtTGlnaHRJdGFsaWMub3Rm.png',
 'QmVsdWNpYW4tRGVtaUJvbGQub3Rm.png',
 'QmVtYm8gTU9OT1RZUEUgRXhwZXJ0IEJvbGQgT3NGLnR0Zg==.png',
 'QmVtYm8tU0Mub3Rm.png',
 'QmVtYm9JbmZhbnRNVFN0ZC1Cb2xkSXQub3Rm.png',
 'QmVtYm9UaXRsaW5nTVRTdGQub3Rm.png',
 'QmVuZ3VpYXQgRnJpc2t5IEFUVC50dGY=.png',
 'QmVuZ3VpYXRCUS1NZWRpdW0ub3Rm.png',
 'QmVuZ3VpYXRDb25kZW5zZWRCUS1Cb29rSXRhbGljLm90Zg==.png',
 'QmVuZ3VpYXRDb25kZW5zZWRCUS1NZWRpdW0ub3Rm.png',
 'QmVuZ3VpYXRFRi1Cb2xkSXRhbGljLm90Zg==.png',
 'QmVuZ3VpYXRHb3RoaWMtTWVkaXVtLm90Zg==.png',
 'QmVuZ3VpYXRHb3RoaWNCUS1Cb2xkSXRhbGljLm90Zg==.png',
 'QmVuZ3VpYXRHb3RoaWNCUS1NZWRpdW0ub3Rm.png',
 'QmVuZ3VpYXRHb3RoaWNMVC1IZWF2eS5vdGY=.png',
 'QmVuZ3VpYXRHb3RoaWNTdGQtTWVkaXVtLm90Zg==.png',
 'QmVuZ3VpYXRJdGNURUUgQm9sZCBJdGFsaWMudHRm.png',
 'QmVuZ3VpYXRJVENieUJULUJvb2tDb25kZW5zZWQub3Rm.png',
 'QmVya2VsZXkgT2xkc3R5bGUgSXRhbGljLnR0Zg==.png',
 'QmVya2VsZXlPbGRTdHlsZUVGLUJvbGRJdGFsaWMub3Rm.png',
 'QmVya2VsZXlPbGRzdHlsZUJvbGQub3Rm.png',
 'QmVya2VsZXlPbGRzdHlsZUlUQ2J5QlQtTWVkSXRhbC5vdGY=.png',
 'QmVya2VsZXlTdGQtQm9vay5vdGY=.png',
 'QmVybGluc2Fucy1Cb2xkRXhwZXJ0Lm90Zg==.png',
 'QmVybGluIFNhbnMgRkIudHRm.png',
 'QmVybGluU21hbGxDYXBzLnR0Zg==.png',
 'QmVybGluZ0JRLVJlZ3VsYXIub3Rm.png',
 'QmVybGluZ0JRLVNlbWlCb2xkLm90Zg==.png',
 'QmVybGluZ0VGLVJlZ3VsYXIub3Rm.png',
 'QmVybGluZXIgR3JvdGVzayBNZWRpdW0udHRm.png',
 'QmVybGluZXJHcm90ZXNrLUxpZ2h0Lm90Zg==.png',
 'QmVybmhhcmQgTW9kZXJuIEJULnR0Zg==.png',
 'QmVybmhhcmQgTW9kZXJuIEJvbGQgSXRhbGljIEJULnR0Zg==.png',
 'QmVybmhhcmRGYXNoaW9uIEJULnR0Zg==.png',
 'QmVybmhhcmRNb2Rlcm5CUS1JdGFsaWMub3Rm.png',
 'QmVybnlLbGluZ29uLnR0Zg==.png',
 'QmVybnN0ZWluLURlbWlCb2xkLm90Zg==.png',
 'QmVydGhhbS50dGY=.png',
 'QmVydGhhbUJvbGQudHRm.png',
 'QmVydGhvbGQgQWt6aWRlbnogR3JvdGVzayBCRSBCb2xkIEV4dGVuZGVkIEl0YWxpYy5wZmI=.png',
 'QmVydGhvbGQgQWt6aWRlbnogR3JvdGVzayBCRSBFeHRlbmRlZC5wZmI=.png',
 'QmVydGhvbGQgV2FsYmF1bSBCb29rIE1lZGl1bSBJdGFsaWMgT2xkc3R5bGUgRmlndXJlcy5wZmI=.png',
 'QmVydHJhbS50dGY=.png',
 'Qmx1ZUFyc2VuYWxPbmUudHRm.png',
 'Qmxhc3RlciBJbmZpbml0ZS50dGY=.png',
 'Qmxhdmlja2UgQ2FwaXRhbHMgU2VtaS1leHBhbmRlZCBSZWd1bGFyLnR0Zg==.png',
 'QmxhY2tCZWFyZC5vdGY=.png',
 'QmxhY2thZGRlciBJVEMgVFQudHRm.png',
 'QmxhY2tmb3JkIE9sZFN0eWxlIFNTaSBTbWFsbCBDYXBzLnR0Zg==.png',
 'QmxhY2tsZXR0ZXIgSFBMSFMudHRm.png',
 'QmxhZGUudHRm.png',
 'QmxhZGVDdXRUaHJ1LnR0Zg==.png',
 'QmxldyBFeHRlbmRlZCBJdGFsaWMudHRm.png',
 'QmxldyBXaWRlIEJvbGQgSXRhbGljLnR0Zg==.png',
 'QmxpbmRmaXNoTGlnaHQub3Rm.png',
 'Qmxpc3MgQm9sZC50dGY=.png',
 'Qmxpc3MgRXh0ZW5kZWQgTm9ybWFsLnR0Zg==.png',
 'QmxpcHBvIEJsYWNrLnBmYg==.png',
 'Qmxvb2R5LnR0Zg==.png',
 'QmxvdHRvb280MG96LnR0Zg==.png',
 'QmxvY0Mub3Rm.png',
 'QmxvY2tCRS1SZWd1bGFyLm90Zg==.png',
 'QmxvY2tVcC50dGY=.png',
 'QnJ1aG5TY3JpcHQtU3RyYWlnaHRlbmVkLm90Zg==.png',
 'QnJ1aXNlZCBUd2VudHlGaXZlLnR0Zg==.png',
 'QnJ1bm9KQi5vdGY=.png',
 'QnJ1c2g0NTUgQlQudHRm.png',
 'QnJ1c2ggNDQ1IEJULnR0Zg==.png',
 'QnJ1c2hCUS5vdGY=.png',
 'QnJ1c2hTY3JpcHRCVC1SZWd1bGFyLm90Zg==.png',
 'QnJ1c2htYW4udHRm.png',
 'QnJ1Y2VzSGFuZCBSZWd1bGFyLnR0Zg==.png',
 'QnJhaW5oZWFkLnR0Zg==.png',
 'QnJhbmRpbmcgSXJvbi50dGY=.png',
 'QnJhbmRvIEJvbGRJdGFsaWMudHRm.png',
 'QnJhbmRvIENvbmRlbnNlZCBOb3JtYWwudHRm.png',
 'QnJhbmRvIEVuZ3JhdmVkIENvbmRlbnNlZCBCb2xkLnR0Zg==.png',
 'QnJhc3NldHRfQm9sZC50dGY=.png',
 'QnJhc3NmaWVsZCBSZWd1bGFyLnR0Zg==.png',
 'QnJlbGEtVGhpbi50dGY=.png',
 'QnJlYWtiZWF0IEJUTiBPdXRsaW5lLnR0Zg==.png',
 'QnJpb3NvUHJvLUJvbGRJdERpc3Aub3Rm.png',
 'QnJpbmcgdGhhIG5vaXplLnR0Zg==.png',
 'QnJpc2sudHRm.png',
 'QnJpZGdld29ya0EudHRm.png',
 'QnJpZGdld29yay5vdGY=.png',
 'QnJpZGdub3J0aF9CbG9ja2VkLnR0Zg==.png',
 'QnJpZW1Ba2FkZW1pU3RkLVJlZ3VsYXIub3Rm.png',
 'QnJpZW1TY3JpcHRTdGQtQm9sZC5vdGY=.png',
 'QnJpZW1TY3JpcHRTdGQtQmxhY2sub3Rm.png',
 'QnJpZWluY2FybmF0aW9uLnR0Zg==.png',
 'QnJvdGhlcnNTdXBlclNsYW50Lm90Zg==.png',
 'QnJvY2h1cmUudHRm.png',
 'QnJvYWR3YXlFbmdyYXZlZEJULVJlZ3VsYXIub3Rm.png',
 'QnJvYWR3YXlGb250Lm90Zg==.png',
 'QnJvYWR3YXlNVFN0ZC5vdGY=.png',
 'QnJvYWR3YXlPdXRELnR0Zg==.png',
 'QnJvYWR3YXlQLnR0Zg==.png',
 'QnJvZHkgUmVndWxhci50dGY=.png',
 'Qnl0ZSBQb2xpY2UudHRm.png',
 'QnV0dGVyZmx5IENocm9tb3NvbWUgQU9FLnR0Zg==.png',
 'QnV6emVyVGhyZWVTdGQub3Rm.png',
 'QnViYmFMb3ZlLUJvbGQub3Rm.png',
 'QnViYmxlZG90SUNHRmluZVBvc2l0aXZlLnR0Zg==.png',
 'QnVjY2FyZGlTdGQtQm9sZC5vdGY=.png',
 'QnVkSGFuZCBSZWd1bGFyLnR0Zg==.png',
 'QnVsbC1VbmRlcklua2VkLm90Zg==.png',
 'QnVsbWVyLUl0bC5vdGY=.png',
 'QnVsbWVyTVQtQm9sZERpc3BsYXkub3Rm.png',
 'QnVsbWVyTVRTdGQtQm9sZERpc3BsYXkub3Rm.png',
 'QnVuZHkgWWVsbG93IEhvbGxvd1NoYWRvd2VkLnR0Zg==.png',
 'QnVyaW5TYW5zU3RkLm90Zg==.png',
 'QnVyb2tyYXQtVGhyZWUub3Rm.png',
 'QnVybm91dEEudHRm.png',
 'QnVybmluZyBMaWdodC50dGY=.png',
 'QnVyYmFua0JpZ1dpZGUtTGlnaHQub3Rm.png',
 'QnVyYmFua0JpZ1JlZ3VsYXItQmxhY2sub3Rm.png',
 'QnVyZWF1R3JvdFRocmVlU2V2ZW4ub3Rm.png',
 'QnVzb3JhbWFMaWdodC5vdGY=.png',
 'QnVzdGVyLUJvbGRDb25kZW5zZWQub3Rm.png',
 'QSBNb2JpbGUgTGlmZS50dGY=.png',
 'QUdCdWNoUm91bmRlZEJRLUJvbGRPdXRsaW5lLm90Zg==.png',
 'QUdfQ29vcGVyIEl0YWxpYy50dGY=.png',
 'QUdhcmFtb25kLUJvbGQub3Rm.png',
 'QUdPbGRGYWNlQlEtQm9sZE91dGxpbmUub3Rm.png',
 'QUdPbGRGYWNlQlEtT3V0bGluZS5vdGY=.png',
 'QUdSZXZ1ZUN5ciBSb21hbiBNZWRpdW0udHRm.png',
 'QUJTNy50dGY=.png',
 'QUJTOS50dGY=.png',
 'QUNhc2xvbi1SZWd1bGFyU0Mub3Rm.png',
 'QUplbnNvblByby1Cb2xkLm90Zg==.png',
 'QURNT05PLVJlZ3VsYXIudHRm.png',
 'QUxFWDIub3Rm.png',
 'QUZDYXJwbGF0ZXMtQm9sZC5vdGY=.png',
 'QVRaYXBmSW50ZXJuYXRpb25hbC1MaWdodEl0Lm90Zg==.png',
 'QVRCcmFtbGV5LU1lZGl1bS5vdGY=.png',
 'QVRTbGltYmFjaC1NZWRpdW1JdGFsaWMub3Rm.png',
 'QW10eXBlLnR0Zg==.png',
 'QW1hbmRhc0hhbmQgUmVndWxhci50dGY=.png',
 'QW1hc2lzIE1UIEJvbGQudHRm.png',
 'QW1lbGlhTEwub3Rm.png',
 'QW1lcmljYW4gUG9wIFBsYWluLnR0Zg==.png',
 'QW1lcmljYW4gVHlwZXdyaXRlciBNZWRpdW0gQlQudHRm.png',
 'QW1lcmljYW5Hb3RVUldUTWVkIEl0YWxpYy50dGY=.png',
 'QW1lcmljYW5hLUJvbGQub3Rm.png',
 'QW1lcmljYW5hU3RkLUJvbGQub3Rm.png',
 'QW1lcmljYW5HYXJhbW9uZEJULUl0YWxpYy5vdGY=.png',
 'QW1lcmljYW5UeXBld3JpdGVyLUxpZ2h0Q29uZEEub3Rm.png',
 'QW1lcmljYW5UeXBld3JpdGVyQ29uQlEtQm9sZC5vdGY=.png',
 'QW1lcmljYW5UeXBld3JpdGVyQlEtQm9sZEl0YWxpYy5vdGY=.png',
 'QW1lcmljYW5UeXBld3JpdGVyQlEtTWVkaXVtLm90Zg==.png',
 'QW1lcmljYW5UeXBld3JpdGVyTFQtTGlnaHRBLm90Zg==.png',
 'QW1lcmljYW5UeXBld3JpdGVyU3RkLUx0Q25kLm90Zg==.png',
 'QW1lcmljYW5UeXBlQm9sZC5vdGY=.png',
 'QW1lcmlnbyBCVCBCb2xkIEl0YWxpYy50dGY=.png',
 'QW1lcmlnbyBCVCBCb2xkLnR0Zg==.png',
 'QW1lcmV0dG8gV2lkZSBCb2xkLnR0Zg==.png',
 'QW1pbnRhQm9sZC50dGY=.png',
 'QW1UeXBld3JpdGVyRUYtQm9sZEl0YWxpYy5vdGY=.png',
 'QW1UeXBld3JpdGVyRUYtTGlnaHQub3Rm.png',
 'QW1vcyBFeHRlbmRlZCBJdGFsaWMudHRm.png',
 'QW1wb3VsZSBIZWF2eS50dGY=.png',
 'QW1wbGl0dWRlQ29tcC1MaWdodC50dGY=.png',
 'QW1wbGl0dWRlQ29tcC1VbHRyYS50dGY=.png',
 'QW1wbGl0dWRlRXh0cmFDb21wLU1lZGl1bS50dGY=.png',
 'QW1wbGl0dWRlRXh0cmFDb21wLUJvbGQudHRm.png',
 'QW1wbGlmaWVyQm9sZEV4dGVuc2lvbnMub3Rm.png',
 'QW1wbGlmaWVyTGlnaHRTbWFsbENhcHMub3Rm.png',
 'QW4gaXJyaXRhdGluZyBzcGVjay50dGY=.png',
 'QW50aHJvUG9zb3BoLUJvbGQudHRm.png',
 'QW50aW1vbnlCbHVlLm90Zg==.png',
 'QW50aXF1YSAxMDEgQ29uZGVuc2VkIEl0YWxpYy50dGY=.png',
 'QW50aXF1YSAxMDEgTm9ybWFsLnR0Zg==.png',
 'QW50aXF1YSAxMDEgV2lkZSBCb2xkLnR0Zg==.png',
 'QW50aXF1ZSBPbGl2ZSBCbGFjay5wZmI=.png',
 'QW50aXF1ZU1vZGVybmUtUmVndWxhci5vdGY=.png',
 'QW50aXF1ZU9saVNDVC1SZWd1Lm90Zg==.png',
 'QW50aXF1ZU9saVQtQ29tcEl0YWwub3Rm.png',
 'QW50aXF1ZU9saVQtQm9sZENvbmRJbjEub3Rm.png',
 'QW50aXF1ZU9saXZlLUl0YWxpYy5vdGY=.png',
 'QW50aXF1ZU9saXZlU3RkLUl0YWxpYy5vdGY=.png',
 'QW50aXF1ZUFuY2llbm5lQ0UtSXRhbGljLm90Zg==.png',
 'QW5Ba3JvbmlzbS50dGY=.png',
 'QW5hbGdlc2ljcy50dGY=.png',
 'QW5hcmNoaXN0aWMudHRm.png',
 'QW5hdG9sZSBEaXNwbGF5IFNTaS50dGY=.png',
 'QW5kcmV3QW5keUthY3R1cy50dGY=.png',
 'QW5kcmV3U2NyaXB0LnR0Zg==.png',
 'QW5kYWxlTW9ub01UU3RkLUJvbGQub3Rm.png',
 'QW5kZXJzb24ub3Rm.png',
 'QW5nbG8tU2F4b24gQ2Fwcy50dGY=.png',
 'QW5nZWwgRm9udC50dGY=.png',
 'QW5nZWxvLm90Zg==.png',
 'QW5pc2V0dGUtTGlnaHQub3Rm.png',
 'QW5uYSBJQ0cudHRm.png',
 'QW5uYWJlbGxlIEpGLnR0Zg==.png',
 'QWdlbmN5RkItQm9sZENvbmRlbnNlZC5vdGY=.png',
 'QWdlbmN5RkItQmxhY2tDb25kZW5zZWQub3Rm.png',
 'QWdlbmRhdHlwZS1SZWd1bGFyLm90Zg==.png',
 'QWdlbmRhdHlwZVN3YXNoLUJvbGRJdGFsaWMub3Rm.png',
 'QWdlbmRhLUxpZ2h0Lm90Zg==.png',
 'QWdlbmRhLUxpZ2h0RXh0cmFDb25kZW5zZWQub3Rm.png',
 'QWdmYVdpbGVSb21hblN0ZC1Cb2xkLm90Zg==.png',
 'QWdmYVJvdGlzU2Fuc1NlcmlmRXh0cmFCb2xkLm90Zg==.png',
 'QWdmYVJvdGlzU2VtaXNhbnNMaWdodC1JdGFsaWMub3Rm.png',
 'QWdmYVJvdGlzU2VyaWYub3Rm.png',
 'QWF1eCBQcm9NZWRpdW0gSXRhbGljIE9TRi50dGY=.png',
 'QWF1eCBQcm9SZWd1bGFyIEl0YWxpYyBTQy50dGY=.png',
 'QWF1eCBQcm9UaGluIE9TRi50dGY=.png',
 'QWFiY2VkIFJlZ3VsYXIudHRm.png',
 'QWFiY2VkWEJvbGQudHRm.png',
 'QWFjaGVuIExUIEJvbGQudHRm.png',
 'QWFjaGVuLnBmYg==.png',
 'QWFyb25Cb2xkLnR0Zg==.png',
 'QWFyZHZhcmstUmVndWxhci5vdGY=.png',
 'QWhhcm9uaSBCb2xkKDEpLnR0Zg==.png',
 'QWJhZGkgTVQgQ29uZGVuc2VkIExpZ2h0LnR0Zg==.png',
 'QWJhZGlNVFN0ZC1Cb2xkSXRhbGljLm90Zg==.png',
 'QWJhZGlNVFN0ZC1FeHRyYUxpZ2h0Lm90Zg==.png',
 'QWJiZXkgTWVkaXVtIEV4dGVuZGVkLnR0Zg==.png',
 'QWJjUGhvbmljc1R3by50dGY=.png',
 'QWJkdWN0aW9uMjAwMi50dGY=.png',
 'QWJpbGVuZUZMRi5vdGY=.png',
 'QWluc2RhbGUtQm9sZEl0YWxpYy5vdGY=.png',
 'QWlybGluZS1Ob3JtYWwub3Rm.png',
 'QWlybGluZS5vdGY=.png',
 'QWlybW9sZSBBbnRpcXVlLnR0Zg==.png',
 'QWlyIEZsb3cgQlROIEh2IE9ibGlxdWUudHRm.png',
 'QWlyIEZsb3cgQlROIEx0IE9ibGlxdWUudHRm.png',
 'QWlyYWNvYnJhIExlZnRhbGljLnR0Zg==.png',
 'QWlyZm9pbCBTY3JpcHQgU1NpLnR0Zg==.png',
 'QWlyZWRhbGUtUmVndWxhci5vdGY=.png',
 'QWN0aW9uIE1hbiBTaGFkZWQudHRm.png',
 'QWN0aW9uIElzLCBXaWRlciBKTC50dGY=.png',
 'QWN0aXZhIFRoKDEpLnR0Zg==.png',
 'QWNhbnRodXMgU1NpIEl0YWxpYy50dGY=.png',
 'QWNoaWxsZXNCbHVyTGlnaHQtRXh0ZW5kZWQub3Rm.png',
 'QWNoZSBUaGluLnR0Zg==.png',
 'QWNyb3RlcmlvbiBKRi50dGY=.png',
 'QWR2ZW50dXJlciBMaWdodCBTRi50dGY=.png',
 'QWR2ZXJ0aXNlcnNHb3RoaWNMaWdodC1SZWd1bGFyLnR0Zg==.png',
 'QWR2ZXJ0Um91Z2gtT25lLm90Zg==.png',
 'QWRhbXMgVGhpbiBJdGFsaWMudHRm.png',
 'QWRlbG9uLURlbWlCb2xkLm90Zg==.png',
 'QWRlbG9uLURlbWlCb2xkSXRhLm90Zg==.png',
 'QWRpbmVLaXJuYmVyZyBSZWd1bGFyKDEpLnR0Zg==.png',
 'QWRsZXIudHRm.png',
 'QWRvYmVBcmFiaWMtQm9sZEl0YWxpYy5vdGY=.png',
 'QWRvYmVDb3JwSUQtTWluaW9uQmQub3Rm.png',
 'QWRvYmVDb3JwSUQtTWluaW9uU2Iub3Rm.png',
 'QWRyZW5hbGluLm90Zg==.png',
 'QWRyZW5hbGluZSBaZXJvLnR0Zg==.png',
 'QWt6aWRlbnotR3JvdGVzayAoUikgU2NodWxidWNoIDIgUmVndWxhci50dGY=.png',
 'QWt6aWRlbnpHcm90RXh0QlEtUmVndWxhci5vdGY=.png',
 'QWt6aWRlbnpHcm90ZXNrLUJsYWNrLm90Zg==.png',
 'QWt6aWRlbnpHcm90ZXNrQkUtTWQub3Rm.png',
 'QWt6aWRlbnpHcm90ZXNrQkUtWEJkQ25JdC5vdGY=.png',
 'QWt6aWRlbnpHcm90ZXNrRXhwZXJ0QlEtTGlnaHRPc0Yub3Rm.png',
 'QWVvc0xpZ2F0dXJlLm90Zg==.png',
 'QWx0ZVNjaEQub3Rm.png',
 'QWxiYXRyb3NzLnR0Zg==.png',
 'QWxiZXJ0dXMgTVQucGZi.png',
 'QWxiZXJ0YSBSZWd1bGFyLnR0Zg==.png',
 'QWxiZXJ0YW5Cb2xkTC5vdGY=.png',
 'QWxkaW5lIDcyMSBCb2xkLnBmYg==.png',
 'QWxkaW5lNzIxQlQtQm9sZENvbmRlbnNlZC5vdGY=.png',
 'QWxlbWJpY0JldGEtUmVndWxhclR3by5vdGY=.png',
 'QWxleHVzcyBIZWF2eSBIb2xsb3cgQ29uZGVuc2VkLnR0Zg==.png',
 'QWxpeDIudHRm.png',
 'QWxpZW4gTWFya3NtYW5SZWd1bGFyLnR0Zg==.png',
 'QWxsdXJlQm9sZEFsdFNwYWNlLm90Zg==.png',
 'QWxtb250ZS50dGY=.png',
 'QWxtb3N0IEhlYXZlbiBORi50dGY=.png',
 'QWxvciBDb25kZW5zZWQgQm9sZC50dGY=.png',
 'QWxvZSBJdGFsaWMudHRm.png',
 'QWxwaGEgRmxpZ2h0IFNvbGlkIFNtYWxsIENhcHMudHRm.png',
 'QWxwaGEgRmxpZ2h0LnR0Zg==.png',
 'QWxwaGFiZXRTb3VwQlQtVGlsdC5vdGY=.png',
 'QXBleFNlcmlmLUJvb2tJdGFsaWMub3Rm.png',
 'QXBvbGxvIFJlZ3VsYXIudHRm.png',
 'QXBvbGxvLVNlbWlCb2xkLm90Zg==.png',
 'QXBvbGxvOS50dGY=.png',
 'QXBvbGxvTVQtU0Mub3Rm.png',
 'QXBvbGxvTVQtU2VtaUJvbGQub3Rm.png',
 'QXBwbGUgR2FyYW1vbmQgQlQgQm9sZC50dGY=.png',
 'QXBwbGUgR2FyYW1vbmQgQm9vayBCVC50dGY=.png',
 'QXdha2VuLnR0Zg==.png',
 'QXF1aXRhaW5lIEluaXRpYWxzIElDRy50dGY=.png',
 'QXJ0aWZpY2VTU0sudHRm.png',
 'QXJ0aXN0YS50dGY=.png',
 'QXJjaGUgQmxhY2sgQ29uZGVuc2VkIFNTaSBCbGFjayBDb25kZW5zZWQudHRm.png',
 'QXJjdHVydXMub3Rm.png',
 'QXJnZW50YSBCb2xkLnR0Zg==.png',
 'QXJpc3RvY3JhdFN0ZC5vdGY=.png',
 'QXJpYWwgQ0UgQm9sZC50dGY=.png',
 'QXJpYWwgSEMudHRm.png',
 'QXJpYWxNb25vc3BhY2VkTVRTdGQtT2JsaXF1ZS5vdGY=.png',
 'QXJpYWxNVFN0ZC1FeHRyYUJvbGRJdC5vdGY=.png',
 'QXJpYWxNVFN0ZC1MaWdodEl0YWxpYy5vdGY=.png',
 'QXJpYWxNVFN0ZC5vdGY=.png',
 'QXJpYWxSb3VuZGVkTVRTdGQtQm9sZC5vdGY=.png',
 'QXJpZCBJVEMudHRm.png',
 'QXJpZElUQy5vdGY=.png',
 'QXJrb25hIFJlZ3VsYXIucGZi.png',
 'QXJteSBDb25kZW5zZWQudHRm.png',
 'QXJteSBFeHBhbmRlZC50dGY=.png',
 'QXJtYWRhLUJsYWNrQ29tcHJlc3NlZC5vdGY=.png',
 'QXJtYWRhLUxpZ2h0Q29tcHJlc3NlZC5vdGY=.png',
 'QXJtZW5zY2hyaWZ0LnR0Zg==.png',
 'QXJub3ZhSVRDIFRULnR0Zg==.png',
 'QXJyaWJhQXJyaWJhU3RkLm90Zg==.png',
 'QXJyb3dFeHRyYUJvbGQub3Rm.png',
 'QXJydXNPU0ZCVC1Cb2xkLm90Zg==.png',
 'QXJydXNPU0ZCVC1Sb21hbi5vdGY=.png',
 'QXN0aWdtYSBSZWd1bGFyLnR0Zg==.png',
 'QXN0cm9PYmxpcXVlLnR0Zg==.png',
 'QXN0cm9uIEJveSBWaWRlby50dGY=.png',
 'QXN0cmlkIFNob3J0LnR0Zg==.png',
 'QXN0dXRlIEl0YWxpYy50dGY=.png',
 'QXN0dXRlIFNTaSBCb2xkLnR0Zg==.png',
 'QXNobGV5Q3Jhd2ZvcmRNVFN0ZC5vdGY=.png',
 'QXNzZW1ibHlMaWdodFNTSy50dGY=.png',
 'QXRoZW5hZXVtU3RkLUJvbGQub3Rm.png',
 'QXRpbGxhIFdpZGUgTm9ybWFsLnR0Zg==.png',
 'QXRsYW50aXggU1NpIFNlbWkgQm9sZCBJdGFsaWMudHRm.png',
 'QXRsYXMgb2YgdGhlIE1hZ2kudHRm.png',
 'QXRtb3NwaGVyZUlUQ1N0ZC5vdGY=.png',
 'QXV0b21hdGljUmVndWxhckV4cGVydC5vdGY=.png',
 'QXV0b3RyYWNlLUZpdmUub3Rm.png',
 'QXV0bzEtQmxhY2tJdGFsaWNMRi50dGY=.png',
 'QXV0bzEtSXRhbGljTEYudHRm.png',
 'QXV0bzItSXRhbGljLnR0Zg==.png',
 'QXVndXN0LUxpZ2h0Lm90Zg==.png',
 'QXVndXN0YUNhbmNlbGxhcmVzY2FTdGQtUmVnLm90Zg==.png',
 'QXVndXN0YVN0ZC1SZWd1bGFyLm90Zg==.png',
 'QXVndXN0YVNjaG51cmtsU3RkLVJlZy5vdGY=.png',
 'QXVyaW9sLUJvbGRJdGFsaWMub3Rm.png',
 'QXVyaW9sTFRTdGQtQmxhY2tJdGFsaWMub3Rm.png',
 'QXVyaW9sTFRTdGQtSXRhbGljLm90Zg==.png',
 'QXVyYS1PdXRsaW5lT2JsaXF1ZS5vdGY=.png',
 'QXVyZWFVbHRyYS1JdGFsaWMub3Rm.png',
 'QXVyZWxpYUVGLUJvb2tJdGFsaWMub3Rm.png',
 'QXZhbG9uLm90Zg==.png',
 'QXZhbnRHYXJkRUYtRGVtaS5vdGY=.png',
 'QXZhbnRHYXJkRUYtRXh0cmFMaWdodC5vdGY=.png',
 'QXZhbnRHYXJkZS1Db25kQm9vay5vdGY=.png',
 'QXZhbnRHYXJkZUJRLU1lZGl1bU9ibGlxdWUub3Rm.png',
 'QXZhbnRHYXJkZUlUQ1RUIERlbWlPYmxpcXVlLnR0Zg==.png',
 'QXZhbnRHYXJkZUlUQ2J5QlQtQm9sZE9ibGlxdWUub3Rm.png',
 'QXZhbnRHYXJkZUlUQ2J5QlQtQm9va09ibGlxdWUub3Rm.png',
 'QXZhbnRHYXJkZUlUQ2J5QlQtRXh0cmFMaWdodE9ibC5vdGY=.png',
 'QXZhbnRHYXJkZUlUQy1Cb29rT2JsaXF1ZS5vdGY=.png',
 'QXZhbnRHYXJkZUV4dExpdElUQy1PYmxpcXVlLm90Zg==.png',
 'QXZhbnRHYXJkZUxULUJvbGQub3Rm.png',
 'QXZhbnRHYXJkZUxULUNvbmRCb29rLm90Zg==.png',
 'QXZhbnRHYXJkZUxULUNvbmREZW1pLm90Zg==.png',
 'QXZhbnRHYXJkZUxULURlbWkub3Rm.png',
 'QXZhbnRHYXJkZVhMaWdodE9ibGlxdWUub3Rm.png',
 'QXZlbmlyIDQ1IEJvb2sgT2JsaXF1ZS5wZmI=.png',
 'QXZlbmlyIDY1IE1lZGl1bS5wZmI=.png',
 'QXZlbmlyLU9ibGlxdWUub3Rm.png',
 'QXZlcnlzSGFuZCBSZWd1bGFyLnR0Zg==.png',
 'R0UgRWxlZ2FudFNjcmlwdC50dGY=.png',
 'R290aGFtIE5pZ2h0cyBOb3JtYWwudHRm.png',
 'R290aGFtUm91bmRlZC1Cb2xkSXRhbGljLm90Zg==.png',
 'R290aGFtUm91bmRlZC1MaWdodEl0YWxpYy5vdGY=.png',
 'R290aGljIDcyMCBCb2xkIEJULnR0Zg==.png',
 'R290aGljIDcyMCBJdGFsaWMgQlQudHRm.png',
 'R290aGljIDcyNSBCb2xkIEJULnR0Zg==.png',
 'R290aGljIFNTaS50dGY=.png',
 'R290aGljNzIwIEJUIEJvbGQgSXRhbGljLnR0Zg==.png',
 'R290aGljNzIwIEJUIFJvbWFuLnR0Zg==.png',
 'R290aGljNzIwIEx0IEJUIExpZ2h0LnR0Zg==.png',
 'R290aGljODIxIENuIEJULnR0Zg==.png',
 ...]
# Display first 20 images 
for file in fn[:20]:
    path = 'notMNIST_small/A/' + file
    display(Image(path))




















Now let's load the data in a more manageable format. Since, depending on your computer setup you might not be able to fit it all in memory, we'll load each class into a separate dataset, store them on disk and curate them independently. Later we'll merge them into a single dataset of manageable size.

We'll convert the entire dataset into a 3D array (image index, x, y) of floating point values, normalized to have approximately zero mean and standard deviation ~0.5 to make training easier down the road.

A few images might not be readable, we'll just skip them.

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def load_letter(folder, min_num_images):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  print(folder)
  num_images = 0
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
      image_data = (ndimage.imread(image_file).astype(float) - 
                    pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[num_images, :, :] = image_data
      num_images = num_images + 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
    
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset
        
def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(folder, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)
  
  return dataset_names

train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)
notMNIST_large/A.pickle already present - Skipping pickling.
notMNIST_large/B.pickle already present - Skipping pickling.
notMNIST_large/C.pickle already present - Skipping pickling.
notMNIST_large/D.pickle already present - Skipping pickling.
notMNIST_large/E.pickle already present - Skipping pickling.
notMNIST_large/F.pickle already present - Skipping pickling.
notMNIST_large/G.pickle already present - Skipping pickling.
notMNIST_large/H.pickle already present - Skipping pickling.
notMNIST_large/I.pickle already present - Skipping pickling.
notMNIST_large/J.pickle already present - Skipping pickling.
notMNIST_small/A.pickle already present - Skipping pickling.
notMNIST_small/B.pickle already present - Skipping pickling.
notMNIST_small/C.pickle already present - Skipping pickling.
notMNIST_small/D.pickle already present - Skipping pickling.
notMNIST_small/E.pickle already present - Skipping pickling.
notMNIST_small/F.pickle already present - Skipping pickling.
notMNIST_small/G.pickle already present - Skipping pickling.
notMNIST_small/H.pickle already present - Skipping pickling.
notMNIST_small/I.pickle already present - Skipping pickling.
notMNIST_small/J.pickle already present - Skipping pickling.
Pickling

It has two methods:
Dump: dumps an object to a file object.
Load: loads an object from a file object.
Some use cases:
Saving a program's state data to disk so that it can carry on where it left off when restarted (persistence).
Sending python data over a TCP connection in a multi-core or distributed system (marshalling).
Storing python objects in a database.
Converting an arbitrary python object to a string so that it can be used as a dictionary key (e.g. for caching & memorization).
# import pickle

# Create a list
test_values = ['test value','test value 2','test value 3']
display(test_values)

file_Name = "testfile"
# Open the file for writing
fileObject = open(file_Name,'wb') 

# This writes the object a to the
# file named 'testfile'
pickle.dump(test_values, fileObject)   

# Then we close the fileObject
fileObject.close()

# We then open the file for reading
fileObject = open(file_Name,'r')  

# And the object from the file into var b
test_values_loaded = pickle.load(fileObject) 
display(test_values_loaded)
display(test_values == test_values_loaded)
['test value', 'test value 2', 'test value 3']
['test value', 'test value 2', 'test value 3']
True
Problem 2
Let's verify that the data still looks good. Displaying a sample of the labels and images from the ndarray. Hint: you can use matplotlib.pyplot.

# index 0 should be all As, 1 = all Bs, etc.
pickle_file = train_datasets[0]  

# With would automatically close the file after the nested block of code
with open(pickle_file, 'rb') as f:
    
    # unpickle
    letter_set = pickle.load(f)  
    
    # pick a random image index
    sample_idx = np.random.randint(len(letter_set))
    
    # extract a 2D slice
    sample_image = letter_set[sample_idx, :, :]  
    plt.figure()
    
    # display it
    plt.imshow(sample_image)  

Problem 3
Another check: we expect the data to be balanced across classes. Verify that.

Merge and prune the training data as needed. Depending on your computer setup, you might not be able to fit it all in memory, and you can tune train_size as needed. The labels will be stored into a separate array of integers 0 through 9.

Also create a validation dataset for hyperparameter tuning.

def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
  num_classes = len(pickle_files)
  valid_dataset, valid_labels = make_arrays(valid_size, image_size)
  train_dataset, train_labels = make_arrays(train_size, image_size)
  vsize_per_class = valid_size // num_classes
  tsize_per_class = train_size // num_classes
    
  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):       
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        # let's shuffle the letters to have random validation and training set
        np.random.shuffle(letter_set)
        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :]
          valid_dataset[start_v:end_v, :, :] = valid_letter
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class
                    
        train_letter = letter_set[vsize_per_class:end_l, :, :]
        train_dataset[start_t:end_t, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise
    
  return valid_dataset, valid_labels, train_dataset, train_labels
            
            
train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
  train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)
Training: (200000, 28, 28) (200000,)
Validation: (10000, 28, 28) (10000,)
Testing: (10000, 28, 28) (10000,)
Next, we'll randomize the data. It's important to have the labels well shuffled for the training and test distributions to match.

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)
Problem 4
Convince yourself that the data is still good after shuffling!

Finally, let's save the data for later reuse:

pickle_file = 'notMNIST.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise
# Getting statistics of a file using os.stat(file_name)
statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)
Compressed pickle size: 690800441
Problem 5
By construction, this dataset might contain a lot of overlapping samples, including training data that's also contained in the validation and test set! Overlap between training and test can skew the results if you expect to use your model in an environment where there is never an overlap, but are actually ok if you expect to see training samples recur when you use it. Measure how much overlap there is between training, validation and test samples.

Optional questions:

What about near duplicates between datasets? (images that are almost identical)
Create a sanitized validation and test set, and compare your accuracy on those in subsequent assignments.
import time

def check_overlaps(images1, images2):
    images1.flags.writeable=False
    images2.flags.writeable=False
    start = time.clock()
    hash1 = set([hash(image1.data) for image1 in images1])
    hash2 = set([hash(image2.data) for image2 in images2])
    all_overlaps = set.intersection(hash1, hash2)
    return all_overlaps, time.clock()-start

r, execTime = check_overlaps(train_dataset, test_dataset)    
print('Number of overlaps between training and test sets: {}. Execution time: {}.'.format(len(r), execTime))

r, execTime = check_overlaps(train_dataset, valid_dataset)   
print('Number of overlaps between training and validation sets: {}. Execution time: {}.'.format(len(r), execTime))

r, execTime = check_overlaps(valid_dataset, test_dataset) 
print('Number of overlaps between validation and test sets: {}. Execution time: {}.'.format(len(r), execTime))
Number of overlaps between training and test sets: 1153. Execution time: 0.951144.
Number of overlaps between training and validation sets: 952. Execution time: 1.014579.
Number of overlaps between validation and test sets: 55. Execution time: 0.088879.
Problem 6
Let's get an idea of what an off-the-shelf classifier can give you on this data. It's always good to check that there is something to learn, and that it's a problem that is not so trivial that a canned solution solves it.

Train a simple model on this data using 50, 100, 1000 and 5000 training samples. Hint: you can use the LogisticRegression model from sklearn.linear_model.

Optional question: train an off-the-shelf model on all the data!

# Here you have 200000 samples
# 28 x 28 features
# We have to reshape them because scikit-learn expects (n_samples, n_features)
train_dataset.shape
(200000, 28, 28)
test_dataset.shape
(10000, 28, 28)
# Prepare training data
samples, width, height = train_dataset.shape
X_train = np.reshape(train_dataset,(samples,width*height))
y_train = train_labels

# Prepare testing data
samples, width, height = test_dataset.shape
X_test = np.reshape(test_dataset,(samples,width*height))
y_test = test_labels
# Import
from sklearn.linear_model import LogisticRegression

# Instantiate
lg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42, verbose=1, max_iter=1000, n_jobs=-1)

# Fit
lg.fit(X_train, y_train)

# Predict
y_pred = lg.predict(X_test)

# Score
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred)
[Parallel(n_jobs=-1)]: Done   1 out of   1 | elapsed:  5.9min finished
0.90010000000000001
