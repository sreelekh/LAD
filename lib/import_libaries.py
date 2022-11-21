# # Import Libraries
# import matplotlib
# matplotlib.use('agg')
# from google.colab import drive
# drive.mount('/content/gdrive')
# sys.path.append('/content/gdrive/My Drive/Colab Notebooks')
# %load_ext line_profiler

import numpy as np
import pickle
import pandas as pd
import sys
import time
import random               # add some random sleep time
import scipy
import glob
import scipy.stats as stats
import time
import os
import math
import copy
import statsmodels.api as sm
import itertools
import re
import itertools
import shutil
# import h5py
import matplotlib
from scipy.io import arff

from os import listdir

# from spot import SPOT
from os.path import isfile, join
# from statsutils import *
# from boltons.statsutils import *
from datetime import datetime
from itertools import repeat, cycle, islice
# from ipywidgets import interact
from dateutil.parser import parse

from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils import check_random_state, shuffle
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.svm import OneClassSVM as ocsvm
from sklearn import cluster, datasets, metrics, mixture, svm
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor, kneighbors_graph
from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import RandomOverSampler


from scipy import linalg
from scipy.special import gamma, factorial, digamma, betaln, gammaln
from scipy.stats import beta, multivariate_normal, wishart, invwishart, t, mode
from scipy.stats import genextreme as gev
import scipy.spatial as sp
import scipy.io
from scipy.io import arff

from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pylab as pylab
import matplotlib as mpl

sns.set(color_codes=True)
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=2)
# %config InlineBackend.print_figure_kwargs = {'bbox_inches': None}
# %matplotlib inline
