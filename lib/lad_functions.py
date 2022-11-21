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
import glob
from tqdm import tqdm

# Global Figure Parameters
global label_size
global fig_len
global fig_wid
global m_size
global title_size

fig_len = 16
fig_wid = 16
m_size = 50
title_size = 50
label_size = 30

plot_params = {'legend.fontsize': 60,
               'figure.figsize': (fig_len, fig_wid),
               'axes.labelsize': label_size,
               'axes.titlesize': title_size,
               'xtick.labelsize': label_size,
               'ytick.labelsize': label_size}
pylab.rcParams.update(plot_params)


def cleaned_covid_df(data_path):
    data_source_covid = data_path
    all_filenames = [i for i in glob.glob(data_source_covid+'/'+'*.{}'.format('csv'))]
    data_deaths = pd.concat([pd.read_csv(f) for f in all_filenames if '_US' in f and 'death' in f])
    data_confirmed = pd.concat([pd.read_csv(f) for f in all_filenames if '_US' in f and 'confirmed' in f])


    data_deaths0=data_deaths.groupby(['FIPS', 'Admin2', 'Province_State','Country_Region']).sum().reset_index()
    data_deaths1=data_deaths0[data_deaths0['Population']>50000]
    data_deaths2=data_deaths1.drop(['UID',  'code3', 'FIPS', 'Admin2', 'Province_State','Country_Region', 'Lat', 'Long_'],axis=1)
    data_deaths3=data_deaths2.div(data_deaths2['Population'],axis=0)
    data_deaths3=data_deaths3.drop(['Population'],axis=1)


    data_confirmed_0=data_confirmed.join(data_deaths[['FIPS','Province_State','Admin2','UID','Population']].set_index(['FIPS','Province_State','Admin2','UID']), on=['FIPS','Province_State','Admin2','UID'])
    data_confirmed0=data_confirmed_0.groupby(['FIPS', 'Admin2', 'Province_State','Country_Region']).sum().reset_index()
    data_confirmed1=data_confirmed0[data_confirmed0['Population']>50000]
    data_confirmed2=data_confirmed1.drop(['UID',  'code3', 'FIPS', 'Admin2', 'Province_State','Country_Region', 'Lat', 'Long_'],axis=1)
    data_confirmed3=data_confirmed2.div(data_confirmed2['Population'],axis=0)
    data_confirmed3=data_confirmed3.drop(['Population'],axis=1)


    data_deaths_full0=data_deaths.groupby(['FIPS', 'Admin2', 'Province_State','Country_Region']).sum().reset_index()
    data_deaths_full1=data_deaths_full0[data_deaths_full0['Population']>50000]
    data_deaths_full2=data_deaths_full1.drop(['UID',  'code3', 'FIPS', 'Admin2', 'Province_State','Country_Region', 'Lat', 'Long_'],axis=1)
    data_deaths_full3=data_deaths_full2#.div(data_deaths_full2['Population'],axis=0)
    data_deaths_full3=data_deaths_full3.drop(['Population'],axis=1)

    data_confirmed_full_0=data_confirmed.join(data_deaths[['FIPS','Province_State','Admin2','UID','Population']].set_index(['FIPS','Province_State','Admin2','UID']), on=['FIPS','Province_State','Admin2','UID'])
    data_confirmed_full0=data_confirmed_full_0.groupby(['FIPS', 'Admin2', 'Province_State','Country_Region']).sum().reset_index()
    data_confirmed_full1=data_confirmed_full0[data_confirmed_full0['Population']>50000]
    data_confirmed_full2=data_confirmed_full1.drop(['UID',  'code3', 'FIPS', 'Admin2', 'Province_State','Country_Region', 'Lat', 'Long_'],axis=1)
    data_confirmed_full3=data_confirmed_full2#.div(data_confirmed_full2['Population'],axis=0)
    data_confirmed_full3=data_confirmed_full3.drop(['Population'],axis=1)


    data_deaths5=[]
    N_counties,T =data_deaths3.shape
    for i in range(N_counties):
        dd=np.array(data_deaths3.iloc[i])
        c_start=np.where(dd>0)[0]
        if len(c_start)>0:
            dd=np.append(dd[c_start[0]:], np.ones(c_start[0])*np.nan)
        else:
            dd=np.ones(T)*np.nan
        data_deaths5.append(dd)
    data_deaths5=pd.DataFrame(np.array(data_deaths5))
    data_deaths5=data_deaths5.set_index(data_deaths3.index)


    data_confirmed5=[]
    N_counties,T =data_confirmed3.shape
    for i in range(N_counties):
        dd=np.array(data_confirmed3.iloc[i])
        c_start=np.where(dd>0)[0]
        if len(c_start)>0:
            dd=np.append(dd[c_start[0]:], np.ones(c_start[0])*np.nan)
        else:
            dd=np.ones(T)*np.nan
        data_confirmed5.append(dd)
    data_confirmed5=pd.DataFrame(np.array(data_confirmed5))
    data_confirmed5=data_confirmed5.set_index(data_confirmed3.index)

    return data_deaths5, data_confirmed5, pd.DatetimeIndex(data_deaths3.columns)

def load_data(file_path):
    filename, extension = os.path.splitext(file_path)
    name=(os.path.basename(file_path))
    if (extension=='.mat'):
        try:
            mat = scipy.io.loadmat(file_path)
            df = pd.DataFrame(np.hstack((mat['X'], mat['y'])))
            X,y=mat['X'], mat['y']
            return df,X,y

        except:
            try:
                arrays = {}
                f = h5py.File(file_path)
                for k, v in f.items():
                    arrays[k] = np.array(v)
                X,y=arrays['X'].T,arrays['y'].T
                df = pd.DataFrame(np.hstack((arrays['X'].T, arrays['y'].T)))
                return df,X,y
            except:
                print("1 Failed to load", name)

    elif extension=='.csv':
        try:
            df = pd.read_csv(file_path,low_memory=False,delimiter=',',header=None)
            df1=df[df.columns[(df.dtypes=='float')+(df.dtypes=='int')]]
            if df1.shape[1]==0:
                df = pd.read_csv(file_path,low_memory=False,delimiter=',')
                df1=df[df.columns[(df.dtypes=='float')+(df.dtypes=='int')]]                
            X=np.array(df1).astype(float)
            y=df.drop(df.columns[(df.dtypes=='float')+(df.dtypes=='int')],axis=1)
            return df,X,y
        except:
            try:
                df = pd.read_csv(file_path,low_memory=False,delimiter=',')
                df1=df[df.columns[(df.dtypes=='float')+(df.dtypes=='int')]]
                X=np.array(df1).astype(float)
                y=df.drop(df.columns[(df.dtypes=='float')+(df.dtypes=='int')],axis=1)
                return df,X,y
            except:
                print("2 Failed to load", name)

    elif extension=='.pickle':
        try:
            data=pickle.load(open(file_path,'rb'))
            X=data['X']
            y=data['y']
            return data,X,y
        except:
            try:
                d = pickle.load( open(file_path, "rb" ) )
                data=d['rawdata']
                labels=d['labels']
                X=np.array(data).astype(float)
                y=labels
                df=np.hstack((X,y))
                return df,X,y
            except:
                print("3 Failed to load", name)
                
    elif extension=='.arff':
        data = arff.loadarff(file_path)
        df = pd.DataFrame(data[0])
        df1=df[df.columns[(df.dtypes=='float')+(df.dtypes=='int')]]
        X=np.array(df1).astype(float)
        y=df.drop(df.columns[(df.dtypes=='float')+(df.dtypes=='int')],axis=1)
        return df,X,y
        
    else:
        print("Failed to load the extension",extension)
        pass

def lad_z_initialization(X, K_f, cluster_train = 0.25):
    N, F, D = X.shape
    if K_f > 1:
        X_train, _ = train_test_split(np.arange(0, X.shape[0]),
                                      train_size=cluster_train)
        kmeans = MiniBatchKMeans(n_clusters=K_f)

        if F > 20:
            pca = PCA(n_components=20).fit_transform(
                X.reshape((X.shape[0], -1)))
            kmeans = kmeans.partial_fit(pca[X_train, :])
            return kmeans.predict(pca)+1
        else:
            kmeans = kmeans.partial_fit(X[X_train, :])
            return kmeans.predict(X)+1
    else:
        return np.ones(N)

def cleaned_windowed_tsdb(X_full, history_time, start_time_step, d):
    if history_time > 0 and history_time < start_time_step:
        X = np.copy(X_full[:, :, d-history_time:d])
    else:
        X = np.copy(X_full[:, :, :d])

    counties_missing = np.where(np.isnan(X).all(axis=(1,2)))[0]
    counties_present = np.where(np.isnan(X).all(axis=(1,2))*1 == 0)[0]

    X = X[counties_present, :, :]
    X[np.isnan(X)] = 0
    X[X == np.inf] = 0

    return X.reshape(X_full.shape[0], -1), counties_missing, counties_present

def lad_entropy(X, K, z_1, ana_score, th):
    N_s, D_s = X.shape
    clusters, sizes = np.unique(np.abs(z_1), return_counts=True)
    K = len(clusters)
        
    entropy_2 = np.empty((N_s, K, D_s))

    # K = len(clusters)
    thetas = []
    for k in clusters:
        ind_k_0 = z_1 == k
        ind_k = np.where(ind_k_0[ana_score < th])[0]
        c = len(ind_k)
        if c < 3:
            ind_k = np.abs(z_1) == k
            thetas.append(tuple((np.mean(X[ind_k], axis=0),
                                 np.array([np.cov(X[:, d]) for d in range(D_s)]))))
        else:
            thetas.append(tuple((np.mean(X[ind_k], axis=0),
                                 np.array([np.cov(X[ind_k, d]) for d in range(D_s)]))))

    cc = range(len(thetas))
    means_ = np.array([thetas[k][0].T for k in cc])
    covariances = np.array([thetas[k][1] for k in cc])

    ss = [1/(cov) for cov in covariances]
    for k, (mu, prec_chol) in enumerate(zip(means_, ss)):
        prec_chol[np.isnan(prec_chol)] = 0
        prec_chol[np.isinf(prec_chol)] = 0
        y = (np.square(X-mu)*prec_chol)/2
        entropy_2[:, k, :] = (y)/sizes[k]

    return entropy_2, sizes

def updated_labels(entropy_DGProjection, z_1, sizes, D_s):
    K_f = len(np.unique(z_1))
    N_s, K_f = entropy_DGProjection.shape
    if K_f > 1:
        cluster_log_probs = -(entropy_DGProjection*sizes)
        z_1 = np.argmax(cluster_log_probs, axis=1).flatten()+1
    else:
        cluster_log_probs = np.zeros((N_s, K_f))

    ana_score = -np.max(-((entropy_DGProjection)), axis=1)
    ana_score_0 = np.copy(ana_score)
    ana_score = (ana_score-np.min(ana_score)) / \
        (np.max(ana_score)-np.min(ana_score))

    th = min(0.95, np.mean(ana_score)+2*np.cov(ana_score))

    z_1[ana_score >= th] *= (-1)
    if np.sum(z_1 < 0) > min(0.25*N_s, D_s):
        z_1[np.argsort(-ana_score)[:int(0.25*N_s)]] = -1 * \
            np.abs(z_1[np.argsort(-ana_score)[:int(0.25*N_s)]])
        z_1[np.argsort(-ana_score)[int(0.25*N_s):]
            ] = np.abs(z_1[np.argsort(-ana_score)[int(0.25*N_s):]])

    return z_1, ana_score_0, cluster_log_probs

def cluster_update(entropy, sizes):
    N, K, D = entropy.shape
    entropy_DGProjection = np.max(entropy, axis=2)
    cluster_log_probs = (entropy_DGProjection*sizes)
    z = np.argmin(cluster_log_probs, axis=1)+1

    flag = 0
    for i, row in enumerate(cluster_log_probs):
        if np.min(row) > 0.9:
            z[i] = -K-1
            flag+=1
    if flag>0:
        K += 1
        # else:
        #     z[i] = np.argmin(row)+1

    ana_score = -np.max(-((entropy_DGProjection)), axis=1)
    ana_score_0 = np.copy(ana_score)
    ana_score = (ana_score-np.min(ana_score)) / \
        (np.max(ana_score)-np.min(ana_score))
    th = min(0.95, np.mean(ana_score)+2*np.cov(ana_score))

    z[ana_score >= th] *= (-1)
    for k in range(K):
        inds = np.abs(z) == k
        ana_score_k = ana_score[inds]
        if np.sum(z[inds] < 0) > min(0.25*len(inds), D):
            z[inds][np.argsort(-ana_score_k)[:int(0.25*N)]] = -1 * \
                np.abs(z[inds][np.argsort(-ana_score_k)[:int(0.25*N)]])
            z[inds][np.argsort(-ana_score_k)[int(0.25*N):]
                    ] = np.abs(z[inds][np.argsort(-ana_score_k)[int(0.25*N):]])
        
        if np.sum(inds) < min(0.1*N, D):
            z[inds] = -1*z[inds]

    if np.sum(z < 0) > min(0.25*N, D):
        z[np.argsort(-ana_score)[:int(0.25*N)]] = -1 * \
            np.abs(z[np.argsort(-ana_score)[:int(0.25*N)]])
        # z[np.argsort(-ana_score)[int(0.25*N):]
        #   ] = np.abs(z[np.argsort(-ana_score)[int(0.25*N):]])

    return z, K, ana_score_0, cluster_log_probs

def ldp_ts_non_uniform(X_full, K, th, start_time_step, history_time, numiter):
    N, F, D = X_full.shape
    z = {}
    thresholds = np.zeros(D)
    ana_score = np.zeros(N)
    ana_prob_array = np.zeros((N, D))

    z[start_time_step-1] = lad_z_initialization(X_full, K)

    iter = 0
    clusters, sizes = np.unique(z[start_time_step-1], return_counts=True)

    for d in (pbar := tqdm(np.arange(start_time_step, D))):
        pbar.set_description(f'Active clusters = {K}, Time-step = {d}, \
        Clusters = {clusters}, Cluster Sizes ={sizes}')

        z_1 = z[d-1]

        X, ts_missing, ts_present = cleaned_windowed_tsdb(
            X_full, history_time, start_time_step, d)

        z_1 = np.copy(z[d-1][ts_present])
        N_s, D_s = X.shape
        if N_s < 1:
            break

        ana_score = np.zeros(N_s)

        for iter in range(numiter):
            # if d == start_time_step and iter == 0:
            #     cluster_log_probs = np.zeros((N_s, K))

            entropy_2, sizes = lad_entropy(
                X, K, z_1, ana_score, th)

            # entropy_DGProjection = np.max(entropy_2, axis=2)

            # z_1, ana_score_0, cluster_log_probs = updated_labels(
                # entropy_DGProjection, z_1, sizes, D_s)
            
            z_1, K, ana_score_0, cluster_log_probs = cluster_update(
                entropy_2, sizes)
                

        ana_prob_array[ts_present, d] = np.copy(ana_score_0)
        ana_prob_array[ts_missing, d] = np.nan
        z[d] = np.zeros(N)
        z[d][ts_present] = z_1
        thresholds[d] = th

        clusters, sizes = np.unique(z[d], return_counts=True)

    ana_prob_array[np.isnan(ana_prob_array)] = 0

    output = {}
    output['z'] = z
    output['ana_prob_array'] = ana_prob_array
    output['prob'] = ana_score
    output['entropy'] = entropy_2
    output['thresholds'] = thresholds
    output['cluster_log_probs'] = cluster_log_probs
    return output
