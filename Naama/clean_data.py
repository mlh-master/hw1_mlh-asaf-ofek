# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:14:23 2019

@author: smorandv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rm_ext_and_nan(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A dictionary of clean CTG called c_ctg
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    CTG_features_temp = CTG_features.copy()
    CTG_features_temp.drop(columns=[extra_feature], inplace=True)  # drop extra_feature from CTG_features
    c_ctg = {}  # initialize a dictionary
    for column in CTG_features_temp.columns:
        c_ctg[column] = pd.to_numeric(CTG_features_temp[column], errors='coerce').dropna()  # turn non numeric values to nan and drop them
    # -------------------------------------------------------------------------
    return c_ctg


def nan2num_samp(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A pandas dataframe of the dictionary c_cdf containing the "clean" features
    """
    c_cdf = {}
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    CTG_features.drop(columns=[extra_feature], inplace=True)
    for column in CTG_features.columns:
        col = pd.to_numeric(CTG_features[column], errors='coerce')  # turn non numeric values to nan
        i = col.isnull()  # create a boolean vector with true values where col has nan values
        idx = pd.Series([])  # initialize an index vector to save nan locations
        t = 1  # initialize a counter for the nan locations vector (idx)
        for j in range(1, len(i)+1):
            if i[j] == 1:
                idx = idx.set_value(t, j)
                t += 1
        temp = np.random.choice(col, size=len(idx))  # random sampling of len(idx) values from col
        col[idx] = temp
        c_cdf[column] = col
    # -------------------------------------------------------------------------
    return pd.DataFrame(c_cdf)


def sum_stat(c_feat):
    """

    :param c_feat: Output of nan2num_cdf
    :return: Summary statistics as a dicionary of dictionaries (called d_summary) as explained in the notebook
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    d_summary = {}  # initialize a dictionary
    for column in c_feat.columns:
        curr = c_feat[column]
        d_summary[column] = {"min": curr.min(), "Q1": curr.quantile(0.25), "median": curr.median(), "Q3": curr.quantile(0.75), "max": curr.max()}
        # -------------------------------------------------------------------------
    return d_summary


def rm_outlier(c_feat, d_summary):
    """

    :param c_feat: Output of nan2num_cdf
    :param d_summary: Output of sum_stat
    :return: Dataframe of the dictionary c_no_outlier containing the feature with the outliers removed
    """
    c_no_outlier = {}
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    for column in c_feat.columns:
        col = c_feat[column]
        IQR = d_summary[column]['Q3']-d_summary[column]['Q1']
        max = d_summary[column]['Q3']+1.5*IQR
        min = d_summary[column]['Q1']-1.5*IQR
        col = col.replace(col[col > max], np.nan)
        col = col.replace(col[col < min], np.nan)
        c_no_outlier[column] = col
    # -------------------------------------------------------------------------
    return pd.DataFrame(c_no_outlier)

def phys_prior(c_cdf, feature, thresh):
    """

    :param c_cdf: Output of nan2num_cdf
    :param feature: A string of your selected feature
    :param thresh: A numeric value of threshold
    :return: An array of the "filtered" feature called filt_feature
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:-----------------------------
    curr = c_cdf[feature]
    filt_feature = np.array(curr.replace(curr[curr > thresh], np.nan))
    # -------------------------------------------------------------------------
    return filt_feature


def norm_standard(CTG_features, selected_feat=('LB', 'ASTV'), mode='none', flag=False):
    """

    :param CTG_features: Pandas series of CTG features
    :param selected_feat: A two elements tuple of strings of the features for comparison
    :param mode: A string determining the mode according to the notebook
    :param flag: A boolean determining whether or not plot a histogram
    :return: Dataframe of the normalized/standardazied features called nsd_res
    """
    x, y = selected_feat
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    nsd_res = CTG_features.copy()

    if mode == 'standard':
        for column in CTG_features.columns:
            nsd_res[column] = (nsd_res[column]-nsd_res[column].mean())/nsd_res[column].std()
    if mode == 'MinMax':
        for column in CTG_features.columns:
            nsd_res[column] = (nsd_res[column]-nsd_res[column].min())/(nsd_res[column].max()-nsd_res[column].min())
    if mode == 'mean':
        for column in CTG_features.columns:
            nsd_res[column] = (nsd_res[column]-nsd_res[column].mean())/(nsd_res[column].max()-nsd_res[column].min())

    if flag == True:
        plt.hist(nsd_res[x], bins=100, label=x)
        plt.hist(nsd_res[y], bins=100, label=y)
        plt.ylabel('Frequency')
        plt.xlabel('Values')
        plt.title(mode)
        plt.legend(loc='upper left')
        plt.show()

        # -------------------------------------------------------------------------
    return pd.DataFrame(nsd_res)
