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
    c_ctg={}
    c_ctg = dict()
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    nan = CTG_features.copy()
    del nan[extra_feature]
    for col in nan:
        c_ctg[col] = pd.to_numeric(nan[col], errors='coerce')
        c_ctg[col] = nan[col].dropna()
    # --------------------------------------------------------------------------
    return c_ctg


def nan2num_samp(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A pandas dataframe of the dictionary c_cdf containing the "clean" features
    """
    c_cdf = {}
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    c_cdf = CTG_features.copy()
    del c_cdf[extra_feature]
    for col in c_cdf.columns:
        c_cdf[col] = pd.to_numeric(c_cdf[col], errors='coerce')
        hist = c_cdf[col].value_counts(normalize=True)
        len = c_cdf[col].isna()
        idx = len[len].index
        c_cdf.loc[idx, col] = np.random.choice(hist.index, p=hist.values, size=c_cdf[col].isna().sum())
        
    # -------------------------------------------------------------------------
    return pd.DataFrame(c_cdf)


def sum_stat(c_feat):
    """

    :param c_feat: Output of nan2num_cdf
    :return: Summary statistics as a dicionary of dictionaries (called d_summary) as explained in the notebook
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    d_summary={}
    for col in c_feat.columns:
        median = c_feat.median()[col]
        Q1 = c_feat.quantile(0.25)[col]
        Q3 = c_feat.quantile(0.75)[col]
        mini = c_feat.min()[col]
        maxi = c_feat.max()[col]
        d_summary[col] = {'min': mini, 'Q1': Q1, 'median': median, 'Q3': Q3, 'max': maxi}

    # -------------------------------------------------------------------------
    return d_summary


def rm_outlier(c_feat, d_summary):
    """

    :param c_feat: Output of nan2num_cdf
    :param d_summary: Output of sum_stat
    :return: Dataframe of the dictionary c_no_outlier containing the feature with the outliers removed
    """
    c_no_outlier = {}
    c_temp = c_feat.copy()
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    for col in c_temp.columns:
        Q1 = d_summary[col]['Q1']
        Q3 = d_summary[col]['Q3']
        IQR = (Q3 - Q1)*1.5
        outlier_max = Q3 + IQR
        outlier_min = Q1 - IQR
        c_no_outlier[col] = c_temp[col]
        c_no_outlier[col][(c_no_outlier[col] < outlier_min) | (c_no_outlier[col] > outlier_max)] = None

    # -------------------------------------------------------------------------
    return pd.DataFrame(c_no_outlier)


def phys_prior(c_cdf, feature, thresh):
    """

    :param c_cdf: Output of nan2num_cdf
    :param feature: A string of your selected feature
    :param thresh: A numeric value of threshold
    :return: An array of the "filtered" feature called filt_feature
    """
    c_temp = c_cdf.copy()
    filt_feature = []
    # ------------------ IMPLEMENT YOUR CODE HERE:-----------------------------
    c_temp[feature][c_temp[feature]>thresh]= None
    len = c_temp[feature].isna()
    idx = len[len].index
    filt_feature.append(c_temp[feature][idx])
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
    nsd_res = {}

    if mode == 'MinMax':
        for column in CTG_features.columns:
            max = CTG_features[column].max()
            min = CTG_features[column].min()
            nsd_res[column] = (CTG_features[column] - min) / (max - min)
    elif mode == 'standard':
        for column in CTG_features.columns:
            mean = CTG_features[column].mean()
            std = CTG_features[column].std()
            nsd_res[column] = (CTG_features[column] - mean) / std
    elif mode == 'mean':
        for column in CTG_features.columns:
            mean = CTG_features[column].mean()
            max = CTG_features[column].max()
            min = CTG_features[column].min()
            nsd_res[column] = (CTG_features[column] - mean) / (max - min)
    elif mode == 'none':
        print('none scaling')
    else:
        print('Error: Can''t find scaling mode')
        return
    if (flag == True):
        bins = 100
        plt.hist(CTG_features[selected_feat[0]], bins)
        plt.hist(CTG_features[selected_feat[1]], bins)
        plt.show()
        if ((mode != 'none')):
            plt.hist(nsd_res[selected_feat[0]], bins)
            plt.hist(nsd_res[selected_feat[1]], bins)
            plt.title(mode + 'Scaling')
            plt.show()
    # -------------------------------------------------------------------------
    return pd.DataFrame(nsd_res)
