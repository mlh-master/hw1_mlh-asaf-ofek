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
    CTG_df = pd.DataFrame(CTG_features).drop(extra_feature,1)

    c_ctg = {}
    for column in CTG_df.columns:
        c_ctg[column] = pd.to_numeric(CTG_df[column], errors = 'coerce').dropna()

    # --------------------------------------------------------------------------
    return c_ctg


def nan2num_samp(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A pandas dataframe of the dictionary c_cdf containing the "clean" features
    """

    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    CTG_df = pd.DataFrame(CTG_features).drop(extra_feature, 1)
    c_cdf = {}
    for column in CTG_df.columns:
        replacing_value = np.random.choice(CTG_df[column])
        c_cdf[column] = pd.to_numeric(CTG_df[column], errors='coerce').fillna(replacing_value)
    # -------------------------------------------------------------------------
    return pd.DataFrame(c_cdf)


def sum_stat(c_feat):
    """

    :param c_feat: Output of nan2num_cdf
    :return: Summary statistics as a dicionary of dictionaries (called d_summary) as explained in the notebook
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    d_summary = {}
    keys_list = ['Q1', 'Q3', 'min', 'max', 'mean']
    for column in c_feat.columns:
        d_summary[column] = {key: None for key in keys_list}
    for column in c_feat.columns:
        Q1 = c_feat[column].quantile(0.25)
        Q3 = c_feat[column].quantile(0.75)
        max = c_feat[column].max()
        min = c_feat[column].min()
        mean = c_feat[column].mean()
        d_summary[column]['Q1'] =Q1
        d_summary[column]['Q3'] = Q3
        d_summary[column]['min'] = min
        d_summary[column]['max'] = max
        d_summary[column]['mean'] = mean
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
        Q1 = d_summary[column]['Q1']
        Q3 = d_summary[column]['Q3']
        IQR = Q3 - Q1
        outlier_max = Q3 + 1.5 * IQR
        outlier_min = Q1 - 1.5 * IQR
        c_no_outlier[column] = c_feat[column].copy()
        c_no_outlier[column][(c_no_outlier[column] < outlier_min) | (c_no_outlier[column] > outlier_max)] = None
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
    filt_feature = c_cdf[feature][c_cdf[feature] < thresh]
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
            nsd_res[column] = (CTG_features[column]-min)/(max-min)
    elif mode == 'standard':
        for column in CTG_features.columns:
            mean = CTG_features[column].mean()
            std = CTG_features[column].std()
            nsd_res[column] = (CTG_features[column]-mean)/std
    elif mode =='mean':
        for column in CTG_features.columns:
            mean = CTG_features[column].mean()
            max = CTG_features[column].max()
            min = CTG_features[column].min()
            nsd_res[column] = (CTG_features[column]-mean)/(max-min)
    elif mode == 'none':
        print('none scaling')
    else:
        print('Error: Can''t find scaling mode')
        return
    if(flag == True):
        bins = 100
        plt.hist(CTG_features[selected_feat[0]], bins)
        plt.hist(CTG_features[selected_feat[1]], bins)
        plt.show()
        if((mode != 'none')):
            plt.hist(nsd_res[selected_feat[0]], bins)
            plt.hist(nsd_res[selected_feat[1]], bins)
            plt.title(mode + 'Scaling')
            plt.show()
    # -------------------------------------------------------------------------
    return pd.DataFrame(nsd_res)

#def max_min_func (var):
