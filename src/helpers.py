import pandas as pd
import numpy as np


def load_and_check_nans(filepath):
    ''' Loads data into a pandas data frame and locates columns with only Nans.

    Input: filepath and name (.csv files)
    Output: 
        a dictionary of:
            - a dataframe
            - columns in the dataframe, and
            - a list of columns/features where the only data in the column are Nans
    '''
    df = pd.read_csv(filepath)   
    cols = df.columns.to_list()

    only_nans = []
    for col in cols:
        num_nans = df[col].isna().sum()
        if num_nans == df.shape[0]:
            only_nans.append(col)

    return df, cols, only_nans



def clean_data(dataframe, all_nans):
    ''' Takes an existing dataframe and cleans it in preparation for analysis. Used in conjunction
    with load_and_check_nans.

    Input: 
        dataframe
        all_nans: list of columns that only contain nans for removal
    Output: a pandas data frame that needs to be assigned to a variable
    '''

    # In both sets of STATE SADCQ files, the following columns only consisted of Nans:
    # ['qtaughtcondom', 'qcoffeetea', 'qspeakenglish']

    # Remove the empty columns:
    dataframe.drop(columns=all_nans, inplace=True)
    yr_mask = dataframe['year'] > 2009.0

    return dataframe
    


def many_nans(dataframe):
    ''' Takes a dataframe and determines the number of nans in each column of
    the dataframe.

    Input: 
        dataframe
        df_columns: list of columns in dataframe
    Output: a dictinary where percentage of nans are keys and list of columns
        with this percentage of nans are values
    '''

    df_columns = dataframe.columns.to_list()

    # Determine extent of nans:
    ninety, eighty, seventy, sixty, fifty = [], [], [], [], []
    forty, thirty, twenty, ten = [], [], [], []

    for col in df_columns:
        if dataframe[col].isna().sum() / dataframe.shape[0] >= .9:
            ninety.append(col)
        if dataframe[col].isna().sum() / dataframe.shape[0] >= .8:
            eighty.append(col)
        if dataframe[col].isna().sum() / dataframe.shape[0] >= .7:
            seventy.append(col)
        if dataframe[col].isna().sum() / dataframe.shape[0] >= .6:
            sixty.append(col)
        if dataframe[col].isna().sum() / dataframe.shape[0] >= .5:
            fifty.append(col)
        if dataframe[col].isna().sum() / dataframe.shape[0] >= .4:
            forty.append(col)
        if dataframe[col].isna().sum() / dataframe.shape[0] >= .3:
            thirty.append(col)
        if dataframe[col].isna().sum() / dataframe.shape[0] >= .2:
            twenty.append(col)
        if dataframe[col].isna().sum() / dataframe.shape[0] >= .1:
            ten.append(col)

    return {'len_90': len(ninety), '90 percent': ninety, 
            'len_80': len(eighty), '80 percent': eighty, 
            'len_70': len(seventy), '70 percent': seventy, 
            'len_60': len(sixty), '60 percent': sixty,
            'len_50': len(fifty), '50 percent': fifty, 
            'len_40': len(forty), '40 percent': forty, 
            'len_30': len(thirty), '30 percent': thirty, 
            'len_20': len(twenty), '20 percent': twenty, 
            'len_10': len(ten), '10 percent': ten}



