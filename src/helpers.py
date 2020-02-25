import pandas as pd
import numpy as np


def load_and_lst_cols(filepath):
    ''' 
    Loads data into a pandas data frame and lists all columnds.

    Input: filepath and name (.csv files)
    Output: 
        a dictionary of:
            - a dataframe
            - a list of all columns/features in the dataframe
    '''

    df = pd.read_csv(filepath)   
    cols = df.columns.to_list()

    return {'dataframe': df, 'columns': cols}



def filter_and_remove_nans(dataframe, year):
    ''' 
    Takes a dataframe and filters rows according to year column
    and removes columns that only contain nans as well as columns
    provided in cols_lst
    
    Input: 
    dataframe:
    year: float (not date time type)
    remove_cols: list of columns for removal from dataframe
    Output: a dict of the new dataframe and the columns that were removed
    '''

    yr_mask = dataframe['year'] > year
    new_df = dataframe[yr_mask]
    cols = dataframe.columns.to_list()
    remove_cols = [col for col in cols if dataframe[col].isna().sum() >= dataframe.shape[0]]
    
    new_df.drop(columns=remove_cols, inplace=True)
    return {'new_dataframe': new_df, 'columns_removed': remove_cols}



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
    forty, thirty, twenty, ten, five = [], [], [], [], []

    for col in df_columns:
        if dataframe[col].isna().sum() / dataframe.shape[0] >= 0.9:
            ninety.append(col)
        if dataframe[col].isna().sum() / dataframe.shape[0] >= 0.8:
            eighty.append(col)
        if dataframe[col].isna().sum() / dataframe.shape[0] >= 0.7:
            seventy.append(col)
        if dataframe[col].isna().sum() / dataframe.shape[0] >= 0.6:
            sixty.append(col)
        if dataframe[col].isna().sum() / dataframe.shape[0] >= 0.5:
            fifty.append(col)
        if dataframe[col].isna().sum() / dataframe.shape[0] >= 0.4:
            forty.append(col)
        if dataframe[col].isna().sum() / dataframe.shape[0] >= 0.3:
            thirty.append(col)
        if dataframe[col].isna().sum() / dataframe.shape[0] >= 0.2:
            twenty.append(col)
        if dataframe[col].isna().sum() / dataframe.shape[0] >= 0.1:
            ten.append(col)
        if dataframe[col].isna().sum() / dataframe.shape[0] >= 0.05:
            five.append(col)

    return {'len_90': len(ninety), '90 percent': ninety, 
            'len_80': len(eighty), '80 percent': eighty, 
            'len_70': len(seventy), '70 percent': seventy, 
            'len_60': len(sixty), '60 percent': sixty,
            'len_50': len(fifty), '50 percent': fifty, 
            'len_40': len(forty), '40 percent': forty, 
            'len_30': len(thirty), '30 percent': thirty, 
            'len_20': len(twenty), '20 percent': twenty, 
            'len_10': len(ten), '10 percent': ten, 
            'len_5': len(five), '5 percent': five}


### Skip one-hot encoding since features are ordinally encoded with integers ###

def rename_columns(dataframe):
    '''Uses pandas .rename() method to rename columns in a pandas dataframe.
    The new column names are retrieved from the names_dict dictionary 
    (keys = old column names, values = new columns names).
    This is not done inplace, so must save dataframe to new one variable
    Input: 
        dataframe: a pandas dataframe
        col_lst: a list of column headers that need to be replaced
    Output: 
        a dataframe that must be assigned to a variable
    '''
    names_dict = {'q67': 'sexual_id', 
                'sexid': 'sexual_cont', 
                'race7': 'race', 
                'q11': 'text_email_drive',
                'q12': 'carry_weapon',
                'q13': 'carry_weapon_school', 
                'q15': 'felt_unsafe', 
                'q16': 'threatened_w_weapon', 
                'q17': 'physical_fight', 
                'q21': 'dating_forced', 
                'q22': 'dating_physicall_hurt',
                'q23': 'bullied', 
                'q24': 'cyber_bullied', 
                'q25': 'sad_morethan_2wks', 
                'q26': 'consider_suicide', 
                'q33': 'smoked_30days',
                'q35': 'vaped_30days',
                'q37': 'use_tobacco_prods', 
                'q38': 'use_tobacco_smoke', 
                'q47': 'age_mj', 
                'q48': 'mjed_30days', 
                'q49': 'powder_use',
                'q51': 'heroin_use'}
    
    for col in dataframe.columns.to_list():
        if col in names_dict:
            dataframe.rename(columns={col: names_dict[col]}, inplace=True)

    return dataframe



def to_replace(pandas_obj, orig_val, new_val):
    '''Uses pandas .replace() method to replace values in a pandas series.
    This is not done inplace, so must save dataframe to new one 
    Input: 
        pandas_obj: can be a pandas series column or a dataframe
        original_val: a list of values from series
        new_val: a list of values to replace series with
           
    Output: 
        a new series or dataframe dependng on input
    '''
    return pandas_obj.replace(orig_val, new_val)






