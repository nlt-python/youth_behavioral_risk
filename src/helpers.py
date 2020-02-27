import pandas as pd
import numpy as np


def load_and_lst_cols():
    ''' 
    Loads data into a pandas data frame and lists all columnds.

    Input: None
    Output: 
        a dictionary where:
            - keys is the filename and
            - values are a tuple of:
                - a dataframe
                - a list of all columns/features in the dataframe
    '''
    file_paths = ['data/sadcq_natl.csv', 'data/sadcqn_natl.csv',
                'data/sadcq_dist.csv', 'data/sadcqn_dist.csv',
                'data/sadcq_state_a--m.csv', 'data/sadcqn_state_a--m.csv',
                'data/sadcq_state_n--z.csv', 'data/sadcqn_state_n--z.csv']
    
    dicts = {}
    for path in file_paths:
        df = pd.read_csv(path)   
        cols = df.columns.to_list()
        dicts[path[5:]] = (df, cols)

    return dicts



def filter_and_remove_nans(dataframe, year):
    ''' 
    Takes a dataframe and filters rows according to year column
    and removes columns that only contain nans as well as columns
    provided in cols_lst
    
    Input: 
    dataframe:
    year: float (not date time type)
    
    Output: a tuple of the new dataframe and the columns that were removed
    '''

    yr_mask = dataframe['year'] >= year
    new_df = dataframe[yr_mask]
    cols = dataframe.columns.to_list()
    remove_cols = [col for col in cols if dataframe[col].isna().sum() == dataframe.shape[0]]
    
    new_df.drop(columns=remove_cols, inplace=True)
    # return {'new_dataframe': new_df, 'columns_removed': remove_cols}
    return new_df, remove_cols



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
    forty, thirty_five, thirty, twenty_five,  = [], [], [], []
    twenty, fifteen, ten, five = [], [], [], []

    for col in df_columns:
        if dataframe[col].isna().sum() / dataframe.shape[0] >= 0.4:
            forty.append(col)
        if dataframe[col].isna().sum() / dataframe.shape[0] >= 0.35:
            thirty_five.append(col)
        if dataframe[col].isna().sum() / dataframe.shape[0] >= 0.3:
            thirty.append(col)
        if dataframe[col].isna().sum() / dataframe.shape[0] >= 0.25:
            twenty_five.append(col)
        if dataframe[col].isna().sum() / dataframe.shape[0] >= 0.2:
            twenty.append(col)
        if dataframe[col].isna().sum() / dataframe.shape[0] >= 0.15:
            fifteen.append(col)
        if dataframe[col].isna().sum() / dataframe.shape[0] >= 0.1:
            ten.append(col)
        if dataframe[col].isna().sum() / dataframe.shape[0] >= 0.05:
            five.append(col)

    return {'len_40': len(forty), '40 percent': forty,
            'len_35': len(thirty_five), '35 percent': thirty_five, 
            'len_30': len(thirty), '30 percent': thirty, 
            'len_25': len(twenty_five), '25 percent': twenty_five, 
            'len_20': len(twenty), '20 percent': twenty,
            'len_15': len(fifteen), '15 percent': fifteen, 
            'len_10': len(ten), '10 percent': ten, 
            'len_5': len(five), '5 percent': five}



def clean_data(dataframe):

    # Drop sexid2 column b/c it is condensed version of q67.
    
    # q67 and sexid ask the same question, but is phrased differently.
    'q67: Which of the following best describes you' # versus
    'sexid: Sexual identity'
    # Youth that do not respond to q67 also do not respond to sexid.
    # Drop q67
    # Drop sitetype, sitetypenum and since all records are District, 1
    # Drop survyear since it increments from 1 for first year survey was conducted
    # Drop race4 column since it is now fully encapsulated by race7
    dataframe.drop(columns=['sitetype', 'sitetypenum', 'sexid2', 'survyear', 'race4', 'q67'], inplace=True)

    # Originally, Yes = 1 and No = 2.
    # Changed to: Yes = 1 and No = 0.
    # Also changed gender, age, grade and race

    dataframe[['bullied', 'cyber_bullied', 'consider_suicide', 'sad_morethan_2wks']] = to_replace(dataframe[['bullied', 'cyber_bullied', 'consider_suicide', 'sad_morethan_2wks']], 
                                                                                               [1.0, 2.0], [1, 0])

    dataframe.loc[:, 'gender'] = to_replace(dataframe.loc[:, 'gender'], [1.0, 2.0], ['F', 'M'])
    dataframe.loc[:, 'gender'] = to_replace(dataframe.loc[:, 'gender'], ['F', 'M'], [1, 0])

    dataframe.loc[:, 'age'] = to_replace(dataframe.loc[:, 'age'], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 
                                                                  [12, 13, 14, 15, 16, 17, 18])

    dataframe.loc[:, 'grade'] = to_replace(dataframe.loc[:, 'grade'], [1.0, 2.0, 3.0, 4.0], 
                                                                      [9, 10, 11, 12])

    dataframe.loc[:, 'race'] = to_replace(dataframe.loc[:, 'race'], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 
                                                                                    ['Am. Indian', 'Asian', 'Black', 'Hispanic/Latino', 'Hawaiian/Pac. Isl.', 'White', 'Mixed'])

    dataframe.loc[:, 'sexid'] = to_replace(dataframe.loc[:, 'sexid'], [1.0, 2.0, 3.0, 4.0], 
                                                                      ['Heterosexual', 'Homosexual', 'Bisexual', 'Not Sure'])

    
    return dataframe

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
    names_dict = {'race7': 'race', 'sex': 'gender',
                  'q11': 'text_email_drive',
                  'q12': 'carry_weapon', 'q13': 'carry_weapon_school', 
                  'q15': 'felt_unsafe', 'q16': 'threatened_w_weapon', 
                  'q17': 'physical_fight', 
                  'q21': 'dating_forced', 'q22': 'dating_physically_hurt',
                  'q23': 'bullied', 'q24': 'cyber_bullied', 
                  'q25': 'sad_morethan_2wks', 'q26': 'consider_suicide', 
                  'q33': 'smoked_30days', 'q35': 'vaped_30days',
                  'q37': 'use_tobacco_prods', 'q38': 'use_tobacco_smoke', 
                  'q47': 'age_mj', 'q48': 'mjed_30days', 
                  'q49': 'powder_use', 'q51': 'heroin_use'}
    
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






