import pandas as pd
import numpy as np
import time

"""
The function to preprocess a list like column by using one-hot encoding

:param attribute: the column name of the list like column
:param data: the data frame for preprocessing

:type attribute: str
:type data: pandas.DataFrame

:returns output_df: return a new dataframe which is the result preprocessing the list like column

:rtype output_df: pandas.DataFrame
"""
def listProcess(attribute, data):
    df = data[attribute]
    row_list = list()
    keys = set()
    for i in range(df.shape[0]):
        if(isinstance(df[i], str)):
            words = list()
            for word in df[i].split(", "):
                words.append(word)
                keys.add(word)
            row_list.append(words)
        else:
            row_list.append(list())
    
    print("Number of categories")
    print(len(keys))

    output_df = pd.DataFrame(np.zeros((len(row_list), len(keys))), columns=[attribute + "_" + s1 for s1 in keys])

    for i in range(len(row_list)):
        for word in row_list[i]:
            output_df[attribute + "_" + word][i] = 1

    return output_df

"""
The function to preprocess a dictionary like column by using one-hot encoding

:param attribute: the column name of the dictionary like column
:param data: the data frame for preprocessing

:type attribute: str
:type data: pandas.DataFrame

:returns output_df: return a new dataframe which is the result preprocessing the dictionary like column

:rtype output_df: pandas.DataFrame
"""
def dictProcess(attribute, data):
    df = data[attribute]

    dict_list = list()

    for i in range(df.shape[0]):
        if(isinstance(df[i], str)):
            dict_list.append(eval(df[i]))
        else:
            dict_list.append(dict())

    keys = set()

    for i in range(len(dict_list)):
        for key in dict_list[i].keys():
            keys.add(key)

    print("Number of key values in this column")
    print(len(keys))

    output_df = pd.DataFrame(np.zeros((len(dict_list), len(keys))), columns=[attribute + "_" + s1 for s1 in keys])

    for i in range(len(dict_list)):
        if(isinstance(dict_list[i], dict)):
            for key, item in dict_list[i].items():
                if(item):
                    output_df[attribute + "_" + key][i] = 1
                else:
                    output_df[attribute + "_" + key][i] = 0

    return output_df

"""
The function to preprocess a time-period like column

:param attribute: the column name of the time-period like column
:param data: the data frame for preprocessing

:type attribute: str
:type data: pandas.DataFrame

:returns output_df: return a new dataframe which is the result preprocessing the  time-period like column

:rtype output_df: pandas.DataFrame
"""
def timeProcess(attribute, data):
    df = data[attribute]
    new_attribute = ['open_time(min)', 'duration(min)']
    output_df = pd.DataFrame(np.zeros((df.shape[0], 2)), columns=[attribute + "_" + s1 for s1 in new_attribute])
    for i in range(df.shape[0]):
        if(isinstance(df[i], str)):
            hours = df[i].split("-")
            t1 = time.strptime(hours[0], "%H:%M")
            t2 = time.strptime(hours[1], "%H:%M")
            minute1 = t1.tm_hour * 60 + t1.tm_min
            minute2 = t2.tm_hour * 60 + t2.tm_min
            output_df[attribute + "_" + new_attribute[0]][i] = minute1
            if(minute2 - minute1 > 0):
                output_df[attribute + "_" + new_attribute[1]][i] = minute2 - minute1
            else:
                output_df[attribute + "_" + new_attribute[1]][i] = 1440 + (minute2 - minute1)
    return output_df

"""
The function to preprocess a multi-class categories (including boolean) column, encoding it into a hot vector, plus a column "_isMissing"

:param attribute: the column name of the multi-class categories column
:param data: the data frame for preprocessing

:type attribute: str
:type data: pandas.DataFrame

:returns output_df: return a new dataframe which is the result preprocessing the multi-class categories column

:rtype output_df: pandas.DataFrame
"""
def categoriesProcess(attribute, data):
    df = data[attribute]
    keys = set()
    for i in range(df.shape[0]):
        keys.add(df[i])
            
    print("Number of categories")
    print(len(keys))
    
    output_df = pd.DataFrame(np.zeros((df.shape[0], len(keys))), columns=[attribute + "_" + str(s1) for s1 in keys])

    for i in range(df.shape[0]):
        output_df[attribute + "_" + str(df[i])][i] = 1
    return output_df

"""
The function to preprocess a column with missing numeric data

:param attribute: the column name of the column with missing numeric data
:param data: the data frame for preprocessing
:param insert: the way we want to insert the missing value. 
               'mean': insert the missing with the mean of the column
               'mode': insert the missing with the mode of the column

:type attribute: str
:type data: pandas.DataFrame
:type insert: str
"""
def numericColumnProcess(attribute, data, insert = 'mean'):
    if(insert == 'mean'):
        insertNum = data[attribute].mean().values[0]
    elif(insert == 'mode'):
        insertNum = data[attribute].mode().values[0]
    for i in range(data.shape[0]):
        if(np.isnan(data[attribute][i])):
            data.loc[i, attribute] = insertNum