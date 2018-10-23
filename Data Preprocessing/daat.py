import pandas as pd
import numpy as np

Location = r'/mnt/c/Users/shuyi/Desktop/business.csv'
outputLocation = r'/mnt/c/Users/shuyi/Desktop/output.csv'

def listProcess(attribute):
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
    print(len(keys))

    output_df = pd.DataFrame(np.zeros((len(row_list), len(keys))) - 1, columns=[attribute + "_" + s1 for s1 in keys])

    for i in range(len(row_list)):
        for word in row_list[i]:
            output_df[attribute + "_" + word][i] = 1

    return output_df


def dictProcess(attribute):
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

    output_df = pd.DataFrame(np.zeros((len(dict_list), len(keys))) - 1, columns=[attribute + "_" + s1 for s1 in keys])



    for i in range(len(dict_list)):
        if(isinstance(dict_list[i], dict)):
            for key, item in dict_list[i].items():
                if(item):
                    output_df[attribute + "_" + key][i] = 1
                else:
                    output_df[attribute + "_" + key][i] = 0

    return output_df

data = pd.read_csv(Location)

output = listProcess('categories')
output.to_csv("/mnt/c/Users/shuyi/Desktop/categories.csv")

"""
dict_columns = ['attributes_Ambience', 'attributes_BestNights', 'attributes_BusinessParking', 'attributes_GoodForMeal', 'attributes_Music']

for column in dict_columns:
    output_df = dictProcess(column)
    data.drop(column, axis = 1, inplace = True)
    data = pd.concat([data, output_df], axis = 1)

data.to_csv(outputLocation)
"""