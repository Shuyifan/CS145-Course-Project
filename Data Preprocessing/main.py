import pandas as pd
import preprocessFunc as pp

Location = r'/mnt/c/Users/shuyi/Desktop/out.csv'
outputLocation = r'/mnt/c/Users/shuyi/Desktop/output.csv'
categoryLocation = r'/mnt/c/Users/shuyi/Desktop/categories.csv'

data = pd.read_csv(Location)

###Preprocess the list like columns
"""output = pp.listProcess('categories', data)
output.to_csv(categoryLocation)"""

###Preprocess the dictonary like columns
dict_columns = ['attributes_Ambience', 'attributes_BestNights', 'attributes_BusinessParking', 'attributes_GoodForMeal', 'attributes_Music']

for column in dict_columns:
    output_df = pp.dictProcess(column, data)
    data.drop(column, axis = 1, inplace = True)
    data = pd.concat([data, output_df], axis = 1)

time_columns = ['hours_Monday', 'hours_Tuesday', 'hours_Wednesday', 'hours_Thursday', 'hours_Friday', 'hours_Saturday', 'hours_Sunday']

for column in time_columns:
    output_df = pp.timeProcess(column, data)
    data.drop(column, axis = 1, inplace = True)
    data = pd.concat([data, output_df], axis = 1)

##boolean_columns = ['attributes_AcceptsInsurance']
##pp.booleanProcess('attributes_AcceptsInsurance', data)

data.to_csv(outputLocation)
