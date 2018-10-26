import pandas as pd
import preprocessFunc as pp

Location = r'/mnt/c/Users/shuyi/OneDrive/CS145/Data/after preprocess/join_train_queries.csv'
outputLocation = r'/mnt/c/Users/shuyi/OneDrive/CS145/Data/after preprocess/join_final_output.csv'
categoryLocation = r'/mnt/c/Users/shuyi/OneDrive/CS145/Data/after preprocess/categories.csv'

data = pd.read_csv(Location)

###Delete the useless attribute
useless_columns = ['business_id', 'date', 'review_id', 'text', 'user_id', 'address', 'city', 'name', 'neighborhood', 'postal_code', 'state']
for column in useless_columns:
    data.drop(column, axis = 1, inplace = True)

###Delete the features that miss too many data
missing_columns = ['attributes', 'attributes_AcceptsInsurance', 'attributes_AgesAllowed', 'attributes_BYOB',\
                   'attributes_BYOBCorkage', 'attributes_BusinessAcceptsBitcoin',\
                   'attributes_ByAppointmentOnly', 'attributes_CoatCheck', 'attributes_Corkage',\
                   'attributes_DogsAllowed', 'attributes_DriveThru', 'attributes_GoodForDancing',\
                   'attributes_HappyHour', 'attributes_Open24Hours', 'attributes_RestaurantsCounterService',\
                   'attributes_Smoking', 'hours']
for column in missing_columns:
    data.drop(column, axis = 1, inplace = True)

###Preprocess the list like columns
"""output = pp.listProcess('categories', data)
output.to_csv(categoryLocation)"""

###Proprocess the boolean columns
"""boolean_columns = ['attributes_AcceptsInsurance', 'attributes_BYOB', 'attributes_BikeParking', 'attributes_BusinessAcceptsBitcoin',\
                   'attributes_BusinessAcceptsCreditCards', 'attributes_ByAppointmentOnly', 'attributes_Caters', 'attributes_CoatCheck',\
                   'attributes_Corkage', 'attributes_DogsAllowed', 'attributes_DriveThru', 'attributes_GoodForDancing',\
                   'attributes_GoodForKids', 'attributes_HappyHour', 'attributes_HasTV', 'attributes_Open24Hours', \
                   'attributes_OutdoorSeating', 'attributes_RestaurantsCounterService', 'attributes_RestaurantsDelivery',\
                   'attributes_RestaurantsGoodForGroups', 'attributes_RestaurantsReservations', 'attributes_RestaurantsTableService',\
                   'attributes_RestaurantsTakeOut', 'attributes_WheelchairAccessible', ]
for column in boolean_columns:
    pp.booleanProcess(column, data)
"""

###Preprocess the dictonary like columns
dict_columns = ['attributes_Ambience', 'attributes_BestNights', 'attributes_BusinessParking', 'attributes_GoodForMeal', 'attributes_Music']

for column in dict_columns:
    output_df = pp.dictProcess(column, data)
    data.drop(column, axis = 1, inplace = True)
    data = pd.concat([data, output_df], axis = 1)

###Preprocess the time columns
time_columns = ['hours_Monday', 'hours_Tuesday', 'hours_Wednesday', 'hours_Thursday', 'hours_Friday', 'hours_Saturday', 'hours_Sunday']

for column in time_columns:
    output_df = pp.timeProcess(column, data)
    data.drop(column, axis = 1, inplace = True)
    data = pd.concat([data, output_df], axis = 1)

data.to_csv(outputLocation)
