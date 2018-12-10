import pandas as pd
import preprocessFunc as pp
import numpy as np

Location = r'/mnt/c/Users/shuyi/OneDrive/CS145/Data/business.csv'
outputLocation = r'/mnt/c/Users/shuyi/OneDrive/CS145/Data/after preprocess/business_transformed_compacted.csv'

data = pd.read_csv(Location)

###Delete the useless attribute
useless_columns = ['address', 'attributes', 'attributes_AcceptsInsurance', 'attributes_AgesAllowed', \
                   'attributes_BYOB','attributes_BYOBCorkage', 'attributes_BestNights', \
                   'attributes_BusinessAcceptsBitcoin', 'attributes_CoatCheck', 'attributes_Corkage', \
                   'attributes_DietaryRestrictions', 'attributes_DogsAllowed', 'attributes_DriveThru', \
                   'attributes_GoodForDancing', 'attributes_HairSpecializesIn', 'attributes_HappyHour', \
                   'attributes_Music', 'attributes_Open24Hours', 'attributes_RestaurantsCounterService', \
                   'attributes_Smoking', 'city', 'hours', 'name', 'neighborhood', 'postal_code', 'state', \
                   'categories']
for column in useless_columns:
    data.drop(column, axis = 1, inplace = True)

###Preprocess the multi-class categories columns
categories_columns = ['attributes_BikeParking', 'attributes_Alcohol', \
                      'attributes_BusinessAcceptsCreditCards', 'attributes_ByAppointmentOnly', 'attributes_Caters',\
                      'attributes_GoodForKids', 'attributes_HasTV', \
                      'attributes_NoiseLevel', 'attributes_OutdoorSeating',\
                      'attributes_RestaurantsAttire', 'attributes_RestaurantsDelivery',\
                      'attributes_RestaurantsGoodForGroups', 'attributes_RestaurantsReservations', 'attributes_RestaurantsTableService',\
                      'attributes_RestaurantsTakeOut', 'attributes_WheelchairAccessible', 'attributes_WiFi']
for column in categories_columns:
    output_df = pp.categoriesProcess(column, data)
    data.drop(column, axis = 1, inplace = True)
    data = pd.concat([data, output_df], axis = 1)

###Preprocess the dictonary like columns
dict_columns = ['attributes_Ambience', 'attributes_BusinessParking','attributes_GoodForMeal']

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

###Preprocess the list like columns
#output_df = pp.listProcess('categories', data)
#data.drop('categories', axis = 1, inplace = True)
#data = pd.concat([data, output_df], axis = 1)

###Fill missing pricing value with mode
numeric_columns = ['attributes_RestaurantsPriceRange2']
for column in numeric_columns:
    pp.numericColumnProcess(column, data, insert = 'mode')

data.to_csv(outputLocation, index=False)