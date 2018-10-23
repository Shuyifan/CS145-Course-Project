import pandas as pd

df = pd.read_csv('/mnt/c/Users/shuyi/Desktop/business.csv')

ignore = ['address', 'attributes', 'attributes_Ambience', 'attributes_BestNights', 
        'attributes_BusinessParking', 'attributes_GoodForMeal','attributes_HairSpecializesIn',
         'attributes_Music', 'categories', 'hours', 'hours_Friday', 'hours_Monday',	'hours_Saturday', 
         'hours_Sunday', 'hours_Thursday', 'hours_Tuesday', 'hours_Wednesday', 'business_id', 'city',
         'latitude', 'longitude', 'name', 'neighborhood', 'postal_code', 'review_count', 'stars', 'state',
         'attributes_RestaurantsPriceRange2', 'attributes_DietaryRestrictions', 'is_open', 'attributes_WiFi',
         'attributes_AgesAllowed', 'attributes_Alcohol', 'attributes_NoiseLevel']
d = {}
d['attributes_AgesAllowed'] = ['allages', '18plus', '19plus', '21plus']
d['attributes_Alcohol'] = ['none', 'beer_and_wine', 'full_bar']
d['attributes_NoiseLevel'] = ['quiet', 'average', 'loud', 'very_loud']
d['attributes_WiFi'] = ['no', 'paid', 'free']
for col in df.columns.tolist():
    if col not in ignore:
        a = []
        b = df[col].tolist()
        for val in b:
            if val not in a and val == val:
                a.append(val)
        print (col)
        d[col] = sorted(a)
        print (d[col])
print (d)
for col in df.columns.tolist():
    if col not in ignore:
        for i in range(len(df[col])):
            if df[col][i] == df[col][i]:
                df.loc[i, col] = d[col].index(df[col][i])
        print (col)
        print (df[col])