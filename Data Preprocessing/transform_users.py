import csv
import os
from datetime import date

path = os.path.dirname(os.getcwd())
if not os.path.exists(path + "/Data/After Processing"):
    os.makedirs(path + "/Data/After Processing") 
csv.field_size_limit(1000000000)

with open(path + '/Data/users.csv', newline='') as infile:
	with open(path + '/Data/After Processing/users_transformed.csv', 'w', newline='') as outfile:
		reader = csv.reader(infile)
		writer = csv.writer(outfile)
		
		firstrow = True
		
		for row in reader:
			if firstrow:
				row.insert(1, 'compliment_total')
				row[14] = 'is_elite'
				row.insert(17, 'friend_count')
				row[23] = 'days_yelping'
				del row[16] # Friends
				del row[18] #Name
				writer.writerow(row)
				firstrow = False
			else:
				compliment_count = 0
				for i in range(1, 12):
					compliment_count += int(row[i])
				
				is_elite = 0
				if row[13] != "None":
					is_elite = 1
				
				friend_count = 0
				if row[15] != "None":
					friend_list = row[15].split(',')
					friend_count = len(friend_list)
				
				if row[21] != "None":
					right_now = date.today()
					join_date = date(int(row[21][0:4]), int(row[21][5:7]), int(row[21][8:10]))
					difference = (right_now - join_date).days
				
				row.insert(1, compliment_count)
				row[14] = str(is_elite)
				row.insert(17, friend_count)
				row[23] = difference
				del row[16]
				del row[18]
				writer.writerow(row)
				
				