import csv
import os


# opening the CSV file
with open('list_attr_celeba3.csv', mode ='r') as file:            
	# reading the CSV file
	csvFile = csv.reader(file, delimiter=' ')
 
	# displaying the contents of the CSV file
	for lines in csvFile:
		try:
			if lines[1] == "-1" and lines[4] == "1": 
				 print("Removing:", lines[0])
				 os.remove(lines[0])
		except:
			print("Already deleted")