import os
import re
from subprocess import call
from copy import deepcopy
import csv

header = []
with open('./New Formatted Files/1919_2016-06-16-12-45-12.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        header = row
        break

fileList = []
for root, dirs, files in os.walk('./Old Formatted Files'):    
    for afile in files:
    	fileList.append(afile)

fileList = fileList[:-1]
print fileList

for afile in fileList:
	oldRows = []
	with open('./Old Formatted Files/'+afile, 'rb') as f:
	    reader = csv.reader(f)
	    for row in reader:
	        oldRows.append(row)
	    f.close()
	
	oldRows = oldRows[1:]
	rows = [deepcopy(header)]
	rows.extend(oldRows)
    
	filename = './Formatted Files/'+afile
	with open(filename,'wb') as w:
		a = csv.writer(w, delimiter = ',')
		a.writerows(rows)
	w.close()