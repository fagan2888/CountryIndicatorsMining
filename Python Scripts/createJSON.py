import os
import re
import csv
import time
import json
import numpy as np

def createJSON(filename, numTargets, outputFilename = '2006-2013_data.json'):
	""" 
	Create a json data file for iteration 3 from a stacked data file from iteration 2.

	This function separates the information of each country in the input file 
	and transformed it into object (one object per country). 
	After that, it combines all objects into a large object 
	(key = country key, value = the data object of the corresponding country) and stores it in a single json file.

	Parameters
    ----------
    filename : string
        File name or file path of the stacked data file from iteration 2.

    numTargets : int
    	The number of targets in the input file.

    outputFilename : string
        File name of the json data file for iteration 3. It must end with '.json'. 

    Return Value
    ----------
    None

	"""

	allData = dict()

	f = open(filename, 'rb')
	reader = csv.reader(f)

	header = next(reader)
	targetName = header[1:numTargets+1]
	dataName = header[numTargets+1:]
	
	fromFile = list(reader)
	
	for row in fromFile:
		year = int(row[0][0:4])
		cid = int(row[0][7:10])
		cname = row[0][12:].decode('latin-1')
	
		targetList = row[1:numTargets+1]
		dataList = [float(x) for x in row[numTargets+1:]]

		if allData.has_key(cid):
			countryObject = allData.get(cid)
			countryObject['yearList'].append(year)
			countryObject['data'].append(dataList)
			countryObject['target'].append(targetList)
		else:
			firstDict = {'cid':cid, 'cname':cname, 'yearList':[year], 'targetName': targetName, 'dataName': dataName, 'data': [dataList], 'target':[targetList] }
			allData[cid] = firstDict

	with open(outputFilename, 'w') as outfile:
		json.dump(allData, outfile, sort_keys=True, indent=4, separators=(',', ': '))

createJSON('New_2006-2013_FilteredColsTargetMissingBlank.csv', 8)