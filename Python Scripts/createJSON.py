import os
import re
import csv
import time
import json
import numpy as np

def getHeader(filepath):
	with open(filepath, 'rb') as f:
		reader = csv.reader(f)
		header = next(reader)
		f.close()
	return header
	
def createJSON(filename, numTargets):
	# header = getHeader(filename)

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

	with open('2006-2013_data.json', 'w') as outfile:
		json.dump(allData, outfile, sort_keys=True, indent=4, separators=(',', ': '))

createJSON('New_2006-2013_FilteredColsTargetMissingBlank.csv', 8)