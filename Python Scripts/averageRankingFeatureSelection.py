import os
import re
import csv
import numpy as np
from subprocess import call
from operator import add
from sklearn.feature_selection import *
from sklearn.preprocessing import Imputer
np.set_printoptions(threshold='nan')

"""
This file is used to average the results of FS2Years function from featureSelection.py.
It computes the final selectKBest ranking for each target variable to be used in the prediction process in iteration 1.
The prefix of the result files is 'Summary - ' such as 'Summary - FeatureSelectionIndicator811 - k40 - f_regression.csv'

"""

fileList = []
for root, dirs, files in os.walk('./FormattedFilesWithoutMissingToNextYear'):    
    for afile in files:
    	if afile.startswith('FeatureSelectionIndicator'):
    		fileList.append(afile)

for afile in fileList:
	featureDict = dict()
	with open('./FormattedFilesWithoutMissingToNextYear/'+afile,'rb') as f:
		reader = csv.reader(f)
		ranking = 1
		countYear = 0
		year = 0
		for row in reader:
			if row[1] == 'score':
				ranking = 1
				year = row[0]
				countYear += 1
			elif len(row[0]) == 0:
				continue
			else:
				feature = row[0][:row[0].rfind(' - ')]
				oldStat = featureDict.get(feature, [0.0,0.0,list()])
				oldStat[2].append(year)
				newStat = [oldStat[0]+ranking, oldStat[1]+1, oldStat[2]]
				featureDict[feature] = newStat
				ranking += 1

		rows = []
		header = ['Indicator','Average Ranking','Year Count','Year Density','Exists in']
		rows.append(header)

		allStat = []
		for key, value in featureDict.iteritems():
			aRow = [key]
			val = [1.0*value[0]/value[1], value[1], value[1] / countYear, value[2]]
			aRow.extend(val)
			allStat.append(aRow)
		allStat.sort(key=lambda x: x[1],reverse=False)
		rows.extend(allStat)

		filename = './FormattedFilesWithoutMissingToNextYear/'+'Summary - '+afile
		with open(filename,'wb') as w:
			a = csv.writer(w, delimiter = ',')
			a.writerows(rows)
			w.close()

		f.close()