import os
import re
import csv
import numpy as np
from subprocess import call
from operator import add
from sklearn.feature_selection import *
from sklearn.preprocessing import Imputer
np.set_printoptions(threshold='nan')

def writeCSV(path,aList):
	with open(path,'wb') as w:
		a = csv.writer(w, delimiter = ',')
		a.writerows(aList)
	w.close()

def FS1Year():
	fileList = []
	for root, dirs, files in os.walk('./Formatted Files Without Missing'):    
	    for afile in files:
	    	fileList.append(afile)

	targetList = [2704,2707,2713,2716,2718,808,811,1954]
	# targetList = [1994,1997,2003,2006,2008,807,810,1953]
	yearList = [(1998,2015),(2005,2015),(2005,2015),(2005,2015),(2005,2015),(1960,2014),(1961,2014),(2002,2012)]


	for i in range(len(targetList)):
		# i = 0
		rows = []
		for year in range(yearList[i][0],yearList[i][1]+1):
			# print str(year) + '-' + str(targetList[i]) 
			regex = re.compile("("+ str(year) +").*")
			files = [m.group(0) for l in fileList for m in [regex.search(l)] if m and len(l) == 28]
			# print files
			# call(["java","-jar","MINE.jar","./New Formatted Files/"+files[0],str(targetList[i]+1),"cv=0.5"])
			

			# load the CSV file as a numpy matrix
			# dataset = np.loadtxt('./New Formatted Files/'+files[0], delimiter=",", skiprows=1, usecols=tuple(range(1,3240)))
			# dataset = np.genfromtxt('./New Formatted Files/'+files[0], delimiter=",", names=True, autostrip=True, max_rows=10, missing_values=np.nan, usecols=tuple(range(1,30)))
			with open('./Formatted Files Without Missing/'+files[0],'rb') as f:
			    reader = csv.reader(f)
			    header = next(reader)
			    num_cols = len(header)
			    # print header
			    print i
			    target_idx = [idx for idx, item in enumerate(header) if item.startswith(str(targetList[i]).zfill(4))]
			    if len(target_idx) > 0:
			    	target = target_idx[0]-1
			    	print ('OK',year, targetList[i], './Formatted Files Without Missing/'+files[0])
			    else:
			    	print (year, targetList[i], './Formatted Files Without Missing/'+files[0])
			    	break
			    f.close()
			dataset = np.genfromtxt('./Formatted Files Without Missing/'+files[0], delimiter=",", skip_header=1, autostrip=True, missing_values=np.nan, usecols=tuple(range(1,num_cols)))
			# print (dataset.shape)
			X = np.concatenate((dataset[:,0:target],dataset[:,target+1:dataset.shape[1]]),axis=1)
			# X = np.concatenate((dataset[:,0:2],dataset[:,3:dataset.shape[1]),axis=1)
			y = dataset[:,target]
			# print tuple(range(1,3240))
			# print dataset.dtype.names[0]
			# print dataset.dtype.names[-1]
			# print dataset[0]
			imp = Imputer(missing_values='NaN', strategy='median', axis=0)
			imputedX = imp.fit_transform(X,y)
			imputedX = np.array([imputedX[j] for j in range(imputedX.shape[0]) if not np.isnan(y[j])])
			deleteMissingY = np.array([x1 for x1 in y if not np.isnan(x1)])
			# print dataset[0]
			# print (imputedX.shape, y.shape)
			# print (imputedX.shape, deleteMissingY.shape)
			# print (np.any(np.isnan(imputedX)), np.all(np.isfinite(imputedX)))
			# imputedX_new = SelectKBest(chi2, k=10).fit_transform(imputedX, y)
			k = 30
			selection = SelectKBest(f_regression, k=k)
			imputedX_new = selection.fit_transform(imputedX, deleteMissingY)
			# print (len(selection.get_support()), len(header[1:target+1]+header[target+2:]))
			selectedFeatures = [[item, selection.scores_[idx], selection.pvalues_[idx]] for idx, item in enumerate(header[1:target+1]+header[target+2:]) if selection.get_support()[idx]]
			selectedFeatures.sort(key=lambda x: x[1], reverse=True)
			# for sf in selectedFeatures:
			# 	print sf
			# print selection.scores_
			# print selection.get_support()
			# print (imputedX_new.shape, y.shape)
			# print (imputedX_new.shape, deleteMissingY.shape)
			# print imputedX[0,1994]
			# print dataset['3137_Estimates_and_projections_of_the_total_population_by_sex_age_and_rural__urban_areasSexTotal_10year_age_bands__2534_Geographical_coverage__National_Thousands_Persons__ILO']
			# print dataset
			# separate the data from the target attributes
			# X = np.concatenate((imputedDataset[:,0:7],imputedDataset[:,0:7]),axis=1)
			# y = imputedDataset[:,8]
			rows.append([year, 'score', 'p-value'])
			rows.extend(selectedFeatures)
			rows.append(['', '', ''])
			print 'Hey'

		filename = './Feature Selection/'+('Indicator%d - k%d - %s.csv' % (targetList[i], k, 'f_regression'))
		with open(filename,'wb') as w:
			a = csv.writer(w, delimiter = ',')
			a.writerows(rows)
		w.close()
		
def FS2Years():
	fileList = []
	for root, dirs, files in os.walk('./FormattedFilesWithoutMissingToNextYear'):    
	    for afile in files:
	    	fileList.append(afile)

	targetList = [2704,2707,2713,2716,2718,808,811,1954]
	# targetList = [1994,1997,2003,2006,2008,807,810,1953]
	yearList = [(1998,2015),(2005,2015),(2005,2015),(2005,2015),(2005,2015),(1960,2014),(1961,2014),(2002,2012)]


	for i in range(len(targetList)):
		# i = 0
		rows = []
		for year in range(yearList[i][0],yearList[i][1]+1):
			# print str(year) + '-' + str(targetList[i]) 
			regex = re.compile("("+ str(year) +").*")
			files = [m.group(0) for l in fileList for m in [regex.search(l)] if m and len(l) == 28]
			

			# load the CSV file as a numpy matrix
			with open('./FormattedFilesWithoutMissingToNextYear/'+files[0],'rb') as f:
			    reader = csv.reader(f)
			    header = next(reader)
			    num_cols = len(header)
			    # print header
			    print i
			    target_idx = [idx for idx, item in enumerate(header) if item.startswith(str(targetList[i]).zfill(4)+'N')]
			    regex = re.compile("....N:.*")
			    nextYearIDs = [idx for idx, item in enumerate(header) if regex.search(item)]
			    nextYearCount = len(nextYearIDs)
			    if len(target_idx) > 0:
			    	target = target_idx[0]-1
			    	print ('OK',year, targetList[i], './FormattedFilesWithoutMissingToNextYear/'+files[0])
			    else:
			    	print (year, targetList[i], './FormattedFilesWithoutMissingToNextYear/'+files[0])
			    	break
			    f.close()
			dataset = np.genfromtxt('./FormattedFilesWithoutMissingToNextYear/'+files[0], delimiter=",", skip_header=1, autostrip=True, missing_values=np.nan, usecols=tuple(range(1,num_cols)))
			# print (dataset.shape)
			# X = np.concatenate((dataset[:,0:target],dataset[:,target+1:dataset.shape[1]]),axis=1)
			X = dataset[:,nextYearCount:dataset.shape[1]]
			# X = np.concatenate((dataset[:,0:2],dataset[:,3:dataset.shape[1]),axis=1)
			y = dataset[:,target]
			
			imp = Imputer(missing_values='NaN', strategy='median', axis=0)
			imputedX = imp.fit_transform(X,y)
			imputedX = np.array([imputedX[j] for j in range(imputedX.shape[0]) if not np.isnan(y[j])])
			deleteMissingY = np.array([x1 for x1 in y if not np.isnan(x1)])

			k = 40
			selection = SelectKBest(f_regression, k=k)
			imputedX_new = selection.fit_transform(imputedX, deleteMissingY)
			
			selectedFeatures = [[item, selection.scores_[idx], selection.pvalues_[idx]] for idx, item in enumerate(header[nextYearCount+1:]) if selection.get_support()[idx]]
			selectedFeatures.sort(key=lambda x: x[1], reverse=True)
			
			rows.append([year, 'score', 'p-value'])
			rows.extend(selectedFeatures)
			rows.append(['', '', ''])
			print 'Hey'

		filename = './FormattedFilesWithoutMissingToNextYear/'+('FeatureSelectionIndicator%d - k%d - %s.csv' % (targetList[i], k, 'f_regression'))
		with open(filename,'wb') as w:
			a = csv.writer(w, delimiter = ',')
			a.writerows(rows)
		w.close()

def FLargeFile():
	targetList = [2704,2707,2713,2716,2718,808,811,835]
	with open('2006-2013_FilteredColsTargetMissingBlank.csv','rb') as f:
	    reader = csv.reader(f)
	    header = next(reader)
	    num_cols = len(header)
	    f.close()
	dataset = np.genfromtxt('2006-2013_FilteredColsTargetMissingBlank.csv', delimiter=",", skip_header=1, autostrip=True, missing_values=np.nan, usecols=tuple(range(1,num_cols)))

	for target in range(len(targetList)):
		X = dataset[:,8:dataset.shape[1]]
		y = dataset[:,target]
		
		newX = np.array([X[j] for j in range(X.shape[0]) if not np.isnan(y[j])])
		deleteMissingY = np.array([x1 for x1 in y if not np.isnan(x1)])

		k = newX.shape[1]
		selection = SelectKBest(f_regression, k=k)
		imputedX_new = selection.fit_transform(newX, deleteMissingY)

		selectedFeatures = [[item, selection.scores_[idx], selection.pvalues_[idx]] for idx, item in enumerate(header[9:]) if selection.get_support()[idx]]
		selectedFeatures.sort(key=lambda x: x[1], reverse=True)

		rows = []
		rows.append([targetList[target], 'score', 'p-value'])
		rows.extend(selectedFeatures)

		filename = './Feature Selection 2006-2013/'+('AllYear - Indicator%d - k%d - %s.csv' % (targetList[target], k, 'f_regression'))
		writeCSV(filename,rows)

FLargeFile()