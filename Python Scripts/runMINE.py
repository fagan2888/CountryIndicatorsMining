import os
import re
import csv
import numpy as np
from subprocess import call
from operator import add

def MINE1Year():
	fileList = []
	for root, dirs, files in os.walk('./New Formatted Files'):    
	    for afile in files:
	    	fileList.append(afile)

	# targetList = [2704,2707,2713,2716,2718,808,811,1954]
	targetList = [1994,1997,2003,2006,2008,807,810,1953]
	yearList = [(1998,2015),(2005,2015),(2005,2015),(2005,2015),(2005,2015),(1960,2014),(1961,2014),(2002,2012)]

	# Run MINE
	# --------------------------------------------------------------------------------------
	# for i in range(len(targetList)):
	# 	for year in range(yearList[i][0],yearList[i][1]+1):
	# 		# print str(year) + '-' + str(targetList[i]) 
	# 		regex = re.compile("("+ str(year) +").*")
	# 		files = [m.group(0) for l in fileList for m in [regex.search(l)] if m]
	# 		# print files
	# 		call(["java","-jar","MINE.jar","./New Formatted Files/"+files[0],str(targetList[i]+1),"cv=0.5"])

	# Combine Results
	# --------------------------------------------------------------------------------------
	for i in range(len(targetList)):
		# >>> Find all result files
		regex = re.compile(".*(mv="+str(targetList[i]+1)+").*(csv)")
		files = [m.group(0) for l in fileList for m in [regex.search(l)] if m]
		# print files
		stat = {}
		numFiles = len(files)
		indicator = ''

		# >>> Collect Results from each file
		for afile in files:
			with open('./New Formatted Files/'+afile, 'rb') as f:
				reader = csv.reader(f)
				for row in reader:
					if row[0] == 'X var':
						continue
					indicator = row[0]
					oldStat = stat.get(row[1], [0.0,0.0,0.0,0.0,0.0,0.0,0.0])
					measure = row[2:]
					measure.append(1.0)
					newStat = [float(x) for x in measure]
					stat[row[1]] = map(add, oldStat, newStat)
				f.close()

		# >>> Combine results (Find mean)
		rows = []
		header = ['X var','Y var','MIC (strength)','MIC-p^2 (nonlinearity)','MAS (non-monotonicity)','MEV (functionality)','MCN (complexity)','Linear regression (p)']
		rows.append(header)

		allStat = []
		for key, value in stat.iteritems():
			aRow = [indicator]
			aRow.append(key)
			val = [1.0*x/value[6] for x in value[:-1]]
			aRow.extend(val)
			allStat.append(aRow)
		allStat.sort(key=lambda x: x[2],reverse=True)
		rows.extend(allStat)
		# for i in rows:
		# 	print i

		# >>> Write file
		filename = './New Formatted Files/Indicator'+str(targetList[i]+1)+'.csv'
		with open(filename,'wb') as w:
			a = csv.writer(w, delimiter = ',')
			a.writerows(rows)
		w.close()

		print indicator

def MINE2Years():
	fileList = []
	for root, dirs, files in os.walk('./FormattedFilesWithoutMissingToNextYear'):    
	    for afile in files:
	    	fileList.append(afile)

	# targetList = [2704,2707,2713,2716,2718,808,811,1954]
	targetList = [1994,1997,2003,2006,2008,807,810,1953]
	yearList = [(1998,2015),(2005,2015),(2005,2015),(2005,2015),(2005,2015),(1960,2014),(1961,2014),(2002,2012)]

	# Run MINE
	# --------------------------------------------------------------------------------------
	
	for afile in fileList:
		with open('./FormattedFilesWithoutMissingToNextYear/'+afile, 'rb') as f:
			reader = csv.reader(f)
			header = next(reader)
			regex = re.compile("....N:.*")
			nextYearID = [m.group(0) for l in header for m in [regex.search(l)] if m]
			nextYearIndicator = len(nextYearID)
			f.close()
		print (nextYearIndicator, afile, '"./FormattedFilesWithoutMissingToNextYear/'+afile+'"')
		call(["java","-jar","MINE.jar","./FormattedFilesWithoutMissingToNextYear/"+afile,"-pairsBetween", str(nextYearIndicator+1),"cv=0.5"])
		# break

def MINE2YearsCombineResults():
	fileList = []
	regex = re.compile(".*between.*(csv)")
	for root, dirs, files in os.walk('./FormattedFilesWithoutMissingToNextYear'):    
	    for afile in files:
	    	if regex.search(afile):
	    		fileList.append(afile)

	targetList = [2704,2707,2713,2716,2718,808,811,1954]
	# targetList = [1994,1997,2003,2006,2008,807,810,1953]
	yearList = [(1998,2015),(2005,2015),(2005,2015),(2005,2015),(2005,2015),(1960,2014),(1961,2014),(2002,2012)]


	# Combine Results
	# --------------------------------------------------------------------------------------
	for i in range(len(targetList)):
		# >>> Find all result files
		files = [afile for afile in fileList if int(afile[0:4]) >= yearList[i][0]-1 and int(afile[0:4]) < yearList[i][1]]
		# print files
		stat = {}
		numFiles = len(files)
		indicator = ''

		# >>> Collect Results from each file
		for afile in files:
			with open('./FormattedFilesWithoutMissingToNextYear/'+afile, 'rb') as f:
				reader = csv.reader(f)
				for row in reader:
					if row[0] == 'X var' or not row[0].startswith(str(targetList[i]).zfill(4)+'N'):
						continue
					indicator = row[0]
					oldStat = stat.get(row[1][:row[1].rfind(' - ')], [0.0,0.0,0.0,0.0,0.0,0.0,0.0])
					measure = row[2:]
					measure.append(1.0)
					newStat = [float(x) for x in measure]
					stat[row[1][:row[1].rfind(' - ')]] = map(add, oldStat, newStat)
				f.close()

		# >>> Combine results (Find mean)
		rows = []
		header = ['X var','Y var','MIC (strength)','MIC-p^2 (nonlinearity)','MAS (non-monotonicity)','MEV (functionality)','MCN (complexity)','Linear regression (p)','Num Year Exists']
		rows.append(header)

		allStat = []
		for key, value in stat.iteritems():
			aRow = [indicator]
			aRow.append(key)
			val = [1.0*x/value[6] for x in value[:-1]]
			val.append(value[6])
			aRow.extend(val)
			allStat.append(aRow)
		allStat.sort(key=lambda x: x[2],reverse=True)
		rows.extend(allStat)
		# for i in rows:
		# 	print i

		# >>> Write file
		filename = './FormattedFilesWithoutMissingToNextYear/Indicator'+str(targetList[i]+1)+'.csv'
		with open(filename,'wb') as w:
			a = csv.writer(w, delimiter = ',')
			a.writerows(rows)
		w.close()

		print indicator

def separateResultsByIndicator(originalFilename, folderName, newFilePrefix):
	# f = open('2006-2013_FilteredColsNotImputed.csv,between[break=9],cv=0.5,B=n^0.6,Results.csv','rb')
	f = open(originalFilename,'rb')
	reader = csv.reader(f)
	header = next(reader)
	MINEResults = list(reader)
	category = dict()
	for row in MINEResults:
		category[row[0]] = category.get(row[0],[header])+[row]
	for key, value in category.items():
		# filename = './Feature Selection 2006-2013/MINE-Indicator'+key[0:4]+'.csv'
		filename = './'+folderName+'/'+newFilePrefix+key[0:4]+'.csv'
		with open(filename,'wb') as w:
			a = csv.writer(w, delimiter = ',')
			a.writerows(value)
		w.close()

def separateResultsByIndicatorToARFF():
	f = open('./New_2006-2013_FilteredColsNotImputed.csv,between[break=9],cv=0.5,B=n^0.6,Results.csv','rb')
	# f = open('./2011_2016-08-23-03-22-10.csv,between[break=9],cv=0.5,B=n^0.6,Results.csv','rb')
	reader = csv.reader(f)
	header = next(reader)
	MINEResults = list(reader)
	category = dict()
	for row in MINEResults:
		thisRow = row
		thisRow[0] = "'"+thisRow[0].replace("'",";")+"'"
		thisRow[1] = "'"+thisRow[1].replace("'",";")+"'"
		category[thisRow[0]] = category.get(thisRow[0],[])+[thisRow]
	for key, value in category.items():
		filename = './For Clustering/MINE-Indicator-2006TO2013-'+key[1:5]+'.arff'
		# filename = './For Clustering/MINE-Indicator-2011-'+key[1:5]+'.arff'
		with open(filename,'wb') as w:
			w.write('@relation MINE-Indicator-2006TO2013-%s\n\n'%(key[1:5]))
			# w.write('@relation MINE-Indicator-2011-%s\n\n'%(key[1:5]))
			w.write('@attribute XVar string\n@attribute YVar string\n@attribute MIC numeric\n@attribute MIC-p2 numeric\n@attribute MAS numeric\n@attribute MEV numeric\n@attribute MCN numeric\n@attribute p numeric\n\n')
			w.write('@data\n')
			for aRow in value:
				w.write(",".join(aRow)+'\n')
		w.close()

# separateResultsByIndicatorToARFF()
separateResultsByIndicator('2011_2016-08-23-03-22-10.csv,between[break=9],cv=0.5,B=n^0.6,Results.csv', 'For Clustering', 'MINE-Indicator-2011-')