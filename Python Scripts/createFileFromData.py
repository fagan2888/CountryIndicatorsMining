import MySQLdb
import time
import csv

db = MySQLdb.connect(host="xxxx",    	# your host, usually localhost
	    user="yyyy",         			# your username such as root
	    passwd="zzzz",  				# your password
	    db="world_indicators")        	# name of the database

def createFileByYear(year, destinationFolder='Formatted Files'):
	""" 
	Create a data file of the desired year by querying the data warehouse 
	and save the file into the destinationFolder in the same directory.   

	Each row in the created file is for one country e.g. 001: Aruba (ABW)
	Each column in the created file is for one indicator e.g. 0001: Agricultural machinery; tractors - WDI

	Parameters
	----------
	year : int
	    The year of the created file.

	destinationFolder : string 
		The name of folder in the same directory into which the created file is saved.

	Return Value
	----------
	(filename, data density)

	"""
	rows = []
	allRecords = 0
	
	# Setup database connection
	cur = db.cursor()
	
	# Create a header row
	indicatorList = ['Country']
	indicatorIDs = []

	cur.execute("SELECT * FROM indicator_dim ORDER BY indicator_key ASC;")	
	for row in cur.fetchall():
		indicator = str(row[0]).zfill(4) + ': ' + row[1]
		if row[2] != None:
			indicator += ' - ' + row[2]
		if row[3] != None:
			indicator += ' (' + row[3] + ')'
		indicator += ' - ' + row[6]
		indicator = indicator.replace(',', ';')
		# print indicator
		indicatorList.append(indicator)
		indicatorIDs.append(row[0])
	
	# if year == 1919:
	# 	onlyIndicators = indicatorList[1:]
	# 	for i in range(len(onlyIndicators)):
	# 		print onlyIndicators[i] + '==>' + str(i)

	rows.append(indicatorList)

	# Create a row for each country
	countryList = []
	cur.execute("SELECT * FROM country_dim ORDER BY country_key ASC;")	
	for row in cur.fetchall():
		countryList.append((row[0],row[1],row[2]))
	# countryList = [(3,'AFG','Afghanistan')]
	for (cid, abbr, cname) in countryList:
		cur.execute("SELECT * FROM record_fact WHERE country_key = %d AND year = %d ORDER BY indicator_key ASC;" % (cid,year))	
		facts = []
		for row in cur.fetchall():
			# print (row[0],row[1],row[2],row[4])
			facts.append((row[0],row[1],row[2],row[4]))
		allRecords += len(facts)
		arow = [str(cid).zfill(3) + ': ' + cname + ' (' + abbr + ')']
		for i in range(len(indicatorIDs)):
			if len(facts) == 0:
				arow.append('')
			elif facts[0][1] == indicatorIDs[i]:
				arow.append(facts[0][3])
				facts.pop(0)
			else:
				arow.append('')
		# print arow
		rows.append(arow)
		# print (cid, abbr, cname)

	db.close()

	# CWrite to file
	filename = './'+destinationFolder+'/'+str(year)+'_'+time.strftime("%Y-%m-%d-%H-%M-%S")+'.csv'
	with open(filename,'wb') as w:
		a = csv.writer(w, delimiter = ',')
		a.writerows(rows)
	w.close()

	return (filename, 1.0*allRecords/(len(countryList)*len(indicatorIDs)))


def createFileByYearIgnoreMissingColumn(year, destinationFolder='Formatted Files Without Missing'):
	""" 
	Create a data file of the desired year by querying the data warehouse 
	and save the file into the destinationFolder in the same directory.   

	The file does not include the column with no data in that year.

	Each row in the created file is for one country e.g. 001: Aruba (ABW)
	Each column in the created file is for one indicator e.g. 0001: Agricultural machinery; tractors - WDI

	Parameters
	----------
	year : int
	    The year of the created file.

	destinationFolder : string 
		The name of folder in the same directory into which the created file is saved.

	Return Value
	----------
	(filename, data density)

	"""
	rows = []
	allRecords = 0
	
	# Setup database connection
	cur = db.cursor()

	
	# Create a header row
	indicatorList = ['Country']
	indicatorIDs = []

	cur.execute("SELECT DISTINCT indicator_key FROM record_fact WHERE year = %d ORDER BY indicator_key ASC;" % (year))	
	for row in cur.fetchall():
		indicatorIDs.append(row[0])
	
	i = 0	
	for indicator_key in indicatorIDs:
		cur.execute("SELECT * FROM indicator_dim WHERE indicator_key = %d;" % (indicator_key))
		row = cur.fetchone()
		indicator = str(row[0]).zfill(4) + ': ' + row[1]
		if row[2] != None:
			indicator += ' - ' + row[2]
		if row[3] != None:
			indicator += ' (' + row[3] + ')'
		indicator += ' - ' + row[6]
		indicator += ' - ' + str(i)
		indicator = indicator.replace(',', ';')
		# print indicator
		indicatorList.append(indicator)
		# indicatorIDs.append(row[0])
		i += 1
	
	# if year == 1919:
	# 	onlyIndicators = indicatorList[1:]
	# 	for i in range(len(onlyIndicators)):
	# 		print onlyIndicators[i] + '==>' + str(i)

	rows.append(indicatorList)

	# Create a row for each country
	countryList = []
	cur.execute("SELECT * FROM country_dim ORDER BY country_key ASC;")	
	for row in cur.fetchall():
		countryList.append((row[0],row[1],row[2]))
	# countryList = [(3,'AFG','Afghanistan')]
	for (cid, abbr, cname) in countryList:
		cur.execute("SELECT * FROM record_fact WHERE country_key = %d AND year = %d ORDER BY indicator_key ASC;" % (cid,year))	
		facts = []
		for row in cur.fetchall():
			# print (row[0],row[1],row[2],row[4])
			facts.append((row[0],row[1],row[2],row[4]))
		allRecords += len(facts)
		arow = [str(cid).zfill(3) + ': ' + cname + ' (' + abbr + ')']
		for i in range(len(indicatorIDs)):
			if len(facts) == 0:
				arow.append('')
			elif facts[0][1] == indicatorIDs[i]:
				arow.append(facts[0][3])
				facts.pop(0)
			else:
				arow.append('')
		# print arow
		rows.append(arow)
		# print (cid, abbr, cname)

	db.close()

	# CWrite to file
	filename = './'+destinationFolder+'/'+str(year)+'_'+time.strftime("%Y-%m-%d-%H-%M-%S")+'.csv'
	with open(filename,'wb') as w:
		a = csv.writer(w, delimiter = ',')
		a.writerows(rows)
	w.close()

	return (filename, 1.0*allRecords/(len(countryList)*len(indicatorIDs)))

def createFileToTheNextYearIgnoreMissingColumn(year, destinationFolder='NewFormattedFilesWithoutMissingToNextYear'):
	""" 
	Create a data file of the desired year with target values of the next year by querying the data warehouse 
	and save the file into the destinationFolder in the same directory.   

	The file does not include the column with no data in that year.

	Each row in the created file is for one country e.g. 001: Aruba (ABW)
	Each column in the created file is for one indicator e.g. 0001: Agricultural machinery; tractors - WDI
	The first few columns are for target indicators of the next year e.g. 2704N: Corruption perceptions index - Transparency - 0

	This function is used to create data files in iteration 1.

	Parameters
	----------
	year : int
	    The year of the created file.

	destinationFolder : string 
		The name of folder in the same directory into which the created file is saved.

	Return Value
	----------
	(filename, data density, the number of next year indicators included)

	"""
	targetList = [2704,2707,2713,2716,2718,808,811,1954]
	# targetList = [1994,1997,2003,2006,2008,807,810,1953]
	yearList = [(1998,2015),(2005,2015),(2005,2015),(2005,2015),(2005,2015),(1960,2014),(1961,2014),(2002,2012)]

	rows = []
	allRecords = 0
	
	# Setup database connection
	cur = db.cursor()

	
	# Create a header row
	indicatorList = ['Country']
	indicatorIDs = []

	for yrIndex in range(len(yearList)):
		if year >= yearList[yrIndex][0]-1 and year < yearList[yrIndex][1]:
			indicatorIDs.append(targetList[yrIndex])

	nextYearIndicator = len(indicatorIDs)

	cur.execute("SELECT DISTINCT indicator_key FROM record_fact WHERE year = %d ORDER BY indicator_key ASC;" % (year))	
	for row in cur.fetchall():
		indicatorIDs.append(row[0])
	
	i = 0	
	for indicator_key in indicatorIDs:
		isN = 'N' if i < nextYearIndicator else ''
		cur.execute("SELECT * FROM indicator_dim WHERE indicator_key = %d;" % (indicator_key))
		row = cur.fetchone()
		indicator = str(row[0]).zfill(4) + isN + ': ' + row[1]
		if row[2] != None:
			indicator += ' - ' + row[2]
		if row[3] != None:
			indicator += ' (' + row[3] + ')'
		indicator += ' - ' + row[6]
		indicator += ' - ' + str(i)
		indicator = indicator.replace(',', ';')
		# print indicator
		indicatorList.append(indicator)
		# indicatorIDs.append(row[0])
		i += 1
	
	# if year == 1919:
	# 	onlyIndicators = indicatorList[1:]
	# 	for i in range(len(onlyIndicators)):
	# 		print onlyIndicators[i] + '==>' + str(i)

	rows.append(indicatorList)

	# Create a row for each country
	countryList = []
	cur.execute("SELECT * FROM country_dim ORDER BY country_key ASC;")	
	for row in cur.fetchall():
		countryList.append((row[0],row[1],row[2]))
	# countryList = [(3,'AFG','Afghanistan')]
	for (cid, abbr, cname) in countryList:
		arow = [str(cid).zfill(3) + ': ' + cname + ' (' + abbr + ')']

		for yrIndex in range(len(yearList)):
			if year >= yearList[yrIndex][0]-1 and year < yearList[yrIndex][1]:
				cur.execute("SELECT * FROM record_fact WHERE country_key = %d AND year = %d AND indicator_key = %d;" % (cid,year+1,targetList[yrIndex]))
				row_count = cur.rowcount
				if row_count == 0:
					arow.append('')
				else:
					row = cur.fetchone()
					arow.append(row[4])
					allRecords += 1

		cur.execute("SELECT * FROM record_fact WHERE country_key = %d AND year = %d ORDER BY indicator_key ASC;" % (cid,year))	
		facts = []
		for row in cur.fetchall():
			# print (row[0],row[1],row[2],row[4])
			facts.append((row[0],row[1],row[2],row[4]))
		allRecords += len(facts)
		
		for i in range(nextYearIndicator,len(indicatorIDs)):
			if len(facts) == 0:
				arow.append('')
			elif facts[0][1] == indicatorIDs[i]:
				arow.append(facts[0][3])
				facts.pop(0)
			else:
				arow.append('')
		# print arow
		rows.append(arow)
		# print (cid, abbr, cname)

	db.close()

	# CWrite to file
	filename = './'+destinationFolder+'/'+str(year)+'_'+time.strftime("%Y-%m-%d-%H-%M-%S")+'.csv'
	with open(filename,'wb') as w:
		a = csv.writer(w, delimiter = ',')
		a.writerows(rows)
	w.close()

	return (filename, 1.0*allRecords/(len(countryList)*len(indicatorIDs)), nextYearIndicator)

# year = 2010
# for year in range(1959,2016):
# 	print createFileToTheNextYearIgnoreMissingColumn(year)
# for year in range(2005,2017):
# 	createFileByYear(year)

createFileToTheNextYearIgnoreMissingColumn(2013)