<snippet>
	<content>
		
# Mining Country Indicators using Big Data Approach
This repository stores data files, python scripts, and results of the MSc Project "Exploring and Forecasting Country Indicators using Big Data Approach"
by Piyawat Lertvittayakumjorn (supervised by Dr Chao Wu and Dr Liu Yue). This project is submitted in partial fulfillment of the requirements for the MSc degree in
Computing / Machine Learning of Imperial College London (2015-2016). 

## Related Literature
- Lertvittayakumjorn P, Wu C, Liu Y, Mi H, Guo Y. _Exploratory Analysis of Big Social Data Using MIC/MINE Statistics._ In 2017 9th International Conference on Social Informatics (SocInfo), Oxford, United Kingdom; Sep 2017. \[[Link](https://link.springer.com/chapter/10.1007/978-3-319-67256-4_41)\]
- Guo Y, Lertvittayakumjorn P, Wu C, Liu Y, Mi H. _Multiple Domains Interdisciplinary Analysis of Country Indicators._ In 2017 28th International Population Conference of the International Union for the Scientific Study of Population (IUSSP), Cape Town, South Africa; Oct 2017. \[[Link](https://iussp.confex.com/iussp/ipc2017/meetingapp.cgi/Paper/6999)\]

## Usage and Repository Structure
Please ask Dr Wu for the project report in order to understand overview of the project and structure of this repository.

In a nutshell, 

1. After collecting data from many sources and conducting ETL into a data warehouse, we divide the analysis process into three iterations. Each of the iterations consists of data formatting, data exploration, and data forecasting except iteration 3 which does not have data exploration.
2. If you want to work on the original data warehouse, please download it from [here](https://drive.google.com/file/d/0B6dxM_iLlLqwYmdNSEw0VG5seGs/view?usp=sharing)
3. Repository Structure

* Data Files : contains all formatted files for each iteration.
* Data Exploration : contains plots and exploration results in iteration 1 and 2. 
* Data Forecasting : contains feature selection and cross validation results from each iteration.  
* Python Scripts : contains python files used to perform data formatting, exploration, and forecasting.
	* createFileFromData.py : create data files of iteration 1 from the data warehouse (Please edit database connection at line 5 of this file before use)
	* aggregateToSingleFile.py : create staked data files of iteration 2 from the data files in iteration 1.
	* createJSON.py : create a json data file for iteration 3 from a stacked data file from iteration 2.
	* eda.py : calculate Spearman correlation and plot MIC-Rho graph
	* runMINE.py : execute MINE software and processing its results
	* featureSelection.py : perform feature selection using selectKBest method
	* averageRankingFeatureSelection.py : average the results from featureSelection.py.
	* plotGraphClustering.py : generate scatter plots of all relationships categorized by clusters
	* prediction.py : predict target variables in iteration 1 and 2
	* predictionJSON.py : predict target variables in iteration 3 using our method 
	* benchmarkTest.py : predict target variables in iteration 3 using our method and other benchmarks and report their accuracy	
* MINE.jar is a software to calculate MIC and MINE statistics which is one of the important tools in this project. I downloaded it from http://www.exploredata.net/Downloads. So, all credit of this MINE software goes to its inventors (Reshef et al.). 

## Disclaimer
This repository is **not** the actual workspace of this project. The workspace is too messy for other people to understand (I'm so sorry), so I copy essential files and scripts from the workspace and restructure them so that you can find the things you want.
As a result, it is impossible to rerun the experiment straightforwardly from this repository. But, if you want to do so, you may download this repo, edit and run python scripts on your own (in your local computer). I have added explanations to all functions for this reason.   
By combining them with my report, I'm sure that you are able to reproduce the experiments.

## Contact
If you have any inquiries, please email me at **plkumjorn@gmail.com**.
	</content>
</snippet>
