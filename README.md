<snippet>
	<content>
# Mining Country Indicators using Big Data Approach
This repository stores data files, python scripts, and results of the MSc Project "Exploring and Forecasting Country Indicators using Big Data Approach"
by Piyawat Lertvittayakumjorn (supervised by Dr Chao Wu and Dr Liu Yue). This project is submitted in partial fulfillment of the requirements for the MSc degree in
Computing / Machine Learning of Imperial College London. 
## Related Literature
To be published
## Usage
Please ask Dr Wu for the project report in order to understand overview of the project and structure of this repository.

In a nutshell, 
1. After collecting data from many sources and conducting ETL into a data warehouse, we divide the analysis process into three iterations. Each of the iterations consists of data formatting, data exploration, and data forecasting except iteration 3 which does not have data exploration.
2. Repository Structure
* Data Files : contains all formatted files for each iteration.
* Data Exploration : contains plots and exploration results in iteration 1 and 2. 
* Data Forecasting : contains feature selection and cross validation results from each iteration.  
* Python Scripts : contains python files used to perform data formatting, exploration, and forecasting.
* MINE.jar is a software to calculate MIC and MINE statistics which is one of the important tools in this project. I downloaded it from http://www.exploredata.net/Downloads. So, all credit of this software goes to its inventors (Reshef et al.). 
3. If you want to work on the original data warehouse, please download it from 
## Disclaimer
This repository is **not** the actual workspace of this project. The workspace is too messy for other people to understand (I'm so sorry), so I copy essential files and scripts from the workspace and restructure them so that you can find the things you want.
Therefore, it is impossible to rerun the experiment straightforwardly from this repository. But, if you want to do so, you may download this repo, edit and run python scripts on your own. I have added explanations to all functions for this reason.   
	</content>
</snippet>