<div align="center"> <h1 align="center"> Prediction of the biological response of molecules by their chemical composition </h1> </div>
<div align="center"> <h3 align="center"> by Aleksey Kolychev </h3> </div>

### [Link to the dataset](https://lms.skillfactory.ru/assets/courseware/v1/9f2add5bca59f8c4df927432d605fff3/asset-v1:SkillFactory+DSPR-2.0+14JULY2021+type@asset+block/_train_sem09__1_.zip)

- The practice is based on the [***Kaggle competition: Prediction of a Biological Reaction***](https://www.kaggle.com/c/bioresponse).
- The data is presented in CSV format.  Each row represents a molecule. 

<div align="center"> <h2 align="center"> Description of the dataset </h2> </div> 

<div align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The first Activity column contains experimental data describing the actual biological response [0, 1]; The remaining columns D1-D1776 represent molecular descriptors — these are calculated properties that can capture some characteristics of the molecule, such as size, shape or composition of elements.</div>



<div align="center"> <h2 align="center"> Initial data </h2> </div>

- No preprocessing is required, the data is already encoded and normalized.
- F1-score is used as a metric.

<div align="center"> <h2 align="center"> Results </h2> </div>

1. Two models were trained: "logistic regression" and "random forest".
2. Next, the selection of hyperparameters is made using basic and advanced optimization methods.
3. Four methods were used: GridSearchCV, Randomized Search CV, Hyperopt, Optuna (at least once).
4. The maximum number of iterations does not exceed 50.


<div align="center"> <h2 align="center"> Description of the dataset </h2> </div>

Signs that are in the dataset:

- **age** (age);
- **job** (field of employment);
- **marital** (marital status);
- **education** (level of education);
- **default** (is there an overdue loan);
- **housing** (is there a loan for housing);
- **loan** (is there a loan for personal needs);
- **balance** (balance).

 ### Data related to the last contact in the context of the current marketing campaign: 

- **contact** (the type of contact with the client);
- **month** (the month in which the last contact was made);
- **day** (the day on which the last contact was made);
- **duration** (contact duration in seconds).

 ### Other signs:

- **campaign** (number of contacts with this client during the current campaign);
- **pdays** (the number of missed days from the last marketing campaign to the contact in the current campaign);
- **previous** (the number of contacts before the current campaign)
poutcome (the result of the last marketing campaign).

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; And, of course, our target variable *deposit*, which determines whether the client agrees to open a deposit in the bank. This is what we will try to predict in this case.

<div align="center"> <h2 align="center">  CONCLUSIONS </h2> </div>

1. <div align="justify"> Four optimization methods were used to find the optimal values of the model parameters: GRIDSEARCHCV, RANDOMIZEDSEARCHCV, HYPEROPT (Tree-Structured Parzen Estimators algorithm) and Optuna. </div>
2. <div align="justify">To reduce the optimization time, the ranges and the number of values of categorical parameters were reduced (for example, the method of finding the maximum of the likelihood function in logistic regression)</div>
3. <div align="justify">On all types of optimization, higher scores (f1-world) are observed for Random Forest than for logistic regression</div>
4. <div align="justify">The classical GridSearchCV method showed the longest operating time even with relatively small ranges of model parameters </div>

<div align="center"> <h2 align="center">  GLOBAL CONCLUSIONS </h2> </div>

1. <div align="justify"> The stages of constructing a model for classifying real bank customers into those who will open a deposit and those who will not. </div>
2. <div align="justify">The initial data processing was carried out, within the framework of which omissions and outliers were processed. </div>
3. <div align="justify">An Intelligence analysis of the data (EDA) was carried out. As a result, the first patterns were discovered and hypotheses about dependencies in the data were put forward. </div>
4. <div align="justify">The selection and transformation of features has been carried out. The data was recoded and transformed so that it could be used in solving the classification problem. </div>
5. <div align="justify"> The classification problem was solved by the methods of logistic regression and decision trees. </div>
6. <div align="justify"> The classification problem was solved using ensembles of models (gradient boosting and stacking) with the search for optimal parameters using optuna and grid search.</div>
7. <div align="justify"> The best classification metrics were obtained: f1 and Accuracy on the test = 0.82 and 0.83, respectively. </div>

