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

<div align="center"> <h2 align="center">  CONCLUSIONS </h2> </div>

1. <div align="justify"> Four optimization methods were used to find the optimal values of the model parameters: GRIDSEARCHCV, RANDOMIZEDSEARCHCV, HYPEROPT (Tree-Structured Parzen Estimators algorithm) and Optuna. </div>
2. <div align="justify">To reduce the optimization time, the ranges and the number of values of categorical parameters were reduced (for example, the method of finding the maximum of the likelihood function in logistic regression)</div>
3. <div align="justify">On all types of optimization, higher scores (f1-world) are observed for Random Forest than for logistic regression</div>
4. <div align="justify">The classical GridSearchCV method showed the longest operating time even with relatively small ranges of model parameters </div>

<div align="center"> <h2 align="center"> Description of the dataset </h2> </div>
<div align="center"> <h3 align="center"> Signs that are in the dataset </h3> </div>

**age** (возраст);
**job** (сфера занятости);
**marital** (семейное положение);
**education** (уровень образования);
**default** (имеется ли просроченный кредит);
**housing** (имеется ли кредит на жильё);
**loan** (имеется ли кредит на личные нужды);
**balance** (баланс).

## Данные, связанные с последним контактом в контексте текущей маркетинговой кампании:

**contact** (тип контакта с клиентом);
**month** (месяц, в котором был последний контакт);
**day** (день, в который был последний контакт);
**duration** (продолжительность контакта в секундах).

## Прочие признаки:

**campaign** (количество контактов с этим клиентом в течение текущей кампании);
**pdays** (количество пропущенных дней с момента последней маркетинговой кампании до контакта в текущей кампании);
**previous** (количество контактов до текущей кампании)
poutcome (результат прошлой маркетинговой кампании).

<div align="justify"> И, разумеется, наша целевая переменная **deposit**, которая определяет, согласится ли клиент открыть депозит в банке. Именно её мы будем пытаться предсказать в данном кейсе. </div>