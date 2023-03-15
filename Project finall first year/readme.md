<div align="center"> <h1 align="center"> Задача предсказания отношения количества репостов к просмотрам для постов телеграмм с целью выявления лучших постов и размещения их на канале.
 </h1> </div>
<div align="center"> <h3 align="center"> by Aleksey Kolychev </h3> </div>

<div align="center"> <h3 align="center">Предыстория</h3> </div>

<div align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Девушка завела телеграмм канал и мне стало интересно, отчего зависит популярность канала. Я предположил, что от контента. Тогда сразу вопрос, какой должен быть контент, чтобы канал имел стабильную аудиторию, которой можно продавать рекламу. Тогда и родился этот проект. </div>
<div align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Целью было создать модель машинного обучения, предсказывающую основной критерий хорошего сообщения для каналов в жанре «Познавательное». Причем, пост – это либо гиф или видео с небольшим текстом Основным критерием был выбрано отношение количества репостов к количеству просмотров. Это, по моему мнению, означает, что пост настолько хорош, что его можно переслать друзьям или родственникам, поделиться им с другими. </div>

<div align="center"> <h3 align="center"> Проект состоит из нескольких частей:</h3> </div>
1)	Сбор информации о сообщениях с помощью библиотеки telethone
2)	Подготовка датасета к ML, 
3)	Обучение модели ML
4)	Написание бота, которые по присланному ему сообщению (с GIF или с video) выдает прогноз репостовости
5)	EDA для попытки объяснить полученные результаты


RESULTS OF WORK ON THIS PROJECT:
1. The first model based on machine learning algorithms has been created
2. Took part in a competition on Kaggle (score: 11.51996)
3. Gained experience in preparing data to improve the machine learning model

<div align="center"> <h2 align="center"> 1.	Сбор информации о сообщениях с помощью библиотеки telethone </h2> </div>
<div align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Банки хранят огромные объёмы информации о своих клиентах. Эти данные можно использовать для того, чтобы оставаться на связи с клиентами и индивидуально ориентировать их на подходящие именно им продукты или банковские предложения. </div>

<div align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Обычно с выбранными клиентами связываются напрямую через разные каналы связи: лично (например, при визите в банк), по телефону, по электронной почте, в мессенджерах и так далее. Этот вид маркетинга называется прямым маркетингом. На самом деле, прямой маркетинг используется для взаимодействия с клиентами в большинстве банков и страховых компаний. Но, разумеется, проведение маркетинговых кампаний и взаимодействие с клиентами — это трудозатратно и дорого.</div>

<div align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Банкам хотелось бы уметь выбирать среди своих клиентов именно тех, которые с наибольшей вероятностью воспользуются тем или иным предложением, и связываться именно с ними </div>

Проект будет состоит из пяти частей:
1. Первичная обработка данных
2. Разведывательный анализ данных (EDA)
3. Отбор и преобразование признаков
4. Решение задачи классификации: логистическая регрессия и решающие деревья
5. Решение задачи классификации: ансамбли моделей 


<div align="center"> <h2 align="center"> Датасет </h2> </div>
<div align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; There is a dataset that contains information about 515,000 reviews of hotels in Europe. The machine learning model should predict the rating of the hotel according to the website Booking.com based on the data available in the dataset. Intelligence analysis skills will help improve the model.</div>

### [Link to the dataset](https://drive.google.com/file/d/1Qj0iYEbD64eVAaaBylJeIi3qvMzxf2C_/view?usp=sharing )

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

И, разумеется, наша целевая переменная **deposit**, которая определяет, согласится ли клиент открыть депозит в банке. Именно её мы будем пытаться предсказать в данном кейсе.



**hotel_address** — the address of the hotel;

**review_date** — the date when the reviewer posted the corresponding review;

**average_score** — the average score of the hotel calculated based on the last comment for the last year;

**hotel_name** — the name of the hotel;

**reviewer_nationality** — reviewer's country;

**negative_review** — negative review that the reviewer gave to the hotel;

**review_total_negative_word_counts** — the total number of words in a negative review;

**positive_review**— positive review that the reviewer gave to the hotel;

**review_total_positive_word_counts** — the total number of words in a positive review;

**reviewer_score** — the rating that the reviewer gave to the hotel based on his experience;

**total_number_of_reviews_reviewer_has_given** — the number of reviews that reviewers have given in the past;

**total_number_of_reviews** — total number of valid hotel reviews;

**tags** — tags that the reviewer gave to the hotel;

**days_since_review** — the number of days between the verification date and the cleaning date;

**additional_number_of_scoring** — there are also some guests who just rated the service, but did not leave a review. This number indicates how many valid estimates there are without verification.

**lat** — geographical latitude of the hotel;

**lng** is the geographical longitude of the hotel.

<div align="center"> <h2 align="center"> Stages of work on the project </h2> </div>
<div align="justify"> </div>
1. SPLITTING THE DATA SET
<div align="justify">  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; First of all, to create a model, it was necessary to divide the dataframe into a data set that was used to train the model, called 'X', and into a target variable, the value of which we will predict, 'y' (in our case, this is the rating of hotels).
Further, each of the obtained sets of scores is divided into training (train, used to train the model) and test (test, used to evaluate the accuracy of the model). This division was carried out using a special method train_test_split() of the sklearn library. In the method parameters (the test_size parameter), we specify which part of the original dataframe should be left for testing the model. In our code, this part is 20%, or 0.2. </div>  
<div align="justify"> </div>
2. CREATING AND TRAINING A MODEL

<div align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The process of creating and testing a model takes only four lines of code. The popular and rather powerful RandomForestRegressor algorithm was used as an algorithm. It is implemented in the sklearn library. </div>
<div align="justify"> </div>
3. MODEL QUALITY ASSESSMENT
<div align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; To assess the quality of the model — the accuracy of the predictions made by the model — we will use a metric (some numerical indicator) called MAPE (mean absolute percentage error), the average absolute percentage error. This metric is very easy to interpret. </div>


<div align="center"><h1 align="center"> $$MAPE = \frac{{1}}{n} \sum\limits_{i=1}^n\frac{y_{true_i} - y_{pred_i}}{y_{true_i}} * 100\% $$ </h1></div>

where $y_{true_i}$ are the actual values of the forecast, and $y_{pred_i}$ are the predicted values.

<div align="center"><h3 align="center"> CONCLUSIONS: </h3></div>

1. My first machine learning case was created: the dataset was cleaned, several new features were extracted from it and data for model training was prepared.
2. I took part in a machine learning competition on [Kaggle](https://www.kaggle.com/competitions/sf-booking/leaderboard).
3. Received MAPE: 11.51996

