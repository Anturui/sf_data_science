<div align="center"> <h1 align="center"> Analysis of hotel ratings data from the website booking.com </h1> </div>


Case study: creating the first model using machine learning algorithms.

RESULTS OF WORK ON THIS PROJECT:
1. The first model based on machine learning algorithms has been created
2. Took part in a competition on Kaggle (score: 11.51996)
3. Gained experience in preparing data to improve the machine learning model

<div align="center"> <h2 align="center"> Problem description </h2> </div>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; One of the ***booking.com*** company's problems  — these are dishonest hotels that wind up their ratings. One of the ways to detect such hotels is to build a model that predicts the rating of the hotel. If the model's predictions are very different from the actual result, then perhaps the hotel is behaving dishonestly, and it is necessary to conduct an additional check of the correctness of the lined estimate.

<div align="center"> <h2 align="center"> Task </h2> </div>
<div align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; There is a dataset that contains information about 515,000 reviews of hotels in Europe. The machine learning model should predict the rating of the hotel according to the website Booking.com based on the data available in the dataset. Intelligence analysis skills will help improve the model.</div>

### [Link to the dataset](https://drive.google.com/file/d/1Qj0iYEbD64eVAaaBylJeIi3qvMzxf2C_/view?usp=sharing )

<div align="center"> <h2 align="center"> Description of the dataset </h2> </div>
<div align="center"> <h3 align="center"> Signs that are in the dataset </h3> </div>

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

