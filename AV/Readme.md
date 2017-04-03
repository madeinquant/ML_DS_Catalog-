This is the work done as part of the Analytics Vidhya Rampaging Datahulk Competition.
I secured 3rd place in the competition.


Below  is the problem statement followed by the approach that I have used -



Problem Statement

Congratulations! you have been hired as a Chief Data Scientist by “QuickMoney”, a fast growing Hedge fund. They rely on automated systems to carry out trades in the stock market at inter-day frequencies. They wish to create a machine learning-based strategy for predicting the movement in stock prices.

They ask you to create a trading strategy to maximize their profits in the stock market. Stock markets are known to have high degree of unpredictability, but you have an opportunity to beat the odds and create a system which will outperform others.

The task for this challenge is to predict probability whether the price for a particular stock at the tomorrow’s market close will be higher(1) or lower(0) compared to the price at today’s market close.


Important Points:

    Information derived from the use of information from future is not permitted. For example, If you're predicting for timestamp x, you must not use features from timestamp x+1 or any timestamp after that
    Anyone found using such features will be disqualified from the hackathon


Data
Variable 	        Definition
ID 	                Unique ID for each observation
Timestamp 	        Unique value representing one day
Stock_ID 	        Unique ID representing one stock
Volume 	        Normalized values of volume traded of given stock ID on that timestamp
Three_Day_Moving_Average 	Normalized values of three days moving average of Closing price for given stock ID (Including Current day)
Five_Day_Moving_Average 	Normalized values of five days moving average of Closing price for given stock ID (Including Current day)
Ten_Day_Moving_Average 	Normalized values of ten days moving average of Closing price for given stock ID (Including Current day)
Twenty_Day_Moving_Average 	Normalized values of twenty days moving average of Closing price for given stock ID (Including Current day)
True_Range 	Normalized values of true range for given stock ID
Average_True_Range 	Normalized values of average true range for given stock ID
Positive_Directional_Movement 	Normalized values of positive directional movement for given stock ID
Negative_Directional_Movement 	Normalized values of negative directional movement for given stock ID
Outcome 	Binary outcome variable representing whether price for one particular stock at the tomorrow’s market close is higher(1) or lower(0) compared to the price at today’s market close



Evaluation Metric is log-loss


Solution Approach -

- Build as many features as possible from the existing features mostly by taking two variables at a time and computing their difference and sums.

- Rum gradient boosting.


- Details in the ipython notebook

    
