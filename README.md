# Santander_Customer_Transaction_Prediction
Santander group is a Spanish multinational commercial bank and financial services company. We will be working on the anonymous dataset publicly available on Kaggle Website. It is a classification problem where we need to predict whether the customer transaction is valid or not. We can see that we have around 200 features which are anonymous and we have a Target variable which needs to be predicted. In our training dataset we have around 2 lakhs observation and 2 lakhs observation in test dataset. We have identified that we have class imbalance probem in our dataset as in training dataset we have 90% genuine transactions and 10% not genuine transactions.

We will start with general Exploratory Data Analysis by checking for the Missing values if any, checking for the outliers, feature scaling and feature seection.

Since we have imbalanced dataset we need to proceed with either up sampling or down sampling. We will check with both the ways and will use Synthetic Minority Over Sampling Technique for up sampling by adding synthetic observations to balance the dataset, we will use strtified sampling to down sample and see how our model predicts.

In the modelling part we will be using below 3 algorithms:
1. Logistic Regression
2. Random Forest Classifier
3. Naive Bayyes Classifier
4. Extreme Gradient Boosting

We will validate the model by checking its accuracy score, precision, recall and ROC-AUC curve and select the best performing model on the data.
