# Santander_Customer_Transaction_Prediction

### Background
At Santander , mission is to help people and businesses prosper. We are always looking for ways to help our customers understand their financial health and identify which products and services might help them achieve their monetary goals.
Our aim is to find customer satisfaction and will a customer buy this product? Can a customer pay this loan?

### Problem Statement
In this challenge, we need to identify which customers will make a specific transaction in the future, irrespective of the amount of money transacted.

#### Number of attributes:
We have an anonymized dataset containing numeric feature variables, the binary target column, and a string ID column. The task is to predict the value of target column in the test dataset. 

We will start with general Exploratory Data Analysis by checking for the Missing values if any, checking for the outliers, feature scaling and feature seection. Once data preprocessing is done we will start with model training, we will do parameter tuning and later proceed with validation metrics such as precision, recall and area under curve (AUC). Post that we will select best model based on various metrics mentioned above and finalize the model.

Since we have imbalanced dataset we need to proceed with either up sampling or down sampling. We will check with both the ways and will use Synthetic Minority Over Sampling Technique for up sampling by adding synthetic observations to balance the dataset, we will use strtified sampling to down sample and see how our model predicts. Below is the metric comparison among various machine learning model.

In the modelling part we will be using below 3 algorithms:
1. Logistic Regression
2. Random Forest Classifier
3. Naive Bayyes Classifier
4. Extreme Gradient Boosting

We will validate the model by checking its accuracy score, precision, recall and ROC-AUC curve and select the best performing model on the data.
