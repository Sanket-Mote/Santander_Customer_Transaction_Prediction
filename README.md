# Santander_Customer_Transaction_Prediction

### Background
At Santander , mission is to help people and businesses prosper. We are always looking for ways to help our customers understand their financial health and identify which products and services might help them achieve their monetary goals.
Our aim is to find customer satisfaction and will a customer buy this product? Can a customer pay this loan?

### Problem Statement
In this challenge, we need to identify which customers will make a specific transaction in the future, irrespective of the amount of money transacted.

### Dataset is available on Kaggle website. 
[Click here to go to Kaggle and Download Dataset](https://www.kaggle.com/c/santander-customer-transaction-prediction/data)

#### Number of attributes:
We have an anonymized dataset containing numeric feature variables, the binary target column, and a string ID column. The task is to predict the value of target column in the test dataset. 

#### General Process ffollowed:
We will start with general Exploratory Data Analysis by checking for the Missing values if any, checking for the outliers, feature scaling and feature seection. Once data preprocessing is done we will start with model training, we will do parameter tuning and later proceed with validation metrics such as precision, recall and area under curve (AUC). Post that we will select best model based on various metrics mentioned above and finalize the model.

Since we have imbalanced dataset we need to proceed with either up sampling or down sampling. We will check with both the ways and will use Synthetic Minority Over Sampling Technique for up sampling by adding synthetic observations to balance the dataset, we will use strtified sampling to down sample and see how our model predicts. Below is the metric comparison among various machine learning model.

In the modelling part we will be using below 3 algorithms:
1. Logistic Regression
2. Random Forest Classifier
3. Naive Bayyes Classifier
4. Extreme Gradient Boosting

## Results of Up-Sampling (Using Synthetic Minority Over-sampling Technique (SMOTE))
|Model | Precision | Recall |F1 Score | Accuracy* |
|---|---|---|---|---|
|Logistic Regression | 26% | 76% | 38% | 75% |
|Extreme Gradient Boosting (XGB) | 27% | 30% | 28% | 85% |
|Random Forest Classifier | 32% | 13% | 18% | 89% |
|Naive Bayyes Classifier | 22% | 47% | 30% | 77% |

## Results of Down-Sampling (Equal distribution 50-50 split)
|Model | Precision | Recall |F1 Score | Accuracy* |
|---|---|---|---|---|
|Logistic Regression | 78% | 78% | 78% | 78% |
|Extreme Gradient Boosting (XGB) | 79% | 76% | 77% | 78% |
|Random Forest Classifier | 80% | 76% | 78% | 78% |
|Naive Bayyes Classifier | 81% | 80% | 81% | 81% |

As we can see on the provided dataset Down-Sampling works well. Based on various metrics Naive Bayes is performing well and we will finalise the model.
