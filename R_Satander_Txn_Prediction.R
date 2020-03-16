
#clearing environment
rm(list = ls())

#importing required libraries

library(caret) #done
library(Matrix) #done
library(randomForest)#done
library(mlr) #done
library(rBayesianOptimization)#done
library(lightgbm)
library(pROC)#done
library(DMwR)#done
library(ROSE)#done
library(yardstick)#done
library(ggplot2)
library(RRF)
library(e1071)

#setting working directory
setwd("D:/Edwisor Data Science/Final Project 2 Submission")

#verifying directory
getwd()

#importing train dataset
df_train = read.csv("train.csv", header = TRUE, na.strings = c(" ","","NA"))

#checking the train data
str(df_train)

#changing data type of target variable
df_train$target = as.factor(df_train$target)

#importing test dataset
df_test = read.csv("test.csv", header = TRUE, na.strings = c(" ","","NA"))

#checking the test data
str(df_test)

#Viewing target variable data distribution in train dataset
table(df_train$target)
# 179902 -> 0 
# 20098  -> 1

table(df_train$target) / length(df_train$target) * 100
#SO APPROXIMATELY 90% PEOPLE WILL NOT TRANSACT AND APPROXIMATELY 10% PEOPLE WILL TRANSACT
#0 -> 89.951 %
#1 -> 10.049 %

#plotting bar graph for the same
plot1 = ggplot(df_train, aes(target))+theme_bw() + geom_bar(stat = 'count', fill = 'red')
plot1

#Missing Value Analysis:-
#Finding the missing values in train data
missing_val = data.frame(missing_val=apply(df_train,2,function(x){sum(is.na(x))}))
missing_val$variable_name = row.names(missing_val)
missing_val$missing_val = (missing_val$missing_val / nrow(df_train)) * 100
missing_val = missing_val[,c(2,1)]

#TRAIN DATASET DOES NOT INCLUDE ANY MISSING VALUES SO NO NEED OF IMPUTING VALUES

#Finding the missing values in test data
missing_val_test = data.frame(missing_val=apply(df_test,2,function(x){sum(is.na(x))}))
missing_val_test$variable_name = row.names(missing_val_test)
missing_val_test$missing_val = (missing_val_test$missing_val / nrow(df_test)) * 100
missing_val_test = missing_val_test[,c(2,1)]

#TEST DATASET DOES NOT INCLUDE ANY MISSING VALUES SO NO NEED OF IMPUTING VALUES

#Creating eqully distributed R Dataframe for training

df_1 = subset(df_train , df_train$target == 1)
df_0 = subset(df_train , df_train$target == 0)
df_bal_data = rbind(df_1, df_0)

#Split the training data using simple random sampling
train_index = sample(1:nrow(df_bal_data), 0.80*nrow(df_bal_data))

#train_index

#Training data
training_data<-df_bal_data[train_index,]

table(training_data$target)

#Test data
testing_data<-df_bal_data[-train_index,]

table(testing_data$target)

### Random Forest Algorithm ###

#fitting the random forest
rf_model = randomForest(training_data$target~.,training_data[,3:202],ntree=100,importance=TRUE)

#Predicting Test Data set
RF_predictions = predict(rf_model, testing_data[,3:202])

#Evaluate performance via confusion Matrix
CM_RF = table(testing_data$target, RF_predictions)

print(CM_RF)

#Calculating Accuracy ((TP + TN) * 100) / (TP + TN + FP + FN)
((3016 + 3015) * 100) / (3016 + 3015 + 1051 + 958)
#75.01244

#Accuracy for Random Forest 75.01

#Calculating Precision for Random Forest Model Predicions (TP / TP + FP)
(3015) / (3015 + 1051) 
#0.741

#Calculating Recall for Random Forest Model Predicions (TP / TP + FN)
(3015) / (3015 + 958) 
#0.758

#F1 = 2 * ((Precision * Recall)/(Precision + Recall))
F1_RF = 2 * (0.741 * 0.758) / (0.741 + 0.758)
print(F1_RF)
#0.749

table(testing_data$target)
#0 -> 4067
#1 -> 3973

###################### Run 2 Selecting Variables with p value < 0.05 #########################

dnames = names(training_data) %in% c('var_10','var_14','var_16','var_17','var_27','var_29','var_30','var_38','var_39','var_41','var_42','var_46','var_73',
                                     'var_79','var_96','var_98','var_100','var_103','var_117','var_124','var_126','var_129','var_136','var_153','var_158',
                                     'var_160','var_183','var_185')

training_data_v1 = training_data[!dnames]

testing_data_v1 = testing_data[!dnames]

str(training_data_v1)

#fitting the random forest
rf_model_v2 = randomForest(training_data_v1$target~.,training_data_v1[,3:174],ntree=100,importance=TRUE)

#Predicting Test Data set
RF_predictions_v2 = predict(rf_model_v2, testing_data_v1[,3:174])

#Evaluate performance via confusion Matrix
CM_RF_v2 = table(testing_data_v1$target, RF_predictions_v2)

print(CM_RF_v2)

#Calculating Accuracy ((TP + TN) * 100) / (TP + TN + FP + FN)
((3031 + 2961) * 100) / (3031 + 2961 + 1106 + 942)

#Accuracy for Random Forest 74.52

#Calculating Precision for Random Forest Model Predicions (TP / TP + TN)
(3031) / (3031 + 1106) 
#0.732

#Calculating Recall for Random Forest Model Predicions (TP / TP + FN)
(3031) / (3031 + 942) 
#0.762

#F1 = 2 * ((Precision * Recall)/(Precision + Recall))
F1_RF_v2 = 2 * (0.732 * 0.762) / (0.732 + 0.762)
print(F1_RF_v2)
#0.746

### Predicting dependent variables on our Actual Test Set  ###

tnames = names(df_test) %in% c('var_10','var_14','var_16','var_17','var_27','var_29','var_30','var_38','var_39','var_41','var_42','var_46','var_73',
                               'var_79','var_96','var_98','var_100','var_103','var_117','var_124','var_126','var_129','var_136','var_153','var_158',
                               'var_160','var_183','var_185')


#Predicting dependent variable for actual TEST dataset using Random Forest
df_test_v1 = df_test[!tnames]

df_test_v1$Target_Prdicted = predict(rf_model_v2, df_test_v1[,2:173])

table(df_test_v1$Target_Prdicted)

#Writing Output file
RF_output = data.frame(ID = df_test_v1$ID_code, Target_Prdicted = df_test_v1$Target_Prdicted)
write.csv(RF_output, "RF_Output_R.csv", index = FALSE)

#ROC_AUC score and curve
set.seed(420)
num.samples = 100

plot(x=testing_data_v1$target, y=RF_predictions_v2)

glm.fit = glm(testing_data_v1$target ~ RF_predictions_v2, family = binomial)

lines(testing_data_v1$target, glm.fit$fitted.values)

#to remove whitespaces around the graph
par(pty = "s")

#plotting AUC ROC graph
roc(testing_data_v1$target, glm.fit$fitted.values, plot = TRUE, legacy.axes = TRUE,
    percent = TRUE, xlab = 'Actual Target', ylab = 'Predicted Target', col = "#377eb8", lwd=3)


### Logistic Regression ###

Logit_model = glm(training_data$target~.,training_data[,3:202], family = "binomial")

#Summary of Logistic Regression Model 
summary(Logit_model)

#Predictions using Logistic Regression
logit_predictions = predict(Logit_model, newdata = testing_data, type = "response")

#convert probabilities
logit_predictions = ifelse(logit_predictions > 0.05, 1, 0)

#Evaluate performance via confusion Matrix
CM_LM = table(testing_data$target, logit_predictions)

#Calculating Accuracy ((TP + TN) * 100) / (TP + TN + FP + FN)
((21988 + 3516) * 100) / (21988 + 3516 + 14037 + 459)

#Accuracy for Logistic Regression 63.76

#Calculating Precision for Naive Bayes Model Predicions (TP / TP + TN)
(3516) / (3516 + 21988) 
#0.13

#Calculating Recall for Naive Bayes Model Predicions (TP / TP + FN)
(3516) / (3516 + 459) 
#0.88

#F1 = 2 * ((Precision * Recall)/(Precision + Recall))
F1_LR = 2 * (0.13 * 0.88) / (0.13 + 0.88)
print(F1_LR)
#0.226

###################### Run 2 Selecting Variables with p value > 0.05 #########################

Logit_model_v2 = glm(training_data_v1$target~.,training_data_v1[,3:174], family = "binomial")

#Summary of Logistic Regression Model 
summary(Logit_model_v2)

#Predictions using Logistic Regression
logit_predictions_v2 = predict(Logit_model_v2, newdata = testing_data_v1, type = "response")

#convert probabilities
logit_predictions_v2 = ifelse(logit_predictions_v2 > 0.5, 1, 0)

#Evaluate performance via confusion Matrix
CM_LM_v2 = table(testing_data_v1$target, logit_predictions_v2)

print(CM_LM_v2)

#Calculating Accuracy ((TP + TN) * 100) / (TP + TN + FP + FN)
((3091 + 3172) * 100) / (3091 + 3172 + 895 + 882)

#Accuracy for Logistic Regression 77.89

#Calculating Precision for Naive Bayes Model Predicions (TP / TP + TN)
(3091) / (3091 + 895) 
#0.775

#Calculating Recall for Naive Bayes Model Predicions (TP / TP + FN)
(3091) / (3091 + 882) 
#0.778

#F1 = 2 * ((Precision * Recall)/(Precision + Recall))
F1_LR_v2 = 2 * (0.775 * 0.778) / (0.775 + 0.778)
print(F1_LR_v2)
#0.776

#Predicting dependent variable for actual TEST dataset using Logistic Regression
df_test_v1$Target_Prdicted_LR = predict(Logit_model_v2, df_test_v1[,2:173], type = "response")

df_test_v1$Target_Prdicted_LR = ifelse(df_test_v1$Target_Prdicted_LR > 0.5, 1, 0)

table(df_test_v1$Target_Prdicted_LR)

#Writing Output file
lr_output = data.frame(ID = df_test_v1$ID_code, Target_Prdicted = df_test_v1$Target_Prdicted_LR)
write.csv(lr_output, "LR_Output_R.csv")

#ROC_AUC score and curve
set.seed(420)
num.samples = 100

plot(x=testing_data_v1$target, y=logit_predictions_v2)

glm.fit = glm(testing_data_v1$target ~ logit_predictions_v2, family = binomial)

lines(testing_data_v1$target, glm.fit$fitted.values)

#to remove whitespaces around the graph
par(pty = "s")

#plotting AUC ROC graph
roc(testing_data_v1$target, glm.fit$fitted.values, plot = TRUE, legacy.axes = TRUE,
    percent = TRUE, xlab = 'Actual Target', ylab = 'Predicted Target', col = "#377eb8", lwd=3)


### Naive Bayes ###

#Develop Naive Bayes Model
NB_model = naiveBayes(training_data$target~., data = training_data[,3:202])

#predict on test cases
NB_predictions = predict(NB_model, testing_data[,3:202], type = 'class')

#Evaluate performance via confusion Matrix
CM_NB = table(testing_data$target, NB_predictions)

#Calculating Accuracy ((TP + TN) * 100) / (TP + TN + FP + FN)
((1452 + 35412) * 100) / (1452 + 35412 + 613 + 2523)

#Accuracy for Naive Bayes 92.16

#Calculating Precision for Naive Bayes Model Predicions (TP / TP + TN)
(1452) / (1452 + 35412) 
#0.039

#Calculating Recall for Naive Bayes Model Predicions (TP / TP + TN)
(1452) / (1452 + 2523) 
#0.36

#F1 = 2 * ((Precision * Recall)/(Precision + Recall))
F1_NB = 2 * (0.039 * 0.36) / (0.039 + 0.36)
print(F1_NB)
#0.0703


###################### Run 2 Selecting Variables with p value > 0.05 #########################


NB_model_v2 = naiveBayes(training_data_v1$target~., data = training_data_v1[,3:174])

#predict on test cases
NB_predictions_v2 = predict(NB_model_v2, testing_data_v1[,3:174], type = 'class')

#Evaluate performance via confusion Matrix
CM_NB_v2 = table(testing_data_v1$target, NB_predictions_v2)

print(CM_NB_v2)

#Calculating Accuracy ((TP + TN) * 100) / (TP + TN + FP + FN)
((3178 + 3325) * 100) / (3178 + 3325 + 742 +  795)

#Accuracy for Naive Bayes 80.88

#Calculating Precision for Naive Bayes Model Predicions (TP / TP + TN)
(3178) / (3178 + 742) 
#0.810

#Calculating Recall for Naive Bayes Model Predicions (TP / TP + FN)
(3178) / (3178 + 795) 
#0.799

#F1 = 2 * ((Precision * Recall)/(Precision + Recall))
F1_NB = 2 * (0.810 * 0.799) / (0.810 + 0.799)
print(F1_NB)
#0.8044

#Predicting dependent variable for actual TEST dataset using Naive Bayes
df_test_v1$Target_Prdicted_NB = predict(NB_model_v2, df_test_v1[,2:173], type = 'class')

table(df_test_v1$Target_Prdicted_NB)

#Writing Output file
df_nb_output = data.frame(ID = df_test_v1$ID_code, Target_Prdicted = df_test_v1$Target_Prdicted_NB)
write.csv(df_nb_output, "NB_Output_R.csv")

#ROC_AUC score and curve
set.seed(420)
num.samples = 100

plot(x=testing_data_v1$target, y=NB_predictions_v2)

glm.fit = glm(testing_data_v1$target ~ NB_predictions_v2, family = binomial)

lines(testing_data_v1$target, glm.fit$fitted.values)

#to remove whitespaces around the graph
par(pty = "s")

#plotting AUC ROC graph
roc(testing_data_v1$target, glm.fit$fitted.values, plot = TRUE, legacy.axes = TRUE,
    percent = TRUE, xlab = 'Actual Target', ylab = 'Predicted Target', col = "#377eb8", lwd=3)


