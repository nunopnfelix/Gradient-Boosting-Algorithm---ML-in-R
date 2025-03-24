rm(list=ls())

setwd("D://")

#install.packages("xgboost")
library(xgboost)
library(caTools)
library(dplyr)
#install.packages("caret")
library(caret)
library(readr)
library(stringr)
library(car)

#########################  Dataset Loading  ####################################

DATA <- read.csv("aaa.csv")
DATA = na.omit(DATA)

DATA$B <- ifelse(DATA$A > 0.5,1,0)

#Preparation

sample_split <- sample.split(Y = DATA$B, SplitRatio = 0.7)
train_set <- subset(x = DATA, sample_split == TRUE)
test_set <- subset(x = DATA, sample_split == FALSE)

labels <- (train_set$B)
ts_label <- (test_set$B)


y_train <- as.integer(labels)
y_test  <- as.integer(labels)
X_train <- train_set %>% select(-B)
X_test  <- test_set %>% select(-B)

num.class = length(levels(y_train))
unique(y_train)

#Predictive Modeling with R XGBoost

xgb_train <- xgb.DMatrix(data = as.matrix(X_train), label = labels)
xgb_test <- xgb.DMatrix(data = as.matrix(X_test), label = ts_label)

xgb_params <- list(
  booster = "gbtree",
  eta = 0.01,
  max_depth = 8,
  gamma = 4,
  subsample = 0.75,
  colsample_bytree = 1,
  objective = "binary:logistic",
  eval_metric = "mlogloss",
  num_class = length(levels(DATA$B)))


xgb_model <- xgb.train(
  params = xgb_params,
  data = xgb_train,
  nrounds = 5000,
  verbose = 1)

xgb_model

#XGBoost Feature Importance

importance_matrix <- xgb.importance(
  feature_names = colnames(xgb_train), 
  model = xgb_model)

importance_matrix

xgb.plot.importance(importance_matrix)


#Predictions

xgb_preds <- predict(xgb_model, 
                     as.matrix(X_test), 
                     reshape = TRUE,
                     na.action = na.pass)

xgb_preds <- ifelse(xgb_preds>0.5,1,0)
xgb_preds <- as.data.frame(xgb_preds)

xgb_preds

#Evaluations and Models Accuracy

accuracy <- sum(xgb_preds)/ nrow(xgb_preds)
accuracy

#

class(xgb_preds)
class(test_set$Goals)
levels(xgb_preds)
levels(ts_label)

cm <- confusionMatrix(as.factor(xgb_preds$xgb_preds),
                      as.factor(test_set$Goals))

cm
