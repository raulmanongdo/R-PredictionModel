#***********************************************************************************************
#  R  Program : kc_client_churn_FINAL_G.R
#
#  Author     : Raul Manongdo, University of Technology Sydney
#               Advance Analytics Institute
#  Date       : June 2017

#  Program description:
#     Birary client churn prediction modelling using 3 candidate models;
#        Logistic Regression, Random Forest and C5.0 Decison Trees.
#     Feature selection using GLM + RF models - Pearson correlation analysis.
#     Train/Test as separate datasets.
#     Threshold for prediction probability initialy set at default value of .5 and
#         tweak to mean of probability of churn class
#     Model comparison by area under ROC and test of signifance using Delong method
#     Model selection by Overall prediction Accuracy scores.
#     10 fold X validation is 90%/10% training and testing split
#     10 fold X validation model comparison measure is mean prediction accuracy and AUC
# ***********************************************************************************************

library(lattice)
library(randomForest, quietly = TRUE)
library(gplots, quietly = TRUE)
library(ROCR)
library(plyr)
library(C50)
library (pROC)
library(partykit)
library(tidyverse)
library(corrplot)
library(stargazer)
# C5.0 plot


#**************************
#returns model performance measures
#**************************
computePerformanceMeasures <- function(confusion_maxtix, b) {
  tn <- confusion_maxtix[1, 1]
  fp <- confusion_maxtix[1, 2]
  fn <- confusion_maxtix[2, 1]
  tp <- confusion_maxtix[2, 2]
  acc <- (tp + tn) / (tp + tn + fp + fn)
  prec <- (tp / (tp + fp))
  rec  <- (tp / (tp + fn))
  b <- ifelse(is.null(b), 1, b)
  fScore <- (1 + b ^ 2) * prec * rec / ((b ^ 2 * prec) + rec)
  return(c(acc, prec, rec, fScore))
}

selectRawDataVariables <- function(dataset) {
  # Remove unique key values <= 1
  dataset <-
    dataset[-which(names(dataset) == 'myUniqueClientID')]
  
  # Remove attributes that have more than 50% missing values
  n50pcnt <- round(nrow(dataset) * .50)
  x <- sapply(dataset, function(x)
    sum(is.na(x)) + sum(is.nan(x)))
  drops <- x[x > n50pcnt]
  dataset <- dataset[, !(names(dataset) %in% names(drops))]
  print("Attributes with more than 50% missing values")
  drops <- cbind(drops, drops / nrow(dataset))
  print(drops)
  
  
  # Remove attributes with only 1 value
  x <-
    sapply(dataset, function(x)
      ifelse(is.numeric(x), max(x) - min(x), NA))
  drops <- x[x == 0 & !is.na(x)]
  dataset <- dataset[,!(names(dataset) %in% names(drops))]
  print("Attributes with only one value")
  print(names(drops))
  
  return(dataset)
}

#**************************
#returns a cleansed raw dataset
#**************************
cleanRawData <- function(dataset) {
  tmp <- sapply(dataset, function(x)
    is.factor(x))
  factor_vars <- names(tmp[tmp == TRUE])
  
  for (var in factor_vars) {
    #  Assign most used value to missing categorical attributes
    tmp <- table(dataset[[var]], useNA = "no")
    default_val <- names(tmp[which(tmp == max(tmp))])
    print(paste(var, default_val,  sum(is.na(dataset[[var]]) + sum(
      is.nan(dataset[[var]])
    ))))
    dataset[[var]][is.na(dataset[[var]]) |
                     is.nan(dataset[[var]])] <- default_val
  }
  
  # Convert binary outcome variable
  dataset$Label <- ifelse(dataset$Label == 'Churn', 1, 0)
  dataset$Label <- as.factor(dataset$Label)
  
  #Change the following attributes to ordered factors
  dataset$Grade <- factor(dataset$Grade, ordered = TRUE)
  dataset$MostUsedBillingGrade <-
    factor(dataset$MostUsedBillingGrade, ordered = TRUE)
  dataset$MostUsedPayGrade <-
    factor(dataset$MostUsedPayGrade, ordered = TRUE)
  
  tmp <- sapply(dataset, function(x)
    is.numeric(x))
  num_vars <- names(tmp[tmp == TRUE])
  
  # For numeric values, set NaN and negative values to NA : bad data
  for (var in num_vars)
    dataset[[var]] <-
    ifelse(is.nan(dataset[[var]]), NA, dataset[[var]])
  
  # Impute mean as for missing  numeric values
  for (var in num_vars)
    dataset[[var]][is.na(dataset[[var]])] <-
    mean(dataset[[var]], na.rm = TRUE)
  
  # Assign lower and upper quantile values to numeric outliers
  print('Assign lower and upper quantile values to numeric outliers')
  k_iqr_multiplier <- 3
  for (var in num_vars) {
    tmp <- quantile(dataset[[var]], na.rm = TRUE)
    iqr <- tmp[4] - tmp[2]
    extreme.upper <- tmp[4] + (iqr * k_iqr_multiplier)
    extreme.lower <- tmp[2] - (iqr * k_iqr_multiplier)
    extreme.lower <-  ifelse(extreme.lower < 0, 0, extreme.lower)
    
    print(paste(
      var,
      sum(dataset[[var]] >= extreme.upper),
      'extreme.upper',
      extreme.upper
    ))
    print(paste(
      var,
      sum(dataset[[var]]  < extreme.lower),
      'extreme.lower',
      extreme.lower
    ))
    
    dataset[[var]] <-
      ifelse(dataset[[var]] >= extreme.upper, extreme.upper, dataset[[var]])
    dataset[[var]] <-
      ifelse(dataset[[var]] <= extreme.lower, extreme.lower, dataset[[var]])
  }
  
  # Drop number columns with only one value
  x <-
    sapply(dataset[num_vars], function(x)
      ifelse((max(x) == min(x)), T, F))
  drops <- x[x == TRUE]
  dataset <- dataset[, !(names(dataset) %in% names(drops))]
  print('Drop number columns with only one value')
  print(names(drops))
  
  
  return(dataset)
}

#**************************
#returns a Z score normalised  dataset
#**************************
scaleNumbers <- function(dataset) {
  # Get numeric only variables
  numericVars <-
    sapply(dataset, function(x) {
      ifelse((is.numeric(x) & !is.factor(x)), T, F)
    })
  
  for (varName in names(numericVars[numericVars == TRUE])) {
    dataset[[varName]] <-
      scale(dataset[[varName]], center = TRUE, scale = TRUE)
  }
  
  return(dataset)
}
#**************************
#find the previous conditions in an RF tree
#**************************
prevCond <- function(tree, i) {
  if (i %in% tree$right_daughter) {
    id <- which(tree$right_daughter == i)
    cond <- paste(tree$split_var[id], ">", tree$split_point[id])
  }
  if (i %in% tree$left_daughter) {
    id <- which(tree$left_daughter == i)
    cond <- paste(tree$split_var[id], "<", tree$split_point[id])
  }
  
  return(list(cond = cond, id = id))
}

#remove spaces in a word
collapse <- function(x) {
  x <- sub(" ", "_", x)
  
  return(x)
}


#**************************
#return the rules of an RF tree
#**************************
getConds <- function(tree) {
  #store all conditions into a list
  conds <- list()
  #start by the terminal nodes and find previous conditions
  id.leafs <- which(tree$status == -1)
  j <- 0
  for (i in id.leafs) {
    j <- j + 1
    prevConds <- prevCond(tree, i)
    conds[[j]] <- prevConds$cond
    while (prevConds$id > 1) {
      prevConds <- prevCond(tree, prevConds$id)
      conds[[j]] <- paste(conds[[j]], " & ", prevConds$cond)
      if (prevConds$id == 1) {
        conds[[j]] <- paste(conds[[j]], " => ", tree$prediction[i])
        break()
      }
    }
    
  }
  return(conds)
}


# -----------------------------------------------------------------------------
# LOAD RAW DATA
setwd("/Users/raulmanongdo/Documents/R-KinCare-FINAL/")
raw.data.train  <-
  read.csv(
    "KinCareTraing.csv",
    header = TRUE,
    na.strings = c("", "NA", "<NA>"),
    stringsAsFactors = TRUE
  )
raw.data.test  <-
  read.csv(
    "KinCareTesting.csv",
    header = TRUE,
    na.strings = c("", "NA", "<NA>"),
    stringsAsFactors = TRUE
  )

raw.data.train$role <- 'Train'
raw.data.test$role <-  'Test'
raw.data <- rbind(raw.data.train, raw.data.test)
summary(raw.data)
# Data Cleansing
var_names <-  selectRawDataVariables(raw.data)
d1 <- cleanRawData(var_names)
print('Summary of Cleansed Raw Data')
summary(d1)

#=========================
#FEATURE SELECTION
ndxRole <- which(names(d1) == 'role')
ndxLabel <- which(names(d1) == 'Label')

d2 <- d1[, -ndxRole,-ndxLabel]
d2.scaled <- scaleNumbers(d2)

set.seed(1)
mod_d2_glm <-
  glm(Label ~ ., family = binomial(link = "logit"), data = d2.scaled)
mod_d2_glm.sumry <- summary(mod_d2_glm, signif.stars = TRUE)

sel_pValue <- .03
print(paste(
  'Logit significant variable p-Value threshold used is ',
  sel_pValue
))
tmp <- mod_d2_glm.sumry$coefficients[, 4]
sel_var_logit <- subset(tmp, tmp <= sel_pValue)
print(sort(sel_var_logit))

GLM_signficant_features <-  c(
  'Issues_Raised',
  'AgeAtCreation',
  'MostUsedBillingGrade',
  'ClientType',
  'MaxCoreProgramHours',
  'Client_Initiated_Cancellations',
  'default_contract_group',
  'Client_Programs_count_at_observation_cutoff',
  'HOME_STR_state',
  'PCNeedsFlag'
  # 'FrequentschedStatusGroup',
  # 'MinCoreProgramHours',
  # 'RespiteNeedsFlag',
  # 'GenderRequired'
)

RFnTree_param <- 1000
ndxLabel <- which(names(d2) == 'Label')

set.seed(1)
# bestmtry <-
#   tuneRF(
#     d2[,-ndxLabel],
#     d2$Label,
#     ntreeTry = RFnTree_param,
#     stepFactor = 1.5,
#     improve = 0.01,
#     plot = TRUE,
#     dobest = FALSE,
#     type = k_RF_var_imp_measure
#   )
# RFmtry_param <- bestmtry[min(bestmtry[, 2]) == bestmtry[, 2]][1]
RFmtry_param <-  round(sqrt(ncol(d2[,-ndxLabel])), 0)

print(paste('tuneRF()  mtry for feature selectio is  ', RFmtry_param))

set.seed(1)
mod_d2_rf <-
  randomForest(
    Label ~ .,
    data = d2,
    mtry = RFmtry_param,
    ntree = RFnTree_param,
    keep.forest = FALSE,
    importance = TRUE
  )


par(mfrow = c(2, 2))
PlotTitle <-
  c('No Churn Accuracy',
    'Churn Accuracy',
    'Churn/No Churn Accuracy',
    'Gini Index')
for (i in 1:4)
  plot(
    sort(mod_d2_rf$importance[, i], dec = TRUE),
    type = "h",
    main = paste(PlotTitle[i]),
    xlab = 'Variable Index',
    ylab = ''
  )

par(mfrow = c(1, 1))

# From the plot, chose importance measure 3:  mean decrease in accuracy of BOTH class
k_RF_var_imp_measure <- 3
k_RF_var_imp_type <- 1

varImpPlot(
  mod_d2_rf,
  sort = TRUE,
  type = k_RF_var_imp_type,
  scale = FALSE,
  class = NULL,
  main = ''
)

# After inspecting the importance plot, choose top 15 variables sorted by mean decrease in accuracy for all class
nRFSelVars <- 15

tmp <-
  importance(mod_d2_rf,
             type = k_RF_var_imp_type,
             class = NULL,
             scale = FALSE)
tmp <- tmp[order(tmp[, 1], decreasing = TRUE),]
print(tmp[1:nRFSelVars])

sel_RF_imp_vars <- names(tmp[1:nRFSelVars])

#Combine the RF and GLM features.
model_features <- union(GLM_signficant_features, sel_RF_imp_vars)
model_features <- unique(model_features)

d2 <- d2[model_features]

#Convert  factors into integers
x <- sapply(d2, function(x)
  class(x))
factor_vars <- x[x == "factor" | length(x) > 1]

for (var in names(factor_vars)) {
  d2[[var]] <- as.numeric(d2[[var]])
}

for (var in names(d2)) {
  d2[[var]] <- scale(d2[[var]], center = TRUE, scale = TRUE)
}

# Perform Correlation Analysis
d2_corr <- cor(d2, method = "pearson")
write.csv(d2_corr, file = "kc_correlation_pearson_model_features.csv")
corrplot(d2_corr, type = "lower")

# High Collinearity is defined as abs(correlation coefficient) >=  0.80
# This threshold is arbitrary and is correlation application domain specific.
# The selection can not be automated. Selecting candidate features to remove
# requires application domain knowledge.
# By manual inspection of the correlation map, following  variables are to be removed

collinear_features_to_be_removed <- c(
  'Closed_Issues',
  'AllRecordsNums',
  'CoreRecordNums',
  'TotalCoreProgramHours',
  'CoreTotalKM',
  'MaxCoreProgramHours',
  'AverageCoreServiceHours',
  'MostUsedPayGrade',
  'Canned_Appointments',
  'FrequentschedStatusGroup'
)
write(collinear_features_to_be_removed, file = "collinear_features_to_be_removed.txt")

model_features <-
  setdiff(model_features, collinear_features_to_be_removed)
print(model_features)

# END FEATURE SELECTION

#================================================================
# START PREDICTION MODELLING
model_features_with_label <- unique(model_features)
model_features_with_label[length(model_features_with_label) + 1] <-
  'Label'

ndx <- which(names(d1) == 'role')
train <- subset(d1, d1$role == 'Train')
train <- train[model_features_with_label]
train <- train[, -ndx]

test <- subset(d1, d1$role == 'Test')
test <- test[model_features_with_label]
test <- test[, -ndx]

# Make the factor levels the same for both train and test
for (feature in model_features) {
  if (is.factor(train[[feature]]) | is.ordered(train[[feature]])) {
    all_levels <-
      union(levels(train[[feature]]), levels(test[[feature]]))
    levels(train[[feature]]) <- all_levels
    levels(test[[feature]]) <- all_levels
  }
}

#Prediction initial threshold for rounding probability
fitThreshold <- 0.5

#This weights both recall and precison the same as a measure of predction accuracy F-score.
fScoreB_param <- 1

train.scaled <- scaleNumbers(train)
test.scaled <- scaleNumbers(test)
#========================
#GLM model and performance
set.seed(1)
kc.glm <-
  glm(Label ~ ., family = binomial(link = "logit"), data = train.scaled)
# kc.glm
summary(kc.glm)

pr.kc.glm <-
  predict(kc.glm, newdata = test.scaled, type = 'response')
fitted.results.glm <- ifelse(pr.kc.glm > fitThreshold, 1, 0)
confusion_maxtix <- table(test.scaled$Label, fitted.results.glm)
print.table(confusion_maxtix)

kc.glm.eval.results <-
  computePerformanceMeasures(confusion_maxtix, fScoreB_param)

#==========================
# Random Forest model and performance

ndxLabel <- which(names(train) == "Label")
set.seed(1)
bestmtry <-
  tuneRF(
    train[,-ndxLabel],
    train$Label,
    ntreeTry = RFnTree_param,
    stepFactor = 1.5,
    improve = 0.01,
    plot = TRUE,
    dobest = FALSE,
    type = k_RF_var_imp_measure
  )

RFmtry_param <- bestmtry[min(bestmtry[, 2]) == bestmtry[, 2]][1]
#RFmtry_param <-  round(sqrt(ncol(train[,-ndxLabel])),0)
print(paste('RF  model mtry value with least OOB error is ', RFmtry_param))

set.seed(1)
kc.rf <-
  randomForest(
    Label ~ .,
    data = train,
    mtry = RFmtry_param,
    ntree = RFnTree_param,
    keep.forest = TRUE,
    importance = TRUE
  )

kc.rf

summary(kc.rf)

varImpPlot(
  kc.rf,
  sort = TRUE,
  type = k_RF_var_imp_type,
  class = NULL,
  scale = FALSE,
  main = ''
)

tmp <-
  importance(kc.rf,
             type = k_RF_var_imp_type,
             class = NULL,
             scale = FALSE)
tmp <- tmp[order(tmp[, 1], decreasing = TRUE),]
print(tmp)

RFtree1 <- getTree(kc.rf, k = 1, labelVar = TRUE)
colnames(RFtree1) <- sapply(colnames(RFtree1), collapse)
RFrules1 <- getConds(RFtree1)
print(head(RFrules1, 20))

# make predictions
pr.kc.rf <- predict(kc.rf, newdata = test, type = 'prob')[, 2]
fitted.results.rf <- ifelse(pr.kc.rf > fitThreshold, 1, 0)
confusion_maxtix <- table(test$Label, fitted.results.rf)
print.table(confusion_maxtix)

kc.rf.eval.results <-
  computePerformanceMeasures(confusion_maxtix, fScoreB_param)

#==========================
# C5.0 model tree, rules and performance

C5.0Trials_param <- 100

set.seed(1)
kc.c50 <-
  C50::C5.0(
    x = train[-ndxLabel],
    y = train$Label,
    trial = C5.0Trials_param,
    rules = FALSE,
    control = C5.0Control(earlyStopping = TRUE)
  )
kc.c50

summary(kc.c50)

# Code below errors out
# plot.C5.0(kc.c50)
# myTree <- C50:::as.party.C5.0(kc.c50)
# plot(myTree)
# plot(kc.c50)

set.seed(1)
kc.c50.rules <-
  C50::C5.0(
    x = train[,-ndxLabel],
    y = train$Label,
    trial = C5.0Trials_param,
    rules = TRUE,
    control = C5.0Control(bands = 100, earlyStopping = TRUE)
  )
# head(summary(kc.c50.rules), 50)

# make predictions
pr.kc.c50 <-
  predict(kc.c50, type = "prob", newdata = test[-ndxLabel])[, 2]
fitted.results.c50 <-
  ifelse(pr.kc.c50 > fitThreshold, 1, 0)
confusion_maxtix <-
  table(test$Label, fitted.results.c50)

kc.c50.eval.results <-
  computePerformanceMeasures(confusion_maxtix, fScoreB_param)


#====================================
# COMPARE THE MODELS USING ROC CURVES

#pr.kc.glm <- predict(kc.glm, type = 'response', newdata=test)
pred.kc.glm <-  prediction(pr.kc.glm, test.scaled$Label)

#pr.kc.rf <- predict(kc.rf, type = "prob", newdata=test)[,2]
pred.kc.rf = prediction(pr.kc.rf, test$Label)

#pr.kc.c50 <- predict(kc.c50, type = "prob", newdata=test)[,2]
pred.kc.c50 = prediction(pr.kc.c50, test$Label)

## Render ROC graphs

perf.kc.glm  <- performance(pred.kc.glm, "tpr", "fpr")
perf.kc.rf = performance(pred.kc.rf, "tpr", "fpr")
perf.kc.c50 = performance(pred.kc.c50, "tpr", "fpr")

plot(perf.kc.glm,
     main = "",
     col = "blue",
     lwd = 2)
plot(perf.kc.rf,
     add = TRUE,
     col = "red",
     lwd = 2)
plot(perf.kc.c50,
     add = TRUE,
     col = "black",
     lwd = 2)
abline(
  a = 0,
  b = 1,
  lwd = 2,
  lty = 2,
  col = "gray"
)
legend(
  "bottom",
  legend = c(
    'Logistic Regression',
    'Random Forest',
    'C5.0 Trees',
    'random guess'
  ),
  col = c("blue", "red", "black", "gray"),
  lty = c(1, 1, 1, 2),
  cex = 0.9,
  box.lty = 0
)

#Compare AUC of 2 Curves using prediction probabilities
rocGLM <- roc(test.scaled$Label, pr.kc.glm)
rocRF <- roc(test$Label, pr.kc.rf)
rocC50 <- roc(test$Label, pr.kc.c50)
kc.roc.eval.results <-
  c(rocGLM$auc, rocRF$auc, rocC50$auc)

roctest.eval.GLM.RF <-
  roc.test(rocGLM, rocRF, method = "delong", paired = TRUE)
roctest.eval.GLM.c50 <-
  roc.test(rocGLM, rocC50, method = "delong", paired = TRUE)
roctest.eval.RF.c50 <-
  roc.test(rocRF, rocC50, method = "delong", paired = TRUE)
roc.test.eval <-
  c(
    roctest.eval.GLM.RF$p.value,
    roctest.eval.GLM.c50$p.value,
    roctest.eval.RF.c50$p.value
  )

#======================
## 10 FOLD CROSS VALIDATION

kFoldValidation_param  <- 10
k <- kFoldValidation_param

data <- rbind(train, test)
set.seed(1)
data$id <-
  sample(1:kFoldValidation_param, nrow(data), replace = TRUE)
ndxId <- which(names(data) == 'id')

list <- 1:kFoldValidation_param

# prediction and testset data frames that we add to with each iteration over the folds
cv.prediction.glm <- data.frame()
cv.prediction.rf <- data.frame()
cv.prediction.c50 <- data.frame()

testsetCopy <- data.frame()

#Creating a progress bar to know the status of CV
progress.bar <- create_progress_bar("text")
progress.bar$init(k)

for (i in 1:k) {
  # remove rows with id i from dataframe to create training set
  # select rows with id i to create test set
  trainingset <- subset(data, id %in% list[-i])
  testset <- subset(data, id %in% c(i))
  trainingset.scaled <-
    scaleNumbers(trainingset[,-ndxId])
  testset.scaled <- scaleNumbers(testset[,-ndxId])
  
  for (feature in model_features) {
    if (is.factor(trainingset[[feature]]) |
        is.ordered(trainingset[[feature]])) {
      all_levels <-
        union(levels(trainingset[[feature]]), levels(testset[[feature]]))
      levels(trainingset[[feature]]) <- all_levels
      levels(testset[[feature]]) <- all_levels
    }
  }
  
  cv.kc.glm <-
    glm(Label ~ ., family = binomial(link = "logit"), data = trainingset.scaled)
  cv.prob.results.glm <-
    predict(cv.kc.glm, type = 'response', newdata = testset.scaled)
  
  cv.kc.rf <-
    randomForest(Label ~ .,
                 data = trainingset,
                 mtry = RFmtry_param,
                 ntree = RFnTree_param)
  cv.prob.results.rf <-
    predict(cv.kc.rf, newdata = testset, type = 'prob')[, 2]
  
  cv.kc.c50 <-
    C50::C5.0(
      x = trainingset[,-ndxLabel],
      y = trainingset$Label,
      trial = C5.0Trials_param,
      control = C5.0Control(earlyStopping = TRUE)
    )
  cv.prob.results.c50 <-
    predict(cv.kc.c50, newdata = testset[,-ndxLabel], type = 'prob')[, 2]
  
  # append this iteration's predictions to the end of the prediction data frames
  cv.prediction.glm <-
    rbind(cv.prediction.glm, as.data.frame(cv.prob.results.glm))
  cv.prediction.rf  <-
    rbind(cv.prediction.rf, as.data.frame(cv.prob.results.rf))
  cv.prediction.c50 <-
    rbind(cv.prediction.c50, as.data.frame(cv.prob.results.c50))
  
  # append this iteration's test set to the test set copy data frame
  # keep only the Churn Label Column
  testsetCopy <-
    rbind(testsetCopy, as.data.frame(testset[ndxLabel]))
  progress.bar$step()
}

# add predictions and actual churn labels
result.glm <- cbind(cv.prediction.glm, testsetCopy)
result.rf  <- cbind(cv.prediction.rf, testsetCopy)
result.c50 <- cbind(cv.prediction.c50, testsetCopy)

names(result.glm) <- c("Predicted", "Actual")
names(result.rf)  <- c("Predicted", "Actual")
names(result.c50) <- c("Predicted", "Actual")

# AUC as CV criterion
cv.rocGLM <-
  roc(result.glm$Actual, result.glm$Predicted)
cv.rocRF  <- roc(result.rf$Actual, result.rf$Predicted)
cv.rocC50 <-
  roc(result.c50$Actual, result.c50$Predicted)
kc.cv.roc.eval.results <-
  c(cv.rocGLM$auc, cv.rocRF$auc, cv.rocC50$auc)

# AUC Signifcance Test
cv.roctest.eval.GLM.RF <-
  roc.test(cv.rocGLM, cv.rocRF, method = "delong", paired = TRUE)
cv.roctest.eval.GLM.c50 <-
  roc.test(cv.rocGLM, cv.rocC50, method = "delong", paired = TRUE)
cv.roctest.eval.RF.c50 <-
  roc.test(cv.rocRF, cv.rocC50, method = "delong", paired = TRUE)

## Render 10-X Validation ROC graphs

pred.kc.glm <-
  prediction(result.glm$Predicted, result.glm$Actual)
pred.kc.rf <-
  prediction(result.rf$Predicted, result.rf$Actual)
pred.kc.c50 <-
  prediction(result.c50$Predicted, result.c50$Actual)

perf.kc.glm  <- performance(pred.kc.glm, "tpr", "fpr")
perf.kc.rf = performance(pred.kc.rf, "tpr", "fpr")
perf.kc.c50 = performance(pred.kc.c50, "tpr", "fpr")

plot(perf.kc.glm,
     main = "",
     col = "blue",
     lwd = 2)
plot(perf.kc.rf,
     add = TRUE,
     col = "red",
     lwd = 2)
plot(perf.kc.c50,
     add = TRUE,
     col = "black",
     lwd = 2)
abline(
  a = 0,
  b = 1,
  lwd = 2,
  lty = 2,
  col = "gray"
)
legend(
  "bottom",
  legend = c(
    'Logistic Regression',
    'Random Forest',
    'C5.0 Trees',
    'random guess'
  ),
  col = c("blue", "red", "black", "gray"),
  lty = c(1, 1, 1, 2),
  cex = 0.9,
  box.lty = 0
)

# use Overa-all Prediction Accuracy as CV criteria
print(paste('CV Probability fitThreshold used is ', fitThreshold))

cv.fitted.result.glm <-
  ifelse(result.glm$Predicted > fitThreshold, 1, 0)
cv.fitted.result.rf <-
  ifelse(result.rf$Predicted > fitThreshold, 1, 0)
cv.fitted.result.c50 <-
  ifelse(result.c50$Predicted > fitThreshold, 1, 0)

confusion_maxtix <-
  table(result.glm$Actual, cv.fitted.result.glm)
cv.accur.glm.result <-
  computePerformanceMeasures(confusion_maxtix, fScoreB_param)[1]

confusion_maxtix <-
  table(result.rf$Actual, cv.fitted.result.rf)
cv.accur.rf.result <-
  computePerformanceMeasures(confusion_maxtix, fScoreB_param)[1]

confusion_maxtix <-
  table(result.c50$Actual, cv.fitted.result.c50)
cv.accur.c50.result <-
  computePerformanceMeasures(confusion_maxtix, fScoreB_param)[1]

kc.cv.accur.results <-
  c(cv.accur.glm.result, cv.accur.rf.result, cv.accur.c50.result)


# Tweak C50 modelfor bias in population
pr.kc.c50.train <-
  predict(kc.c50, type = "prob", newdata = train[-ndxLabel])[, 2]
x <- cbind(test$Label, pr.kc.c50)
y <- subset(x, x[, 1] == 2)
summary(y[, 2])

fitThreshold.adj <-
  summary(y[, 2])[4]  #Mean Probability value
print(
  paste(
    'New C5.0 prediction prob. threshold set to min probabilty of churn class',
    fitThreshold.adj
  )
)

fitted.results.c50 <-
  ifelse(pr.kc.c50 > fitThreshold.adj, 1, 0)
confusion_maxtix <-
  table(test$Label, fitted.results.c50)
kc.c50.eval.results.adj <-
  c(computePerformanceMeasures(confusion_maxtix, fScoreB_param),
    NA,
    NA,
    NA)

# Recompute fScores for C5.0 model where prediction recall weights 2 times more than precision accuracy
print('C5.0 acc, prec, rec, F2 Score')

computePerformanceMeasures(confusion_maxtix, 2)

# Retrain C5.0 and rePredict with misclasification cost is
# twice more for false positive (Type 2 misclasification error).
miscosts <- matrix(c(NA, 2, 1, NA),
                   nrow = 2,
                   ncol = 2,
                   byrow = TRUE)

kc.c50.mis <-
  C50::C5.0(
    x = train[-ndxLabel],
    y = train$Label,
    trial = C5.0Trials_param,
    costs = miscosts,
    rules = FALSE,
    control = C5.0Control(earlyStopping = TRUE)
  )

kc.c50.mis

summary(kc.c50.mis)

pr.kc.c50.mis <-
  predict(kc.c50.mis, newdata = test[-ndxLabel])
confusion_maxtix <- table(test$Label, pr.kc.c50.mis)
kc.c50.mis.eval.results <-
  c(computePerformanceMeasures(confusion_maxtix, fScoreB_param),
    NA,
    NA,
    NA)

### Construct the Model comparison Performance Matrix

kc.eval.matrix <- data.frame()
kc.eval.matrix <-
  rbind(kc.glm.eval.results, kc.rf.eval.results, kc.c50.eval.results)
kc.eval.matrix <-
  cbind(kc.eval.matrix,
        kc.cv.accur.results,
        kc.roc.eval.results,
        kc.cv.roc.eval.results)
kc.eval.matrix <-
  rbind(kc.eval.matrix,
        kc.c50.eval.results.adj,
        kc.c50.mis.eval.results)

rownames(kc.eval.matrix)  <-
  c('Logit',
    'RF',
    'C50',
    'C5.0 adjusted for Bias',
    'C5.0 with Misclassify Cost')
colnames(kc.eval.matrix)  <-
  c(
    'Accuracy',
    'Precision',
    'Recall',
    'F-score',
    '10X-CV Accuracy',
    'Model AUC',
    '10X-CV AUC'
  )

kc.eval.matrix[, 1:7] <-
  round(as.numeric(kc.eval.matrix[, 1:7]), digits = 4)

print('Performance Matrix')

print(kc.eval.matrix)

roc.test.eval <- data.frame()
cv.roc.test.eval <- data.frame()

roc.test.eval <-
  c(
    roctest.eval.GLM.RF$p.value,
    roctest.eval.GLM.c50$p.value,
    roctest.eval.RF.c50$p.value
  )
cv.roc.test.eval <-
  c(
    cv.roctest.eval.GLM.RF$p.value,
    cv.roctest.eval.GLM.c50$p.value,
    cv.roctest.eval.RF.c50$p.value
  )
kc.roc.test.eval <-
  cbind(roc.test.eval, cv.roc.test.eval)
kc.roc.test.eval <-
  cbind(kc.roc.test.eval, (ifelse(kc.roc.test.eval < .05, 'Yes', 'No')))

rownames(kc.roc.test.eval)  <-
  c('Logit to RF', 'Logit to C50', 'RF to C50')
colnames(kc.roc.test.eval) <-
  c('Test', '10-X Test', '95% CI Test', '95% CI 10-X Test')
kc.roc.test.eval[, 1:2] <-
  round(as.numeric(kc.roc.test.eval[, 1:2]), digits = 4)

print('Test for Significance(delong) between Paired Models p-Value')

print(kc.roc.test.eval)

