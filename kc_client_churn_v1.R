# ***********************************************************************************************
#  R  Program : kc_client_churn_v1.R
#  Ran Log    : kc_client_churn_v1.R.log
#
#  Author     : Raul Manongdo, University of Technology, Sydney
#               Advance Analytics Institute
#  Date       : 1 Aug 2016

#  Program description:
#     - Birary client churn prediction modelling for anonymous company  using 3 statistical models; 
#        Logistic Regression, Random Forest and C5.0 Decison Trees
#     - Feature selection by mainly by multiple GLM rans and correlation analysis
#     - Train/Test set is 80%-20% stratified sampling split
#     - Model comparison by various prediction accuracy measures, ROC curves, AUC and 5 fold X Validation
# ***********************************************************************************************

library(randomForest,quietly=TRUE)
library (gplots,quietly=TRUE)
library(ROCR)
library(plyr)
library(C50)

numeric_features <- c(
  'AgeAtCreation',
  'TotalCoreProgramHours',
  'MaxCoreProgramHours',
  'MinCoreProgramHours',
  'AverageCoreProgramHours',
  'AverageCoreServiceHours',
  'Client_Programs_count_at_observation_cutoff',
  'Issues_Raised',
  'Issues_Requiring_Action',
  'Escalated_Issues',
  'Closed_Issues',
  'Client_Initiated_Cancellations',
  'Kincare_Initiated_Cancellations',
  'Canned_Appointments',
  'HCW_Ratio',
  'FirstCoreServiceHours',
  'LastCoreServiceHours',
  "Responses_1","Responses_2","Responses_3","Responses_4","Responses_5","Responses_6",
  "Responses_7","Responses_10")

factor_features <- c(
  "Label",
  "Sex",
  "HOME_STR_state",
  "ClientType",
  "Grade",
  "SmokerAccepted",
  "GenderRequired",
  "EthnicGroupRequired",
  "SpokenLanguageRequired",
  "FrequentschedStatusGroup",
  "MostUsedBillingGrade",
  "MostUsedPayGrade",
  "RespiteNeedsFlag",
  "DANeedsFlag",
  "NCNeedsFlag",
  "PCNeedsFlag",
  "SocialNeedsFlag",
  "TransportNeedsFlag",
  "PreferredWorkersFlag",
  "default_contract_group",
  "complainttier",
  "Responses_8",
  "Responses_9",
  "RequiredWorkersFlag")

model_features <-  c(
  "Label",
  "HOME_STR_state", 
  "Sex",                       
  "ClientType",                
  # "SpokenLanguageRequired",  #insignificant after 2nd GLM run selection
  "CoreProgramsNums",          
  # "TotalCoreProgramHours",   #highly correlated to AverageCoreProgramHours
  "MaxCoreProgramHours",
  "MinCoreProgramHours",
  "AverageCoreProgramHours",  
  # "FrequentschedStatusGroup",  # insignificant after 2nd GLM run selection
  "MostUsedBillingGrade",
  "MostUsedPayGrade",
  "PCNeedsFlag",            
  "Client_Programs_count_at_observation_cutoff",
  "complainttier",
  "Issues_Raised",
  "Client_Initiated_Cancellations"
 )


# -----------------------------------------------------------------------------
# LOAD RAW DATA

raw.data  <-  read.csv("anonymous.csv",na.strings=c("","NA","<NA>"),stringsAsFactors = FALSE)

con <- file("kc_client_churn_v1.R.log")
sink(con,append=FALSE)
sink(con,append=TRUE, type="message")

# DATA CLEANSING

# Remove attributes that have more than 50% missing values
sapply(raw.data[c(10,11,36,37,38,39,40,41,42,43,44,45,46,47,61, 71,72)],function(x) sum(is.na(x)))
raw.data <- raw.data[,c(-10,-11,-36,-37,-38,-39,-40,-41,-42,-43,-44,-45,-46,-47,-61,-71,-72)]

# Remove unique key values
raw.data <- raw.data[-1]  # ClientID

# Assign median to outliers
for (feature in numeric_features) {
  lowerq <- quantile(raw.data[[feature]],na.rm=TRUE)[1]
  upperq <- quantile(raw.data[[feature]],na.rm=TRUE)[4]
  iqr = upperq - lowerq 
  extreme.threshold.upper <- upperq + (iqr * 3.0) 
  extreme.threshold.lower <- lowerq - (iqr * 3.0) 
  medianf <- median(raw.data[[feature]],na.rm=TRUE)
  extreme.threshold.lowrer = lowerq - (iqr*3)
  raw.data[[feature]][raw.data[[feature]] > extreme.threshold.upper] <- medianf
  raw.data[[feature]][raw.data[[feature]] < extreme.threshold.lower] <- medianf
}

# Assign mean to missing values
for (feature in numeric_features) {
  raw.data[[feature]][is.na(raw.data[[feature]])] <- mean(raw.data[[feature]],na.rm=TRUE)
}

#  Assign "missing" to missing categorical values
for (feature in factor_features) {
  raw.data[[feature]][is.na(raw.data[[feature]])] <- "missing"
}
sapply(raw.data,function(x) sum(is.na(x)))

# Convert categorical variables to factors
for (feature in factor_features) {
  raw.data[[feature]] <- as.factor(as.character(raw.data[[feature]]))
}

# FEATURE SELECTION
d1 <- raw.data
# Identify significant variables using GLM
mod_d1 <- glm(Label ~.,family = 'binomial', data = d1)
summary(mod_d1)

# Selected variables are manually listed in model_features variable above
d1 <- d1[model_features]

# CORRELATION VARIABLE ANALYSIS
# Change to all variables into integer and run the model
# for (feature in model_features) {
#   d1[[feature]] <- as.integer(d1[[feature]])
# }
# cor_txt <- cor(d1, method="spearman")
# write.table(cor_txt, file="corfile.csv",eol="\r", sep=",")
# This steps remove 2 attributes due to collinearity in model_feautites shown with #
# Graph the correlation 
# pairs(d2, gap=0, pch=20, cex=0.8, col="darkblue")
# title(sub="Scatterplot of Numeric Variables", cex=0.12)
# This steps further removes 1 variable

# Run GLM for model selection again
# d1 <- d1[model_features]
# d1$Label <- as.factor(d1$Label)
# mod_d1 <- glm(Label ~.,family = 'binomial', data = d1)
# summary(mod_d1)
# This steps further removes X variable in dataset d1

## DATASET PREPARATION
dataset <- raw.data[model_features]

# Train test splitting based on 80/20 stratified sampling
dataset.churn <- subset(dataset,dataset$Label==1)
dataset.nochurn <- dataset[setdiff(rownames(dataset), rownames(dataset.churn)),]
set.seed(1)
churnSample80pcnt <- sample(1:nrow(dataset.churn),0.8*nrow(dataset.churn))
set.seed(1)
nochurnSample80pcnt <- sample(1:nrow(dataset.nochurn),0.8*nrow(dataset.nochurn))

train.churn <- dataset.churn[churnSample80pcnt,]
train.nochurn <- dataset.nochurn[nochurnSample80pcnt,]

train <- rbind(train.churn,train.nochurn)
test <- dataset[setdiff(rownames(dataset),rownames(train)),]

# Make the factor levels the same for both train and test
for (feature in model_features) {
    if (class(train[[feature]])=="factor") {
    all_levels <- union(levels(train[[feature]]),levels(test[[feature]]))
    levels(train[[feature]]) <- all_levels
    levels(test[[feature]]) <- all_levels
  }
}

# GLM MODEL AND PERFORMANCE
kc.glm <- glm(Label ~.,family = 'binomial', data = train)
kg.glm
summary(kc.glm)

anova(kc.glm, test="Chisq")

pr.kc.glm <- predict(kc.glm, type = 'response', newdata=test)
fitted.results <- ifelse(pr.kc.glm > 0.5, 1, 0)
misClasificError <- mean(fitted.results != test$Label)
print(paste('GLM Comp. Accuracy', 1 - misClasificError))

confusion_maxtix <- table(test$Label,fitted.results)
print.table(confusion_maxtix)

pred.kc.glm <-  prediction(fitted.results,test$Label)

# other accuracy measures
acc <- performance(pred.kc.glm,'acc')
acc <- mean(acc@y.values[[1]],na.rm=TRUE)
print (paste("GLM Perf. Accuracy  ",acc))

prec <- performance(pred.kc.glm,measure='prec')
prec <- mean(prec@y.values[[1]],na.rm=TRUE)
print (paste("GLM Precison  ",prec))

rec <- performance(pred.kc.glm,measure='rec')
rec <- mean(rec@y.values[[1]],na.rm=TRUE)
print (paste("GLM Recall  ",rec))

f <- performance(pred.kc.glm,measure='f')
f <- mean(f@y.values[[1]],na.rm=TRUE)
print (paste("GLM F Score ",f))

# RANDOM FOREST MODEL and PERFORMANCE

# Fnding the optimal numbers of variables to try splitting on at each node
bestmtry <- tuneRF(dataset[-1],dataset$Label, ntreeTry=1000, stepFactor=1.5,
                   improve=0.01, trace=TRUE, plot=TRUE, dobest=FALSE)
# From above step, optimal mtry value is 2 at OOB Error of 7.45% over 1,000 trees 

# Run the Random Forest model
kc.rf <-randomForest(Label ~., data = train, mtry=bestmtry, ntree=100, keep.forest=TRUE, importance=TRUE) 
varImpPlot(kc.rf)
kc.rf
summary(kc.rf)

# accuracy
fitted.results.rf <- predict(kc.rf, newdata = test, type = 'prob')[,2]
fitted.results.rf <- ifelse(fitted.results.rf > 0.5, 1, 0)
misClasificError <- mean((fitted.results.rf != test$Label),na.rm=TRUE)
print(paste('RF Comp. Accuracy', 1 - misClasificError))

# Confusion matrix
confusion_maxtix <- table(test$Label,fitted.results)
print.table(confusion_maxtix)

pred.kc.rf = prediction(fitted.results.rf, test$Label)

# other accuracy measures
acc <- performance(pred.kc.rf,measure='acc')
acc <- mean(acc@y.values[[1]])
print (paste("RF Perf. Accuracy ",acc))

prec <- performance(pred.kc.rf,measure='prec')
prec <- mean(prec@y.values[[1]],na.rm=TRUE)
print (paste("RF Precison  ",prec))

rec <- performance(pred.kc.rf,measure='rec')
rec <- mean(rec@y.values[[1]],na.rm=TRUE)
print (paste("RF Recall  ",rec))

f <- performance(pred.kc.rf,measure='f')
f <- mean(f@y.values[[1]],na.rm=TRUE)
print (paste("RF F Score ",f))

## C5.0 MODEL and PERFORMANCE

kc.c50 <- C50::C5.0(x= train[,-1], y=train$Label, trial = 10)
kc.c50
summary(kc.c50)

# make predictions
fitted.results.c50 <- predict(kc.c50, test[,-1], type = 'class')
misClasificError.c50 <- mean((fitted.results.c50 != test$Label),na.rm=TRUE)
print(paste('C5.0 Comp. Accuracy', 1 - misClasificError.c50))

# Confusion matrix
confusion_maxtix <- table(test$Label,fitted.results.c50)
print.table(confusion_maxtix)

fitted.results.c50 <- predict(kc.c50, test[,-1], type = 'prob')[,2]
fitted.results.c50 <- ifelse(fitted.results.c50 > 0.5, 1, 0)
pred.kc.c50 <-  prediction(fitted.results.c50, test$Label)

# other accuracy measures
acc <- performance(pred.kc.c50,measure='acc')
acc <- mean(acc@y.values[[1]])
print (paste("C5.0 Pred. Accuracy  ",acc))

prec <- performance(pred.kc.c50, measure='prec')
prec <- mean(prec@y.values[[1]],na.rm=TRUE)
print (paste("C5.0 Precison  ",prec))

rec <- performance(pred.kc.c50,measure='rec')
rec <- mean(rec@y.values[[1]],na.rm=TRUE)
print (paste("C5.0 Recall  ",rec))

f <- performance(pred.kc.c50,measure='f')
f <- mean(f@y.values[[1]],na.rm=TRUE)
print (paste("C5.0 F Score ",f))

## COMPARE THE MODELS USING ROC CURVES

pr.kc.glm <- predict(kc.glm, type = 'response', newdata=test)
pred.kc.glm <-  prediction(pr.kc.glm,test$Label)
auc <- performance(pred.kc.glm, measure = "auc")
auc <- auc@y.values[[1]]
print (paste("GLM - Area under ROC curve ",auc))
perf.kc.glm  <- performance(pred.kc.glm,"tpr","fpr")

pr.kc.rf <- predict(kc.rf, type = "prob", newdata=test)[,2]
pred.kc.rf = prediction(pr.kc.rf, test$Label)
auc = performance(pred.kc.rf,measure='auc')
auc <- auc@y.values[[1]]
print (paste("RF- Area under ROC curve ",auc))
perf.kc.rf = performance(pred.kc.rf,"tpr","fpr")

pr.kc.c50 <- predict(kc.c50, type = "prob", newdata=test)[,2]
pred.kc.c50 = prediction(pr.kc.c50, test$Label)
auc = performance(pred.kc.c50,measure='auc')
auc <- auc@y.values[[1]]
print (paste("C5.0- Area under ROC curve ",auc))
perf.kc.c50 = performance(pred.kc.c50,"tpr","fpr")

## render ROC graphs
plot(perf.kc.glm, main="ROC comparison of Prediction models",col="blue",lwd=2)
plot(perf.kc.rf, add = TRUE, col="red", lwd=2)
plot(perf.kc.c50, add = TRUE, col="black", lwd=2)
abline(a=0,b=1,lwd=2,lty=2,col="gray")
legend("bottom",legend=c('Logistic Regression','Random Forest','C5.0 Trees','random guess'), 
      col=c("blue","red","black","gray"), lty=c(1,1,1,2), cex=0.9,box.lty=0)

## 5 FOLD CROSS VALIDATION COMPARISON

data <- rbind(train,test)
k = 5 #Folds

set.seed(1)
data$id <- sample(1:k, nrow(data), replace = TRUE)
list <- 1:k

# prediction and testset data frames that we add to with each iteration over the folds
prediction.glm <- data.frame()
prediction.rf <- data.frame()
prediction.c50 <- data.frame()

testsetCopy <- data.frame()

#Creating a progress bar to know the status of CV
progress.bar <- create_progress_bar("text")
progress.bar$init(k)

for (i in 1:k){
  # remove rows with id i from dataframe to create training set
  # select rows with id i to create test set
  trainingset <- subset(data, id %in% list[-i])
  testset <- subset(data, id %in% c(i))

  kc.glm <- glm(trainingset$Label ~.,family = 'binomial', data = trainingset)
  levels_in_models <- kc.glm$xlevels
  for (feature in names(levels_in_models))
    levels(testset[[feature]]) <- factor(as.character(trainingset[[feature]]),levels = levels_in_models[[feature]])
  
  pr.kc.glm <- predict(kc.glm, type = 'response', newdata=testset)
  fitted.results.glm <- ifelse(pr.kc.glm > 0.5, 1, 0)
  
  for (feature in model_features) {
    if (class(trainingset[[feature]])=="factor") {
      all_levels <- union(levels(trainingset[[feature]]),levels(testset[[feature]]))
      levels(trainingset[[feature]]) <- all_levels
      levels(testset[[feature]]) <- all_levels
    }
  }

  kc.rf <- randomForest(Label ~., data = trainingset, mtry=bestmtry, ntree=100) 
  fitted.results.rf <- predict(kc.rf, newdata = testset, type = 'prob')[,2]
  fitted.results.rf <- ifelse(fitted.results.rf > 0.5, 1, 0)

  kc.c50 <- C50::C5.0(x= trainingset[,-1], y=trainingset$Label, trial = 10)
  fitted.results.c50 <- predict(kc.c50, newdata = testset[,-1], type = 'prob')[,2]
  fitted.results.c50 <- ifelse(fitted.results.c50 > 0.5, 1, 0)
  
  # append this iteration's predictions to the end of the prediction data frames
  prediction.glm <- rbind(prediction.glm, as.data.frame(fitted.results.glm))
  prediction.rf  <- rbind(prediction.rf, as.data.frame(fitted.results.rf))
  prediction.c50 <- rbind(prediction.c50, as.data.frame(fitted.results.c50))
  
  # append this iteration's test set to the test set copy data frame
  # keep only the Churn Label Column
  testsetCopy <- rbind(testsetCopy, as.data.frame(testset[1]))
  progress.bar$step()
}

# add predictions and actual churn labels
result.glm <- cbind(prediction.glm, testsetCopy)
result.rf <- cbind(prediction.rf, testsetCopy)
result.c50 <- cbind(prediction.c50, testsetCopy)

names(result.glm) <- c("Predicted", "Actual")
names(result.rf)  <- c("Predicted", "Actual")
names(result.c50) <- c("Predicted", "Actual")

# use Mean as CV Evalution 
result.glm$Difference <- ifelse(result.glm$Actual==result.glm$Predicted,1,0)
print(paste ('GLM 5-X-fold validation  ', mean(result.glm$Difference)))

result.rf$Difference <- ifelse(result.rf$Actual==result.rf$Predicted,1,0)
print(paste('RF 5-X-fold validation   ',mean(result.rf$Difference)))
      
result.c50$Difference <- ifelse(result.c50$Actual==result.c50$Predicted,1,0)
print(paste('C5.0 5-X-fold validation ',mean(result.c50$Difference)))

sink()
unlink(con)



            
