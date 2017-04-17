kc\_client\_churn\_FINAL\_F.Rmd
================

``` r
# ***********************************************************************************************
#  R  Program : kc_client_churn_FINAL_F.R
#
#  Author     : Raul Manongdo, University of Technology, Sydney
#               Advance Analytics Institute
#  Date       : April 2017

#  Program description:
#     Birary client churn prediction modellingusing 3 statistical models;
#        Logistic Regression, Random Forest and C5.0 Decison Trees.
#     Feature selection using GLM + RF models - Pearson correlation analysis.
#     Train/Test as separate datasets.
#     Threshold for prediction probability initialy set at default value of .5 and 
#         tweak to mean of probability of chuen class
#     Model comparison by area under ROC and test of signifance using Delong method
#     Model selection by F score and Accuracy scores.
#     10 fold X validation is 90%/10% training and testing split
#     10 fold X validation measure is mean prediction accuracy and AUC
# ***********************************************************************************************

library(lattice)
library(randomForest, quietly = TRUE)
```

    ## randomForest 4.6-12

    ## Type rfNews() to see new features/changes/bug fixes.

``` r
library (gplots, quietly = TRUE)
```

    ## 
    ## Attaching package: 'gplots'

    ## The following object is masked from 'package:stats':
    ## 
    ##     lowess

``` r
library(ROCR)
library(plyr)
library(C50)
library (pROC)
```

    ## Warning: package 'pROC' was built under R version 3.3.2

    ## Type 'citation("pROC")' for a citation.

    ## 
    ## Attaching package: 'pROC'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     cov, smooth, var

``` r
library(partykit)
```

    ## Loading required package: grid

``` r
# library(stargazer)

selectRawDataVariables <- function (dataset) {
  # Remove unique key values <= 1
  dataset <-
    dataset[-which(names(dataset) == 'myUniqueClientID')]
  
  # Remove attributes that have more than 50% missing values
  n50pcnt <- round(nrow(dataset) * .50)
  x <- sapply(dataset, function(x)
    sum(is.na(x)) + sum(is.nan(x)))
  drops <- x[x > n50pcnt]
  dataset <- dataset[, !(names(dataset) %in% names(drops))]
  
  # Remove attributes with only 1 value
  x <-
    sapply(dataset, function(x)
      ifelse(is.numeric(x), max(x) - min(x), NA))
  drops <- x[x == 0 & !is.na(x)]
  dataset <- dataset[, !(names(dataset) %in% names(drops))]
  
  return (dataset)
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
    dataset[[var]][is.na(dataset[[var]]) |
                     is.nan(dataset[[var]])] <- default_val
  }
  
  # Convert binary outcome variable
  dataset$Label <- ifelse (dataset$Label == 'Churn', 1, 0)
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
  
  # Impute mean as for missing  numeric values
  
  # For numeric values, set NaN and negative values to NA : bad data
  for (var in num_vars)
    dataset[[var]] <-
    ifelse(is.nan(dataset[[var]]), NA, dataset[[var]])

  for (var in num_vars)
    dataset[[var]][is.na(dataset[[var]])] <-
    mean(dataset[[var]], na.rm = TRUE)

  # Assign lower and upper quantile values to numeric outliers
  k_iqr_multiplier <- 3
  for (var in num_vars) {
    tmp <- quantile(dataset[[var]], na.rm = TRUE)
    iqr <- tmp[4] - tmp[2]
    extreme.upper <- tmp[4] + (iqr * k_iqr_multiplier)
    extreme.lower <- tmp[2] - (iqr * k_iqr_multiplier)
    extreme.lower <-  ifelse (extreme.lower < 0, 0, extreme.lower)
    dataset[[var]] <-
      ifelse (dataset[[var]] >= extreme.upper, extreme.upper, dataset[[var]])
    dataset[[var]] <-
      ifelse (dataset[[var]] <= extreme.lower, extreme.lower, dataset[[var]])
  }
  
  # Drop number columns with only one value
  x <-  sapply(dataset[num_vars], function(x) ifelse((max(x) == min(x)), T, F))
  drops <- x[x == TRUE]
  dataset <- dataset[, !(names(dataset) %in% names(drops))]

  return(dataset)
}

#**************************
#returns a Z score normalised  dataset
#**************************
scaleNumbers <- function(dataset) {
  # Get numeric only variables
  numericVars <-
    sapply(dataset, function(x) {
      ifelse ((is.numeric(x) & !is.factor(x)), T, F)
    })
  
  for (varName in names(numericVars[numericVars == TRUE])) {
    dataset[[varName]] <-
      scale(dataset[[varName]], center = TRUE, scale = TRUE)
  }
  
  return (dataset)
}


#**************************
#return the rules of an RF tree
#**************************
getConds<-function(tree){
  #store all conditions into a list
  conds<-list()
  #start by the terminal nodes and find previous conditions
  id.leafs<-which(tree$status==-1)
  j<-0
  for(i in id.leafs){
    j<-j+1
    prevConds<-prevCond(tree,i)
    conds[[j]]<-prevConds$cond
    while(prevConds$id>1){
      prevConds<-prevCond(tree,prevConds$id)
      conds[[j]]<-paste(conds[[j]]," & ",prevConds$cond)
      if(prevConds$id==1){
        conds[[j]]<-paste(conds[[j]]," => ",tree$prediction[i])
        break()
      }
    }
    
  }
  return(conds)
}

#**************************
#find the previous conditions in an RF tree
#**************************
prevCond<-function(tree,i){
  if(i %in% tree$right_daughter){
    id<-which(tree$right_daughter==i)
    cond<-paste(tree$split_var[id],">",tree$split_point[id])
  }
  if(i %in% tree$left_daughter){
    id<-which(tree$left_daughter==i)
    cond<-paste(tree$split_var[id],"<",tree$split_point[id])
  }
  
  return(list(cond=cond,id=id))
}

#remove spaces in a word
collapse<-function(x){
  x<-sub(" ","_",x)
  
  return(x)
}

#**************************
#returns model performance measures
#**************************

computePerformanceMeasures <- function (confusion_maxtix, b) {
  tn <- confusion_maxtix[1, 1]
  fp <- confusion_maxtix[1, 2]
  fn <- confusion_maxtix[2, 1]
  tp <- confusion_maxtix[2, 2]
  acc <- (tp + tn) / (tp + tn + fp + fn)
  prec <- (tp / (tp + fp))
  rec  <- (tp / (tp + fn))
  b <- ifelse(is.null(b), 1, b)
  fScore <- 2 * (prec * rec)/(prec + rec)
  return (c(acc, prec, rec, fScore))
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
summary (raw.data)
```

    ##  myUniqueClientID HOME_STR_state     Sex         ClientType  
    ##  Min.   :  105    ACT: 288       Female:1586   HAC    :1379  
    ##  1st Qu.:10039    NSW:2950       Male  : 784   EAC    : 310  
    ##  Median :15866    QLD: 547       NA's  :2760   Com    : 304  
    ##  Mean   :17069    SA : 982                     CCP    : 163  
    ##  3rd Qu.:22335    VIC:  95                     Dom    : 136  
    ##  Max.   :48723    WA : 268                     (Other): 379  
    ##                                                NA's   :2459  
    ##     Grade      SmokerAccepted GenderRequired EthnicGroupRequired
    ##  Grade1:2829   No  :2739      Either:2405    Australia: 954     
    ##  Grade2:1005   Yes :2390      Female:2619    Others   : 112     
    ##  Grade3: 276   NA's:   1      Male  : 106    Italy    :  19     
    ##  Grade4:  16                                 Greece   :  16     
    ##  Grade5:  55                                 England  :  12     
    ##  Grade6: 164                                 (Other)  :  21     
    ##  NA's  : 785                                 NA's     :3996     
    ##  SpokenLanguageRequired ClientAge      AssDaysAfterCreation
    ##  English:1288           Mode:logical   Min.   : 0.00       
    ##  Others :  43           NA's:5130      1st Qu.: 0.00       
    ##  Arabic :  34                          Median : 0.00       
    ##  Italian:  33                          Mean   :16.50       
    ##  Greek  :  31                          3rd Qu.:20.75       
    ##  (Other):  50                          Max.   :89.00       
    ##  NA's   :3651                          NA's   :4780        
    ##  AgeAtCreation     Responses_1    Responses_2     Responses_3   
    ##  Min.   :  0.00   Min.   :1.00   Min.   :2.000   Min.   :2.000  
    ##  1st Qu.: 73.00   1st Qu.:4.00   1st Qu.:4.000   1st Qu.:4.000  
    ##  Median : 82.00   Median :4.00   Median :4.000   Median :4.000  
    ##  Mean   : 78.83   Mean   :4.33   Mean   :4.406   Mean   :4.388  
    ##  3rd Qu.: 87.00   3rd Qu.:5.00   3rd Qu.:5.000   3rd Qu.:5.000  
    ##  Max.   :115.00   Max.   :5.00   Max.   :5.000   Max.   :5.000  
    ##                   NA's   :4576   NA's   :4576    NA's   :4576   
    ##   Responses_4     Responses_5     Responses_6     Responses_7   
    ##  Min.   :1.000   Min.   :1.000   Min.   :1.000   Min.   :1.000  
    ##  1st Qu.:4.000   1st Qu.:4.000   1st Qu.:4.000   1st Qu.:4.000  
    ##  Median :4.000   Median :5.000   Median :4.000   Median :4.000  
    ##  Mean   :4.359   Mean   :4.471   Mean   :4.071   Mean   :4.005  
    ##  3rd Qu.:5.000   3rd Qu.:5.000   3rd Qu.:5.000   3rd Qu.:5.000  
    ##  Max.   :5.000   Max.   :5.000   Max.   :5.000   Max.   :5.000  
    ##  NA's   :4576    NA's   :4576    NA's   :4598    NA's   :4711   
    ##  Responses_8   Responses_9    Responses_10  AllRecordsNums   
    ##  No    :   7   No    :  18   Min.   :1.00   Min.   :    1.0  
    ##  Unsure:  44   Unsure:  25   1st Qu.:4.00   1st Qu.:    9.0  
    ##  Yes   : 503   Yes   : 511   Median :4.00   Median :   35.0  
    ##  NA's  :4576   NA's  :4576   Mean   :4.33   Mean   :  138.4  
    ##                              3rd Qu.:5.00   3rd Qu.:  100.0  
    ##                              Max.   :5.00   Max.   :11064.0  
    ##                              NA's   :4576                    
    ##  CoreProgramsNums CoreRecordNums    TotalCoreProgramHours
    ##  Min.   :1.000    Min.   :    1.0   Min.   :    0.25     
    ##  1st Qu.:1.000    1st Qu.:    9.0   1st Qu.:   12.50     
    ##  Median :1.000    Median :   34.0   Median :   48.50     
    ##  Mean   :1.309    Mean   :  136.5   Mean   :  214.87     
    ##  3rd Qu.:1.000    3rd Qu.:   99.0   3rd Qu.:  143.00     
    ##  Max.   :7.000    Max.   :11064.0   Max.   :73077.33     
    ##                                     NA's   :4            
    ##  MaxCoreProgramHours MinCoreProgramHours AverageCoreProgramHours
    ##  Min.   :  0.250     Min.   :-22.0000    Min.   :    0.25       
    ##  1st Qu.:  2.000     1st Qu.:  0.5000    1st Qu.:   11.00       
    ##  Median :  3.000     Median :  0.5000    Median :   38.00       
    ##  Mean   :  3.607     Mean   :  0.7868    Mean   :  175.90       
    ##  3rd Qu.:  3.000     3rd Qu.:  1.0000    3rd Qu.:  107.23       
    ##  Max.   :728.000     Max.   : 24.0000    Max.   :73077.33       
    ##  NA's   :4           NA's   :4           NA's   :4              
    ##  AverageCoreServiceHours CoreRecordsRate   CoreTotalKM      
    ##  Min.   : 0.250          Min.   :0.0000   Min.   :    0.00  
    ##  1st Qu.: 1.016          1st Qu.:1.0000   1st Qu.:   22.99  
    ##  Median : 1.464          Median :1.0000   Median :  130.47  
    ##  Mean   : 1.655          Mean   :0.9567   Mean   :  634.03  
    ##  3rd Qu.: 1.875          3rd Qu.:1.0000   3rd Qu.:  442.86  
    ##  Max.   :24.000          Max.   :1.0000   Max.   :44144.31  
    ##  NA's   :4                                                  
    ##  FirstCoreServiceDelayDays FirstCoreServiceHours LastCoreServiceHours
    ##  Min.   : -995.0           Min.   :0             Min.   :0           
    ##  1st Qu.:    0.0           1st Qu.:0             1st Qu.:0           
    ##  Median :    3.0           Median :0             Median :0           
    ##  Mean   :  156.4           Mean   :0             Mean   :0           
    ##  3rd Qu.:    8.0           3rd Qu.:0             3rd Qu.:0           
    ##  Max.   :32153.0           Max.   :0             Max.   :0           
    ##                                                                      
    ##  NoneCoreProgramsNums NoneCoreRecordNums TotalNoneCoreProgramHours
    ##  Min.   :1.000        Min.   :  1.00     Min.   :   0.333         
    ##  1st Qu.:1.000        1st Qu.:  6.00     1st Qu.:   8.562         
    ##  Median :1.000        Median : 27.50     Median :  36.500         
    ##  Mean   :1.027        Mean   : 43.66     Mean   :  60.305         
    ##  3rd Qu.:1.000        3rd Qu.: 65.00     3rd Qu.:  87.688         
    ##  Max.   :2.000        Max.   :195.00     Max.   :1710.983         
    ##  NA's   :4908         NA's   :4908       NA's   :4908             
    ##  MaxNoneCoreProgramHours MinNoneCoreProgramHours
    ##  Min.   : 0.333          Min.   : 0.000         
    ##  1st Qu.: 1.500          1st Qu.: 0.500         
    ##  Median : 2.500          Median : 0.500         
    ##  Mean   : 2.988          Mean   : 1.102         
    ##  3rd Qu.: 3.000          3rd Qu.: 1.000         
    ##  Max.   :24.000          Max.   :24.000         
    ##  NA's   :4908            NA's   :4908           
    ##  AverageNoneCoreProgramHours AverageNoneCoreServiceHours
    ##  Min.   :  0.333             Min.   : 0.333             
    ##  1st Qu.:  8.375             1st Qu.: 1.000             
    ##  Median : 36.250             Median : 1.141             
    ##  Mean   : 55.187             Mean   : 1.701             
    ##  3rd Qu.: 86.812             3rd Qu.: 1.500             
    ##  Max.   :855.492             Max.   :24.000             
    ##  NA's   :4908                NA's   :4908               
    ##  NoneCoreRecordsRate NoneCoreTotalKM   FirstNoneCoreServiceDelayDays
    ##  Min.   :0           Min.   :   0.00   Min.   : -20.00              
    ##  1st Qu.:0           1st Qu.:  12.09   1st Qu.:   0.00              
    ##  Median :0           Median :  70.55   Median :   1.00              
    ##  Mean   :0           Mean   : 158.19   Mean   :  20.22              
    ##  3rd Qu.:0           3rd Qu.: 224.37   3rd Qu.:   3.00              
    ##  Max.   :0           Max.   :1262.18   Max.   :2905.00              
    ##  NA's   :4908        NA's   :4908      NA's   :4908                 
    ##  FirstNoneCoreServiceHours LastNoneCoreServiceHours
    ##  Min.   :0                 Min.   :0               
    ##  1st Qu.:0                 1st Qu.:0               
    ##  Median :0                 Median :0               
    ##  Mean   :0                 Mean   :0               
    ##  3rd Qu.:0                 3rd Qu.:0               
    ##  Max.   :0                 Max.   :0               
    ##  NA's   :4908              NA's   :4908            
    ##      FrequentschedStatusGroup MostUsedBillingGrade MostUsedPayGrade
    ##  Active          :4641        BGrade1:3030         PGrade2:2927    
    ##  Cancelled       :  58        BGrade2:1170         PGrade3:1130    
    ##  KincareInitiated:  21        BGrade3: 506         PGrade1: 487    
    ##  NA's            : 410        BGrade4:   2         PGrade6: 208    
    ##                               BGrade5:  91         PGrade5: 108    
    ##                               BGrade6: 260         (Other):   6    
    ##                               BGrade9:  71         NA's   : 264    
    ##  RespiteNeedsFlag DANeedsFlag NCNeedsFlag PCNeedsFlag SocialNeedsFlag
    ##  N   :4500        N   :1484   N   :4711   N   :3790   N   :3943      
    ##  Y   : 448        Y   :3464   Y   : 237   Y   :1158   Y   :1005      
    ##  NA's: 182        NA's: 182   NA's: 182   NA's: 182   NA's: 182      
    ##                                                                      
    ##                                                                      
    ##                                                                      
    ##                                                                      
    ##  TransportNeedsFlag RequiredWorkersFlag PreferredWorkersFlag
    ##  N   :4473          N   :4733           N   :4470           
    ##  Y   : 475          Y   : 215           Y   : 478           
    ##  NA's: 182          NA's: 182           NA's: 182           
    ##                                                             
    ##                                                             
    ##                                                             
    ##                                                             
    ##  Client_Programs_count_at_observation_cutoff        default_contract_group
    ##  Min.   :0.0000                              CHSP              :2844      
    ##  1st Qu.:0.0000                              Disability        :  34      
    ##  Median :1.0000                              DVA/VHC           :   3      
    ##  Mean   :0.6801                              Package           : 993      
    ##  3rd Qu.:1.0000                              Private/Commercial: 914      
    ##  Max.   :8.0000                              TransPac          : 160      
    ##  NA's   :182                                 NA's              : 182      
    ##  TotalDaysWithKincare_all_Programs complainttier Issues_Raised    
    ##  Min.   :   0.0                    Tier1:  47    Min.   : 0.0000  
    ##  1st Qu.:  83.0                    Tier2:  18    1st Qu.: 0.0000  
    ##  Median : 223.0                    Tier3:   5    Median : 0.0000  
    ##  Mean   : 459.5                    Tier4:   1    Mean   : 0.7667  
    ##  3rd Qu.: 518.0                    NA's :5059    3rd Qu.: 1.0000  
    ##  Max.   :5939.0                                  Max.   :33.0000  
    ##  NA's   :2857                                    NA's   :236      
    ##  Issues_Requiring_Action Escalated_Issues  Closed_Issues    
    ##  Min.   :0.00000         Min.   :0.00000   Min.   : 0.0000  
    ##  1st Qu.:0.00000         1st Qu.:0.00000   1st Qu.: 0.0000  
    ##  Median :0.00000         Median :0.00000   Median : 0.0000  
    ##  Mean   :0.04373         Mean   :0.06702   Mean   : 0.7667  
    ##  3rd Qu.:0.00000         3rd Qu.:0.00000   3rd Qu.: 1.0000  
    ##  Max.   :4.00000         Max.   :6.00000   Max.   :33.0000  
    ##  NA's   :236             NA's   :236       NA's   :236      
    ##  Client_Initiated_Cancellations Kincare_Initiated_Cancellations
    ##  Min.   : 0.000                 Min.   : 0.0000                
    ##  1st Qu.: 0.000                 1st Qu.: 0.0000                
    ##  Median : 1.000                 Median : 0.0000                
    ##  Mean   : 2.242                 Mean   : 0.5646                
    ##  3rd Qu.: 3.000                 3rd Qu.: 1.0000                
    ##  Max.   :46.000                 Max.   :25.0000                
    ##  NA's   :236                    NA's   :236                    
    ##  Canned_Appointments late_first_service_count_all_programs
    ##  Min.   :  0.000     Min.   :1.000                        
    ##  1st Qu.:  0.000     1st Qu.:1.000                        
    ##  Median :  1.000     Median :1.000                        
    ##  Mean   :  3.237     Mean   :1.225                        
    ##  3rd Qu.:  3.000     3rd Qu.:1.000                        
    ##  Max.   :129.000     Max.   :7.000                        
    ##  NA's   :236         NA's   :3379                         
    ##  avg_first_service_days_all_programs   HCW_Ratio           Label     
    ##  Min.   :    1.0                     Min.   :  0.000   Churn  :1026  
    ##  1st Qu.:    4.0                     1st Qu.:  2.000   NoChurn:4104  
    ##  Median :   12.0                     Median :  5.000                 
    ##  Mean   :  486.2                     Mean   :  9.532                 
    ##  3rd Qu.:  628.0                     3rd Qu.: 12.000                 
    ##  Max.   :32153.0                     Max.   :133.000                 
    ##  NA's   :3379                                                        
    ##      role          
    ##  Length:5130       
    ##  Class :character  
    ##  Mode  :character  
    ##                    
    ##                    
    ##                    
    ## 

``` r
# Data Cleansing
var_names <-  selectRawDataVariables (raw.data)
d1 <- cleanRawData(var_names)
print ('Summary of Cleansed Raw Data')
```

    ## [1] "Summary of Cleansed Raw Data"

``` r
summary(d1)
```

    ##  HOME_STR_state   ClientType      Grade      SmokerAccepted GenderRequired
    ##  ACT: 288       HAC    :3838   Grade1:3614   No :2740       Either:2405   
    ##  NSW:2950       EAC    : 310   Grade2:1005   Yes:2390       Female:2619   
    ##  QLD: 547       Com    : 304   Grade3: 276                  Male  : 106   
    ##  SA : 982       CCP    : 163   Grade4:  16                                
    ##  VIC:  95       Dom    : 136   Grade5:  55                                
    ##  WA : 268       Pri    : 102   Grade6: 164                                
    ##                 (Other): 277                                              
    ##  AgeAtCreation    AllRecordsNums   CoreRecordNums   TotalCoreProgramHours
    ##  Min.   : 31.00   Min.   :  1.00   Min.   :  1.00   Min.   :  0.25       
    ##  1st Qu.: 73.00   1st Qu.:  9.00   1st Qu.:  9.00   1st Qu.: 12.50       
    ##  Median : 82.00   Median : 35.00   Median : 34.00   Median : 48.50       
    ##  Mean   : 79.12   Mean   : 83.25   Mean   : 81.46   Mean   :115.39       
    ##  3rd Qu.: 87.00   3rd Qu.:100.00   3rd Qu.: 99.00   3rd Qu.:143.50       
    ##  Max.   :115.00   Max.   :373.00   Max.   :369.00   Max.   :536.50       
    ##                                                                          
    ##  MaxCoreProgramHours MinCoreProgramHours AverageCoreProgramHours
    ##  Min.   :0.250       Min.   :0.0000      Min.   :  0.25         
    ##  1st Qu.:2.000       1st Qu.:0.5000      1st Qu.: 11.00         
    ##  Median :3.000       Median :0.5000      Median : 38.00         
    ##  Mean   :2.916       Mean   :0.7507      Mean   : 88.74         
    ##  3rd Qu.:3.000       3rd Qu.:1.0000      3rd Qu.:107.49         
    ##  Max.   :6.000       Max.   :2.5000      Max.   :396.96         
    ##                                                                 
    ##  AverageCoreServiceHours  CoreTotalKM      FirstCoreServiceDelayDays
    ##  Min.   :0.250           Min.   :   0.00   Min.   : 0.000           
    ##  1st Qu.:1.016           1st Qu.:  22.99   1st Qu.: 0.000           
    ##  Median :1.464           Median : 130.47   Median : 3.000           
    ##  Mean   :1.532           Mean   : 360.14   Mean   : 7.554           
    ##  3rd Qu.:1.875           3rd Qu.: 442.86   3rd Qu.: 8.000           
    ##  Max.   :4.450           Max.   :1702.46   Max.   :32.000           
    ##                                                                     
    ##      FrequentschedStatusGroup MostUsedBillingGrade MostUsedPayGrade
    ##  Active          :5051        BGrade1:3030         PGrade0:   2    
    ##  Cancelled       :  58        BGrade2:1170         PGrade1: 487    
    ##  KincareInitiated:  21        BGrade3: 506         PGrade2:3191    
    ##                               BGrade4:   2         PGrade3:1130    
    ##                               BGrade5:  91         PGrade4:   4    
    ##                               BGrade6: 260         PGrade5: 108    
    ##                               BGrade9:  71         PGrade6: 208    
    ##  RespiteNeedsFlag DANeedsFlag NCNeedsFlag PCNeedsFlag SocialNeedsFlag
    ##  N:4682           N:1484      N:4893      N:3972      N:4125         
    ##  Y: 448           Y:3646      Y: 237      Y:1158      Y:1005         
    ##                                                                      
    ##                                                                      
    ##                                                                      
    ##                                                                      
    ##                                                                      
    ##  TransportNeedsFlag RequiredWorkersFlag PreferredWorkersFlag
    ##  N:4655             N:4915              N:4652              
    ##  Y: 475             Y: 215              Y: 478              
    ##                                                             
    ##                                                             
    ##                                                             
    ##                                                             
    ##                                                             
    ##  Client_Programs_count_at_observation_cutoff        default_contract_group
    ##  Min.   :0.0000                              CHSP              :3026      
    ##  1st Qu.:0.0000                              Disability        :  34      
    ##  Median :0.6801                              DVA/VHC           :   3      
    ##  Mean   :0.6775                              Package           : 993      
    ##  3rd Qu.:1.0000                              Private/Commercial: 914      
    ##  Max.   :4.0000                              TransPac          : 160      
    ##                                                                           
    ##  Issues_Raised    Closed_Issues    Client_Initiated_Cancellations
    ##  Min.   :0.0000   Min.   :0.0000   Min.   : 0.000                
    ##  1st Qu.:0.0000   1st Qu.:0.0000   1st Qu.: 0.000                
    ##  Median :0.0000   Median :0.0000   Median : 1.000                
    ##  Mean   :0.7058   Mean   :0.7058   Mean   : 2.054                
    ##  3rd Qu.:1.0000   3rd Qu.:1.0000   3rd Qu.: 3.000                
    ##  Max.   :4.0000   Max.   :4.0000   Max.   :12.000                
    ##                                                                  
    ##  Kincare_Initiated_Cancellations Canned_Appointments   HCW_Ratio     
    ##  Min.   :0.0000                  Min.   : 0.000      Min.   : 0.000  
    ##  1st Qu.:0.0000                  1st Qu.: 0.000      1st Qu.: 2.000  
    ##  Median :0.0000                  Median : 1.000      Median : 5.000  
    ##  Mean   :0.4995                  Mean   : 2.620      Mean   : 8.989  
    ##  3rd Qu.:1.0000                  3rd Qu.: 3.237      3rd Qu.:12.000  
    ##  Max.   :4.0000                  Max.   :12.947      Max.   :42.000  
    ##                                                                      
    ##  Label        role          
    ##  0:4104   Length:5130       
    ##  1:1026   Class :character  
    ##           Mode  :character  
    ##                             
    ##                             
    ##                             
    ## 

``` r
#=========================
#FEATURE SELECTION
ndxRole <- which(names(d1) == 'role')
ndxLabel <- which(names(d1) == 'Label')

d2 <- d1[, -ndxRole,-ndxLabel]
d2.scaled <- scaleNumbers(d2)

set.seed(1)
mod_d2_glm <-
  glm(Label ~ ., family = binomial(link = "logit"), data = d2.scaled)
mod_d2_glm.sumry <- summary(mod_d2_glm,signif.stars=TRUE)

sel_pValue <- .03
print (paste ('Logit significant variable p-Value threshold used is ', sel_pValue))
```

    ## [1] "Logit significant variable p-Value threshold used is  0.03"

``` r
tmp <- mod_d2_glm.sumry$coefficients[,4]
sel_var_logit <- subset(tmp, tmp <= sel_pValue)
print (sort(sel_var_logit)) 
```

    ##                               AgeAtCreation 
    ##                                7.664414e-17 
    ##                               Issues_Raised 
    ##                                1.482545e-16 
    ##                      MostUsedBillingGrade.L 
    ##                                4.072909e-04 
    ##                         MaxCoreProgramHours 
    ##                                2.671882e-03 
    ##                               ClientTypeYou 
    ##                                3.084973e-03 
    ##              Client_Initiated_Cancellations 
    ##                                5.610230e-03 
    ##    default_contract_groupPrivate/Commercial 
    ##                                7.502689e-03 
    ## Client_Programs_count_at_observation_cutoff 
    ##                                7.854017e-03 
    ##                           HOME_STR_stateVIC 
    ##                                1.011449e-02 
    ##                           HOME_STR_stateNSW 
    ##                                1.272792e-02 
    ##                         MinCoreProgramHours 
    ##                                2.865223e-02

``` r
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
RFmtry_param <-  round(sqrt(ncol(d2[,-ndxLabel])),0)

print (paste('tuneRF()  mtry for feature selectio is  ', RFmtry_param))
```

    ## [1] "tuneRF()  mtry for feature selectio is   6"

``` r
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
PlotTitle <- c('No Churn Accuracy','Churn Accuracy', 'Churn/No Churn Accuracy','Gini Index')
for (i in 1:4)
  plot(sort(mod_d2_rf$importance[,i], dec = TRUE), type="h",main=paste(PlotTitle[i]),xlab='Variable Index', ylab='')
```

![](kc_client_churn_FINAL_F_files/figure-markdown_github/cars-1.png)

``` r
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
```

![](kc_client_churn_FINAL_F_files/figure-markdown_github/cars-2.png)

``` r
# After inspecting the importance plot, choose top 15 variables sorted by mean decrease in accuracy for all class
nRFSelVars <- 15

tmp <- importance(mod_d2_rf, type=k_RF_var_imp_type, class=NULL, scale=FALSE)
tmp <- tmp[order(tmp[,1],decreasing=TRUE),]
print (tmp[1:nRFSelVars])
```

    ##                       TotalCoreProgramHours 
    ##                                 0.047619321 
    ##                              AllRecordsNums 
    ##                                 0.041839484 
    ##                              CoreRecordNums 
    ##                                 0.037308887 
    ##                     AverageCoreProgramHours 
    ##                                 0.036594101 
    ##                                 CoreTotalKM 
    ##                                 0.020220766 
    ##                                   HCW_Ratio 
    ##                                 0.012428966 
    ##                              HOME_STR_state 
    ##                                 0.009577546 
    ##                               AgeAtCreation 
    ##                                 0.009527303 
    ##                         MaxCoreProgramHours 
    ##                                 0.008757183 
    ## Client_Programs_count_at_observation_cutoff 
    ##                                 0.008756473 
    ##                     AverageCoreServiceHours 
    ##                                 0.008254840 
    ##                        MostUsedBillingGrade 
    ##                                 0.007851558 
    ##                         Canned_Appointments 
    ##                                 0.006691456 
    ##                      default_contract_group 
    ##                                 0.006533013 
    ##                               Closed_Issues 
    ##                                 0.006304662

``` r
sel_RF_imp_vars <- names(tmp[1:nRFSelVars])

#Combine the RF and GLM features.
model_features <- union (GLM_signficant_features, sel_RF_imp_vars)
model_features <- unique(model_features)

d2 <- d2[model_features]

#Convert  factors into integers
x <- sapply(d2, function(x) class(x))
factor_vars <- x[x == "factor" | length (x) > 1 ]

for (var in names(factor_vars)) {
  d2[[var]] <- as.numeric (d2[[var]])
}

for (var in names(d2)) {
  d2[[var]] <- scale(d2[[var]], center = TRUE, scale = TRUE)
}

# Perform Correlation Analysis 
d2_corr <- cor(d2, method = "pearson")
print (d2_corr)
```

    ##                                             Issues_Raised AgeAtCreation
    ## Issues_Raised                                  1.00000000   0.041385755
    ## AgeAtCreation                                  0.04138576   1.000000000
    ## MostUsedBillingGrade                          -0.05358913  -0.089335065
    ## ClientType                                     0.02143328  -0.072990557
    ## MaxCoreProgramHours                            0.16177581  -0.022564648
    ## Client_Initiated_Cancellations                 0.38682985  -0.023232266
    ## default_contract_group                         0.07596119   0.057363367
    ## Client_Programs_count_at_observation_cutoff    0.26856404   0.010620043
    ## HOME_STR_state                                -0.07407358   0.071152869
    ## PCNeedsFlag                                    0.28126717   0.106122026
    ## TotalCoreProgramHours                          0.28152452  -0.041141660
    ## AllRecordsNums                                 0.33769902   0.022369768
    ## CoreRecordNums                                 0.32642736   0.016144733
    ## AverageCoreProgramHours                        0.22358846  -0.059472419
    ## CoreTotalKM                                    0.29003361   0.027116618
    ## HCW_Ratio                                      0.30825678   0.040705630
    ## AverageCoreServiceHours                       -0.07497669  -0.115468177
    ## Canned_Appointments                            0.44010194   0.004915774
    ## Closed_Issues                                  1.00000000   0.041385755
    ##                                             MostUsedBillingGrade
    ## Issues_Raised                                        -0.05358913
    ## AgeAtCreation                                        -0.08933506
    ## MostUsedBillingGrade                                  1.00000000
    ## ClientType                                            0.05541798
    ## MaxCoreProgramHours                                  -0.25037664
    ## Client_Initiated_Cancellations                       -0.09568631
    ## default_contract_group                                0.20372788
    ## Client_Programs_count_at_observation_cutoff          -0.11917305
    ## HOME_STR_state                                        0.18037771
    ## PCNeedsFlag                                           0.11012551
    ## TotalCoreProgramHours                                -0.05987676
    ## AllRecordsNums                                       -0.03549782
    ## CoreRecordNums                                       -0.03561334
    ## AverageCoreProgramHours                              -0.06933419
    ## CoreTotalKM                                          -0.03884645
    ## HCW_Ratio                                            -0.14422390
    ## AverageCoreServiceHours                              -0.11035679
    ## Canned_Appointments                                  -0.02038763
    ## Closed_Issues                                        -0.05358913
    ##                                               ClientType
    ## Issues_Raised                                0.021433282
    ## AgeAtCreation                               -0.072990557
    ## MostUsedBillingGrade                         0.055417978
    ## ClientType                                   1.000000000
    ## MaxCoreProgramHours                          0.070040275
    ## Client_Initiated_Cancellations               0.013343895
    ## default_contract_group                      -0.273237595
    ## Client_Programs_count_at_observation_cutoff -0.068846470
    ## HOME_STR_state                              -0.200269152
    ## PCNeedsFlag                                 -0.034278932
    ## TotalCoreProgramHours                       -0.125084567
    ## AllRecordsNums                              -0.126369491
    ## CoreRecordNums                              -0.135408579
    ## AverageCoreProgramHours                     -0.144832836
    ## CoreTotalKM                                 -0.111883002
    ## HCW_Ratio                                   -0.048897326
    ## AverageCoreServiceHours                      0.029086274
    ## Canned_Appointments                          0.008640219
    ## Closed_Issues                                0.021433282
    ##                                             MaxCoreProgramHours
    ## Issues_Raised                                        0.16177581
    ## AgeAtCreation                                       -0.02256465
    ## MostUsedBillingGrade                                -0.25037664
    ## ClientType                                           0.07004027
    ## MaxCoreProgramHours                                  1.00000000
    ## Client_Initiated_Cancellations                       0.20369069
    ## default_contract_group                              -0.17737746
    ## Client_Programs_count_at_observation_cutoff          0.22794693
    ## HOME_STR_state                                      -0.18072055
    ## PCNeedsFlag                                          0.03854789
    ## TotalCoreProgramHours                                0.38149368
    ## AllRecordsNums                                       0.25276731
    ## CoreRecordNums                                       0.25386393
    ## AverageCoreProgramHours                              0.33240847
    ## CoreTotalKM                                          0.23314490
    ## HCW_Ratio                                            0.26001696
    ## AverageCoreServiceHours                              0.57466572
    ## Canned_Appointments                                  0.22625475
    ## Closed_Issues                                        0.16177581
    ##                                             Client_Initiated_Cancellations
    ## Issues_Raised                                                   0.38682985
    ## AgeAtCreation                                                  -0.02323227
    ## MostUsedBillingGrade                                           -0.09568631
    ## ClientType                                                      0.01334389
    ## MaxCoreProgramHours                                             0.20369069
    ## Client_Initiated_Cancellations                                  1.00000000
    ## default_contract_group                                          0.10154183
    ## Client_Programs_count_at_observation_cutoff                     0.22004363
    ## HOME_STR_state                                                 -0.05789804
    ## PCNeedsFlag                                                     0.18117461
    ## TotalCoreProgramHours                                           0.37317144
    ## AllRecordsNums                                                  0.40693584
    ## CoreRecordNums                                                  0.39694951
    ## AverageCoreProgramHours                                         0.33924449
    ## CoreTotalKM                                                     0.38905006
    ## HCW_Ratio                                                       0.42339423
    ## AverageCoreServiceHours                                        -0.05541873
    ## Canned_Appointments                                             0.79422047
    ## Closed_Issues                                                   0.38682985
    ##                                             default_contract_group
    ## Issues_Raised                                           0.07596119
    ## AgeAtCreation                                           0.05736337
    ## MostUsedBillingGrade                                    0.20372788
    ## ClientType                                             -0.27323759
    ## MaxCoreProgramHours                                    -0.17737746
    ## Client_Initiated_Cancellations                          0.10154183
    ## default_contract_group                                  1.00000000
    ## Client_Programs_count_at_observation_cutoff             0.16588365
    ## HOME_STR_state                                          0.30838450
    ## PCNeedsFlag                                             0.29671367
    ## TotalCoreProgramHours                                   0.23964151
    ## AllRecordsNums                                          0.30260524
    ## CoreRecordNums                                          0.27845439
    ## AverageCoreProgramHours                                 0.22764737
    ## CoreTotalKM                                             0.25266595
    ## HCW_Ratio                                               0.19525978
    ## AverageCoreServiceHours                                -0.16395333
    ## Canned_Appointments                                     0.17819327
    ## Closed_Issues                                           0.07596119
    ##                                             Client_Programs_count_at_observation_cutoff
    ## Issues_Raised                                                                0.26856404
    ## AgeAtCreation                                                                0.01062004
    ## MostUsedBillingGrade                                                        -0.11917305
    ## ClientType                                                                  -0.06884647
    ## MaxCoreProgramHours                                                          0.22794693
    ## Client_Initiated_Cancellations                                               0.22004363
    ## default_contract_group                                                       0.16588365
    ## Client_Programs_count_at_observation_cutoff                                  1.00000000
    ## HOME_STR_state                                                              -0.02333375
    ## PCNeedsFlag                                                                  0.17444072
    ## TotalCoreProgramHours                                                        0.50402719
    ## AllRecordsNums                                                               0.50987739
    ## CoreRecordNums                                                               0.50397413
    ## AverageCoreProgramHours                                                      0.40379057
    ## CoreTotalKM                                                                  0.45845090
    ## HCW_Ratio                                                                    0.44849892
    ## AverageCoreServiceHours                                                     -0.02813402
    ## Canned_Appointments                                                          0.29447987
    ## Closed_Issues                                                                0.26856404
    ##                                             HOME_STR_state PCNeedsFlag
    ## Issues_Raised                                 -0.074073582  0.28126717
    ## AgeAtCreation                                  0.071152869  0.10612203
    ## MostUsedBillingGrade                           0.180377706  0.11012551
    ## ClientType                                    -0.200269152 -0.03427893
    ## MaxCoreProgramHours                           -0.180720552  0.03854789
    ## Client_Initiated_Cancellations                -0.057898041  0.18117461
    ## default_contract_group                         0.308384499  0.29671367
    ## Client_Programs_count_at_observation_cutoff   -0.023333747  0.17444072
    ## HOME_STR_state                                 1.000000000  0.05738219
    ## PCNeedsFlag                                    0.057382195  1.00000000
    ## TotalCoreProgramHours                          0.060344768  0.35814756
    ## AllRecordsNums                                 0.082491942  0.45515821
    ## CoreRecordNums                                 0.088894075  0.43992899
    ## AverageCoreProgramHours                        0.066899216  0.30879191
    ## CoreTotalKM                                    0.096959364  0.37219951
    ## HCW_Ratio                                     -0.002210427  0.33590257
    ## AverageCoreServiceHours                       -0.132536094 -0.18842895
    ## Canned_Appointments                           -0.046833176  0.34911641
    ## Closed_Issues                                 -0.074073582  0.28126717
    ##                                             TotalCoreProgramHours
    ## Issues_Raised                                          0.28152452
    ## AgeAtCreation                                         -0.04114166
    ## MostUsedBillingGrade                                  -0.05987676
    ## ClientType                                            -0.12508457
    ## MaxCoreProgramHours                                    0.38149368
    ## Client_Initiated_Cancellations                         0.37317144
    ## default_contract_group                                 0.23964151
    ## Client_Programs_count_at_observation_cutoff            0.50402719
    ## HOME_STR_state                                         0.06034477
    ## PCNeedsFlag                                            0.35814756
    ## TotalCoreProgramHours                                  1.00000000
    ## AllRecordsNums                                         0.91641135
    ## CoreRecordNums                                         0.92151465
    ## AverageCoreProgramHours                                0.95138676
    ## CoreTotalKM                                            0.79390920
    ## HCW_Ratio                                              0.71688622
    ## AverageCoreServiceHours                                0.13528805
    ## Canned_Appointments                                    0.54747754
    ## Closed_Issues                                          0.28152452
    ##                                             AllRecordsNums CoreRecordNums
    ## Issues_Raised                                   0.33769902     0.32642736
    ## AgeAtCreation                                   0.02236977     0.01614473
    ## MostUsedBillingGrade                           -0.03549782    -0.03561334
    ## ClientType                                     -0.12636949    -0.13540858
    ## MaxCoreProgramHours                             0.25276731     0.25386393
    ## Client_Initiated_Cancellations                  0.40693584     0.39694951
    ## default_contract_group                          0.30260524     0.27845439
    ## Client_Programs_count_at_observation_cutoff     0.50987739     0.50397413
    ## HOME_STR_state                                  0.08249194     0.08889407
    ## PCNeedsFlag                                     0.45515821     0.43992899
    ## TotalCoreProgramHours                           0.91641135     0.92151465
    ## AllRecordsNums                                  1.00000000     0.99531222
    ## CoreRecordNums                                  0.99531222     1.00000000
    ## AverageCoreProgramHours                         0.86571733     0.87174311
    ## CoreTotalKM                                     0.84832106     0.85339675
    ## HCW_Ratio                                       0.75712848     0.75886705
    ## AverageCoreServiceHours                        -0.11161325    -0.11012873
    ## Canned_Appointments                             0.59720566     0.58673811
    ## Closed_Issues                                   0.33769902     0.32642736
    ##                                             AverageCoreProgramHours
    ## Issues_Raised                                            0.22358846
    ## AgeAtCreation                                           -0.05947242
    ## MostUsedBillingGrade                                    -0.06933419
    ## ClientType                                              -0.14483284
    ## MaxCoreProgramHours                                      0.33240847
    ## Client_Initiated_Cancellations                           0.33924449
    ## default_contract_group                                   0.22764737
    ## Client_Programs_count_at_observation_cutoff              0.40379057
    ## HOME_STR_state                                           0.06689922
    ## PCNeedsFlag                                              0.30879191
    ## TotalCoreProgramHours                                    0.95138676
    ## AllRecordsNums                                           0.86571733
    ## CoreRecordNums                                           0.87174311
    ## AverageCoreProgramHours                                  1.00000000
    ## CoreTotalKM                                              0.75670368
    ## HCW_Ratio                                                0.67965538
    ## AverageCoreServiceHours                                  0.15359100
    ## Canned_Appointments                                      0.49545487
    ## Closed_Issues                                            0.22358846
    ##                                             CoreTotalKM    HCW_Ratio
    ## Issues_Raised                                0.29003361  0.308256776
    ## AgeAtCreation                                0.02711662  0.040705630
    ## MostUsedBillingGrade                        -0.03884645 -0.144223904
    ## ClientType                                  -0.11188300 -0.048897326
    ## MaxCoreProgramHours                          0.23314490  0.260016962
    ## Client_Initiated_Cancellations               0.38905006  0.423394234
    ## default_contract_group                       0.25266595  0.195259785
    ## Client_Programs_count_at_observation_cutoff  0.45845090  0.448498920
    ## HOME_STR_state                               0.09695936 -0.002210427
    ## PCNeedsFlag                                  0.37219951  0.335902568
    ## TotalCoreProgramHours                        0.79390920  0.716886220
    ## AllRecordsNums                               0.84832106  0.757128485
    ## CoreRecordNums                               0.85339675  0.758867050
    ## AverageCoreProgramHours                      0.75670368  0.679655375
    ## CoreTotalKM                                  1.00000000  0.771037879
    ## HCW_Ratio                                    0.77103788  1.000000000
    ## AverageCoreServiceHours                     -0.11579440 -0.125985798
    ## Canned_Appointments                          0.54168621  0.564234653
    ## Closed_Issues                                0.29003361  0.308256776
    ##                                             AverageCoreServiceHours
    ## Issues_Raised                                           -0.07497669
    ## AgeAtCreation                                           -0.11546818
    ## MostUsedBillingGrade                                    -0.11035679
    ## ClientType                                               0.02908627
    ## MaxCoreProgramHours                                      0.57466572
    ## Client_Initiated_Cancellations                          -0.05541873
    ## default_contract_group                                  -0.16395333
    ## Client_Programs_count_at_observation_cutoff             -0.02813402
    ## HOME_STR_state                                          -0.13253609
    ## PCNeedsFlag                                             -0.18842895
    ## TotalCoreProgramHours                                    0.13528805
    ## AllRecordsNums                                          -0.11161325
    ## CoreRecordNums                                          -0.11012873
    ## AverageCoreProgramHours                                  0.15359100
    ## CoreTotalKM                                             -0.11579440
    ## HCW_Ratio                                               -0.12598580
    ## AverageCoreServiceHours                                  1.00000000
    ## Canned_Appointments                                     -0.08012874
    ## Closed_Issues                                           -0.07497669
    ##                                             Canned_Appointments
    ## Issues_Raised                                       0.440101942
    ## AgeAtCreation                                       0.004915774
    ## MostUsedBillingGrade                               -0.020387634
    ## ClientType                                          0.008640219
    ## MaxCoreProgramHours                                 0.226254753
    ## Client_Initiated_Cancellations                      0.794220473
    ## default_contract_group                              0.178193275
    ## Client_Programs_count_at_observation_cutoff         0.294479875
    ## HOME_STR_state                                     -0.046833176
    ## PCNeedsFlag                                         0.349116412
    ## TotalCoreProgramHours                               0.547477545
    ## AllRecordsNums                                      0.597205657
    ## CoreRecordNums                                      0.586738105
    ## AverageCoreProgramHours                             0.495454874
    ## CoreTotalKM                                         0.541686212
    ## HCW_Ratio                                           0.564234653
    ## AverageCoreServiceHours                            -0.080128740
    ## Canned_Appointments                                 1.000000000
    ## Closed_Issues                                       0.440101942
    ##                                             Closed_Issues
    ## Issues_Raised                                  1.00000000
    ## AgeAtCreation                                  0.04138576
    ## MostUsedBillingGrade                          -0.05358913
    ## ClientType                                     0.02143328
    ## MaxCoreProgramHours                            0.16177581
    ## Client_Initiated_Cancellations                 0.38682985
    ## default_contract_group                         0.07596119
    ## Client_Programs_count_at_observation_cutoff    0.26856404
    ## HOME_STR_state                                -0.07407358
    ## PCNeedsFlag                                    0.28126717
    ## TotalCoreProgramHours                          0.28152452
    ## AllRecordsNums                                 0.33769902
    ## CoreRecordNums                                 0.32642736
    ## AverageCoreProgramHours                        0.22358846
    ## CoreTotalKM                                    0.29003361
    ## HCW_Ratio                                      0.30825678
    ## AverageCoreServiceHours                       -0.07497669
    ## Canned_Appointments                            0.44010194
    ## Closed_Issues                                  1.00000000

``` r
write.csv (d2_corr, file = "kc_correlation_pearson_model_features.csv")

# By manual inspection of the correlation map, following  variables are to be removed
# due to collinearity (Pearson correlation  abs(correlation coefficient) >=  0.814

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
print (model_features)
```

    ##  [1] "Issues_Raised"                              
    ##  [2] "AgeAtCreation"                              
    ##  [3] "MostUsedBillingGrade"                       
    ##  [4] "ClientType"                                 
    ##  [5] "Client_Initiated_Cancellations"             
    ##  [6] "default_contract_group"                     
    ##  [7] "Client_Programs_count_at_observation_cutoff"
    ##  [8] "HOME_STR_state"                             
    ##  [9] "PCNeedsFlag"                                
    ## [10] "AverageCoreProgramHours"                    
    ## [11] "HCW_Ratio"

``` r
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
```

    ## 
    ## Call:
    ## glm(formula = Label ~ ., family = binomial(link = "logit"), data = train.scaled)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -2.2030  -0.6639  -0.4741  -0.1776   3.3285  
    ## 
    ## Coefficients:
    ##                                               Estimate Std. Error z value
    ## (Intercept)                                    0.43370  145.78538   0.003
    ## Issues_Raised                                  0.34927    0.05183   6.739
    ## AgeAtCreation                                 -0.36461    0.04645  -7.850
    ## MostUsedBillingGrade.L                         0.93084    0.19525   4.767
    ## MostUsedBillingGrade.Q                        -7.24166  445.36776  -0.016
    ## MostUsedBillingGrade.C                        -0.26828    0.20914  -1.283
    ## MostUsedBillingGrade^4                         7.59609  493.38880   0.015
    ## MostUsedBillingGrade^5                         0.58078    0.19720   2.945
    ## MostUsedBillingGrade^6                       -10.04761  671.41712  -0.015
    ## ClientTypeCAC                                  1.68353    1.21489   1.386
    ## ClientTypeCCP                                  1.10212    1.16913   0.943
    ## ClientTypeCom                                 -0.16902    1.13103  -0.149
    ## ClientTypeDem                                -13.38780  350.41535  -0.038
    ## ClientTypeDis                                 -0.59841    1.44835  -0.413
    ## ClientTypeDom                                  1.21376    1.15942   1.047
    ## ClientTypeDVA                                 -3.40964 1725.34418  -0.002
    ## ClientTypeEAC                                  1.53377    1.15688   1.326
    ## ClientTypeHAC                                  0.94026    1.12027   0.839
    ## ClientTypeNRC                                -11.87389  390.89921  -0.030
    ## ClientTypeNur                                 -1.02369    1.62273  -0.631
    ## ClientTypePer                                  0.61148    1.22969   0.497
    ## ClientTypePri                                  0.92657    1.15184   0.804
    ## ClientTypeRes                                 -0.06105    1.61001  -0.038
    ## ClientTypeSoc                                  1.95938    1.30036   1.507
    ## ClientTypeTAC                                  0.85389    1.57873   0.541
    ## ClientTypeTCP                                -11.19081  905.04904  -0.012
    ## ClientTypeVHC                                -11.81507  926.62270  -0.013
    ## ClientTypeYou                                  4.16097    1.49018   2.792
    ## Client_Initiated_Cancellations                -0.36419    0.06843  -5.322
    ## default_contract_groupDisability               0.90829    0.88461   1.027
    ## default_contract_groupDVA/VHC                -11.41443  926.62244  -0.012
    ## default_contract_groupPackage                 -0.04747    0.16953  -0.280
    ## default_contract_groupPrivate/Commercial       0.40155    0.14958   2.684
    ## default_contract_groupTransPac                -0.70167    0.37191  -1.887
    ## Client_Programs_count_at_observation_cutoff    0.09586    0.05459   1.756
    ## HOME_STR_stateNSW                             -0.65864    0.25812  -2.552
    ## HOME_STR_stateQLD                             -0.14533    0.28540  -0.509
    ## HOME_STR_stateSA                              -0.07075    0.28029  -0.252
    ## HOME_STR_stateVIC                             -1.25828    0.47128  -2.670
    ## HOME_STR_stateWA                              -0.39709    0.36099  -1.100
    ## PCNeedsFlagY                                   0.18885    0.14038   1.345
    ## AverageCoreProgramHours                       -0.72001    0.09461  -7.610
    ## HCW_Ratio                                     -0.32210    0.09466  -3.403
    ##                                             Pr(>|z|)    
    ## (Intercept)                                 0.997626    
    ## Issues_Raised                               1.60e-11 ***
    ## AgeAtCreation                               4.17e-15 ***
    ## MostUsedBillingGrade.L                      1.87e-06 ***
    ## MostUsedBillingGrade.Q                      0.987027    
    ## MostUsedBillingGrade.C                      0.199576    
    ## MostUsedBillingGrade^4                      0.987716    
    ## MostUsedBillingGrade^5                      0.003228 ** 
    ## MostUsedBillingGrade^6                      0.988060    
    ## ClientTypeCAC                               0.165823    
    ## ClientTypeCCP                               0.345839    
    ## ClientTypeCom                               0.881206    
    ## ClientTypeDem                               0.969524    
    ## ClientTypeDis                               0.679484    
    ## ClientTypeDom                               0.295158    
    ## ClientTypeDVA                               0.998423    
    ## ClientTypeEAC                               0.184913    
    ## ClientTypeHAC                               0.401290    
    ## ClientTypeNRC                               0.975767    
    ## ClientTypeNur                               0.528142    
    ## ClientTypePer                               0.619005    
    ## ClientTypePri                               0.421151    
    ## ClientTypeRes                               0.969750    
    ## ClientTypeSoc                               0.131862    
    ## ClientTypeTAC                               0.588598    
    ## ClientTypeTCP                               0.990135    
    ## ClientTypeVHC                               0.989827    
    ## ClientTypeYou                               0.005234 ** 
    ## Client_Initiated_Cancellations              1.03e-07 ***
    ## default_contract_groupDisability            0.304530    
    ## default_contract_groupDVA/VHC               0.990172    
    ## default_contract_groupPackage               0.779483    
    ## default_contract_groupPrivate/Commercial    0.007265 ** 
    ## default_contract_groupTransPac              0.059204 .  
    ## Client_Programs_count_at_observation_cutoff 0.079069 .  
    ## HOME_STR_stateNSW                           0.010720 *  
    ## HOME_STR_stateQLD                           0.610607    
    ## HOME_STR_stateSA                            0.800722    
    ## HOME_STR_stateVIC                           0.007587 ** 
    ## HOME_STR_stateWA                            0.271331    
    ## PCNeedsFlagY                                0.178552    
    ## AverageCoreProgramHours                     2.74e-14 ***
    ## HCW_Ratio                                   0.000667 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 4133.3  on 4129  degrees of freedom
    ## Residual deviance: 3477.8  on 4087  degrees of freedom
    ## AIC: 3563.8
    ## 
    ## Number of Fisher Scoring iterations: 14

``` r
pr.kc.glm <- predict(kc.glm, newdata = test.scaled, type = 'response')
fitted.results.glm <- ifelse(pr.kc.glm > fitThreshold, 1, 0)
confusion_maxtix <- table(test.scaled$Label, fitted.results.glm)
print.table(confusion_maxtix)
```

    ##    fitted.results.glm
    ##       0   1
    ##   0 782  18
    ##   1 150  50

``` r
kc.glm.eval.results <- computePerformanceMeasures(confusion_maxtix, fScoreB_param)

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
```

    ## mtry = 3  OOB error = 17.92% 
    ## Searching left ...
    ## mtry = 2     OOB error = 17.85% 
    ## 0.004054054 0.01 
    ## Searching right ...
    ## mtry = 4     OOB error = 18.14% 
    ## -0.01216216 0.01

![](kc_client_churn_FINAL_F_files/figure-markdown_github/cars-3.png)

``` r
RFmtry_param <- bestmtry[min(bestmtry[, 2]) == bestmtry[, 2]][1]
#RFmtry_param <-  round(sqrt(ncol(train[,-ndxLabel])),0)
print (paste('RF  model mtry value with least OOB error is ', RFmtry_param))
```

    ## [1] "RF  model mtry value with least OOB error is  2"

``` r
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
```

    ## 
    ## Call:
    ##  randomForest(formula = Label ~ ., data = train, mtry = RFmtry_param,      ntree = RFnTree_param, keep.forest = TRUE, importance = TRUE) 
    ##                Type of random forest: classification
    ##                      Number of trees: 1000
    ## No. of variables tried at each split: 2
    ## 
    ##         OOB estimate of  error rate: 17.7%
    ## Confusion matrix:
    ##      0   1 class.error
    ## 0 3164 140  0.04237288
    ## 1  591 235  0.71549637

``` r
summary(kc.rf)
```

    ##                 Length Class  Mode     
    ## call               7   -none- call     
    ## type               1   -none- character
    ## predicted       4130   factor numeric  
    ## err.rate        3000   -none- numeric  
    ## confusion          6   -none- numeric  
    ## votes           8260   matrix numeric  
    ## oob.times       4130   -none- numeric  
    ## classes            2   -none- character
    ## importance        44   -none- numeric  
    ## importanceSD      33   -none- numeric  
    ## localImportance    0   -none- NULL     
    ## proximity          0   -none- NULL     
    ## ntree              1   -none- numeric  
    ## mtry               1   -none- numeric  
    ## forest            14   -none- list     
    ## y               4130   factor numeric  
    ## test               0   -none- NULL     
    ## inbag              0   -none- NULL     
    ## terms              3   terms  call

``` r
varImpPlot(kc.rf,  sort = TRUE, type = k_RF_var_imp_type, class=NULL, scale = FALSE, main = '')
```

![](kc_client_churn_FINAL_F_files/figure-markdown_github/cars-4.png)

``` r
tmp <- importance(kc.rf, type=k_RF_var_imp_type, class=NULL, scale=FALSE)
tmp <- tmp[order(tmp[,1],decreasing=TRUE),]
print(tmp)
```

    ##                     AverageCoreProgramHours 
    ##                                 0.045056273 
    ##                                   HCW_Ratio 
    ##                                 0.016245890 
    ##                              HOME_STR_state 
    ##                                 0.015536424 
    ##                        MostUsedBillingGrade 
    ##                                 0.015354824 
    ##                      default_contract_group 
    ##                                 0.015176521 
    ## Client_Programs_count_at_observation_cutoff 
    ##                                 0.013379089 
    ##                               Issues_Raised 
    ##                                 0.012716593 
    ##                                  ClientType 
    ##                                 0.009982411 
    ##              Client_Initiated_Cancellations 
    ##                                 0.009397192 
    ##                               AgeAtCreation 
    ##                                 0.007935615 
    ##                                 PCNeedsFlag 
    ##                                 0.004249748

``` r
RFtree1<-getTree(kc.rf, k=1, labelVar=TRUE)
colnames(RFtree1)<-sapply(colnames(RFtree1),collapse)
RFrules1<-getConds(RFtree1)
print (head (RFrules1,20))
```

    ## [[1]]
    ## [1] "Client_Initiated_Cancellations > 2.62096444626073  &  MostUsedBillingGrade > 3.5  &  Client_Programs_count_at_observation_cutoff < 0.840036378334681  =>  0"
    ## 
    ## [[2]]
    ## [1] "ClientType > 1038323  &  Client_Programs_count_at_observation_cutoff > 0.340036378334681  &  Client_Initiated_Cancellations < 2.62096444626073  &  MostUsedBillingGrade > 3.5  &  Client_Programs_count_at_observation_cutoff < 0.840036378334681  =>  0"
    ## 
    ## [[3]]
    ## [1] "Issues_Raised > 3.5  &  AgeAtCreation > 60.5  &  HCW_Ratio < 3.5  &  HOME_STR_state < 9  &  Client_Programs_count_at_observation_cutoff > 0.840036378334681  =>  1"
    ## 
    ## [[4]]
    ## [1] "MostUsedBillingGrade > 1.5  &  Client_Initiated_Cancellations > 9.5  &  HCW_Ratio > 3.5  &  HOME_STR_state < 9  &  Client_Programs_count_at_observation_cutoff > 0.840036378334681  =>  0"
    ## 
    ## [[5]]
    ## [1] "HCW_Ratio > 10.5  &  Issues_Raised < 0.88332652227217  &  AverageCoreProgramHours > 26.25  &  AverageCoreProgramHours > 17.125  &  MostUsedBillingGrade < 3.5  &  Client_Programs_count_at_observation_cutoff < 0.840036378334681  =>  0"
    ## 
    ## [[6]]
    ## [1] "PCNeedsFlag > 1  &  Issues_Raised < 0.88332652227217  &  Client_Programs_count_at_observation_cutoff < 0.340036378334681  &  Client_Initiated_Cancellations < 2.62096444626073  &  MostUsedBillingGrade > 3.5  &  Client_Programs_count_at_observation_cutoff < 0.840036378334681  =>  0"
    ## 
    ## [[7]]
    ## [1] "MostUsedBillingGrade < 6.5  &  ClientType < 1038323  &  Client_Programs_count_at_observation_cutoff > 0.340036378334681  &  Client_Initiated_Cancellations < 2.62096444626073  &  MostUsedBillingGrade > 3.5  &  Client_Programs_count_at_observation_cutoff < 0.840036378334681  =>  1"
    ## 
    ## [[8]]
    ## [1] "PCNeedsFlag < 1  &  Issues_Raised > 0.5  &  AgeAtCreation < 60.5  &  HCW_Ratio < 3.5  &  HOME_STR_state < 9  &  Client_Programs_count_at_observation_cutoff > 0.840036378334681  =>  0"
    ## 
    ## [[9]]
    ## [1] "PCNeedsFlag > 1  &  Issues_Raised > 0.5  &  AgeAtCreation < 60.5  &  HCW_Ratio < 3.5  &  HOME_STR_state < 9  &  Client_Programs_count_at_observation_cutoff > 0.840036378334681  =>  1"
    ## 
    ## [[10]]
    ## [1] "PCNeedsFlag < 1  &  MostUsedBillingGrade < 1.5  &  Client_Initiated_Cancellations > 9.5  &  HCW_Ratio > 3.5  &  HOME_STR_state < 9  &  Client_Programs_count_at_observation_cutoff > 0.840036378334681  =>  0"
    ## 
    ## [[11]]
    ## [1] "AverageCoreProgramHours < 13.6388888888889  &  HCW_Ratio < 4.5  &  PCNeedsFlag > 1  &  AgeAtCreation < 81.5  &  HOME_STR_state > 9  &  Client_Programs_count_at_observation_cutoff > 0.840036378334681  =>  1"
    ## 
    ## [[12]]
    ## [1] "AverageCoreProgramHours > 13.6388888888889  &  HCW_Ratio < 4.5  &  PCNeedsFlag > 1  &  AgeAtCreation < 81.5  &  HOME_STR_state > 9  &  Client_Programs_count_at_observation_cutoff > 0.840036378334681  =>  0"
    ## 
    ## [[13]]
    ## [1] "AverageCoreProgramHours > 99.5069444444445  &  ClientType < 838057  &  AverageCoreProgramHours > 60.7916666666667  &  AgeAtCreation > 81.5  &  HOME_STR_state > 9  &  Client_Programs_count_at_observation_cutoff > 0.840036378334681  =>  0"
    ## 
    ## [[14]]
    ## [1] "MostUsedBillingGrade > 4.5  &  ClientType > 838057  &  AverageCoreProgramHours > 60.7916666666667  &  AgeAtCreation > 81.5  &  HOME_STR_state > 9  &  Client_Programs_count_at_observation_cutoff > 0.840036378334681  =>  1"
    ## 
    ## [[15]]
    ## [1] "ClientType < 1048059  &  MostUsedBillingGrade < 1.5  &  default_contract_group < 24  &  AverageCoreProgramHours < 6.125  &  AverageCoreProgramHours < 17.125  &  MostUsedBillingGrade < 3.5  &  Client_Programs_count_at_observation_cutoff < 0.840036378334681  =>  1"
    ## 
    ## [[16]]
    ## [1] "default_contract_group < 1  &  Client_Initiated_Cancellations < 0.5  &  ClientType < 1043888  &  AverageCoreProgramHours > 6.125  &  AverageCoreProgramHours < 17.125  &  MostUsedBillingGrade < 3.5  &  Client_Programs_count_at_observation_cutoff < 0.840036378334681  =>  1"
    ## 
    ## [[17]]
    ## [1] "Client_Initiated_Cancellations < 3.5  &  Client_Initiated_Cancellations > 0.5  &  ClientType < 1043888  &  AverageCoreProgramHours > 6.125  &  AverageCoreProgramHours < 17.125  &  MostUsedBillingGrade < 3.5  &  Client_Programs_count_at_observation_cutoff < 0.840036378334681  =>  1"
    ## 
    ## [[18]]
    ## [1] "Client_Initiated_Cancellations > 3.5  &  Client_Initiated_Cancellations > 0.5  &  ClientType < 1043888  &  AverageCoreProgramHours > 6.125  &  AverageCoreProgramHours < 17.125  &  MostUsedBillingGrade < 3.5  &  Client_Programs_count_at_observation_cutoff < 0.840036378334681  =>  0"
    ## 
    ## [[19]]
    ## [1] "AgeAtCreation > 76.5  &  ClientType > 1044464  &  ClientType > 1043888  &  AverageCoreProgramHours > 6.125  &  AverageCoreProgramHours < 17.125  &  MostUsedBillingGrade < 3.5  &  Client_Programs_count_at_observation_cutoff < 0.840036378334681  =>  0"
    ## 
    ## [[20]]
    ## [1] "AverageCoreProgramHours > 18.0625  &  ClientType < 1047805  &  default_contract_group < 24  &  AverageCoreProgramHours < 26.25  &  AverageCoreProgramHours > 17.125  &  MostUsedBillingGrade < 3.5  &  Client_Programs_count_at_observation_cutoff < 0.840036378334681  =>  1"

``` r
# make predictions
pr.kc.rf <- predict(kc.rf, newdata = test, type = 'prob')[, 2]
fitted.results.rf <- ifelse(pr.kc.rf > fitThreshold, 1, 0)
confusion_maxtix <- table(test$Label, fitted.results.rf)
# print.table(confusion_maxtix)
kc.rf.eval.results <- computePerformanceMeasures(confusion_maxtix, fScoreB_param)

#==========================
# C5.0 model tree, rules and performance

C5.0Trials_param <- 10

set.seed(1)
kc.c50 <-
  C50::C5.0(
    x = train[-ndxLabel],
    y = train$Label,
    trial = C5.0Trials_param,
    rules = FALSE,
    control = C5.0Control(earlyStopping = TRUE)
  )
summary(kc.c50)
```

    ## 
    ## Call:
    ## C5.0.default(x = train[-ndxLabel], y = train$Label, trials
    ##  = C5.0Trials_param, rules = FALSE, control = C5.0Control(earlyStopping
    ##  = TRUE))
    ## 
    ## 
    ## C5.0 [Release 2.07 GPL Edition]      Mon Apr 17 21:04:46 2017
    ## -------------------------------
    ## 
    ## Class specified by attribute `outcome'
    ## 
    ## Read 4130 cases (12 attributes) from undefined.data
    ## 
    ## -----  Trial 0:  -----
    ## 
    ## Decision tree:
    ## 
    ## MostUsedBillingGrade in [BGrade4-BGrade9]:
    ## :...HOME_STR_state in {ACT,NSW,VIC,WA}: 0 (129/28)
    ## :   HOME_STR_state in {QLD,SA}:
    ## :   :...MostUsedBillingGrade in [BGrade4-BGrade6]: 1 (184/50)
    ## :       MostUsedBillingGrade = BGrade9:
    ## :       :...Issues_Raised <= 0: 0 (6)
    ## :           Issues_Raised > 0:
    ## :           :...AverageCoreProgramHours <= 2.5: 0 (5)
    ## :               AverageCoreProgramHours > 2.5: 1 (33/8)
    ## MostUsedBillingGrade in [BGrade1-BGrade3]:
    ## :...AverageCoreProgramHours > 17.875: 0 (2689/297)
    ##     AverageCoreProgramHours <= 17.875:
    ##     :...AgeAtCreation > 84: 0 (342/62)
    ##         AgeAtCreation <= 84:
    ##         :...ClientType in {Bro,CAC,CCP,Com,Dem,Dis,DVA,NRC,Nur,Per,Pri,Res,Soc,
    ##             :              TCP,VHC,You}: 0 (50/18)
    ##             ClientType in {EAC,TAC}: 1 (14/3)
    ##             ClientType = Dom:
    ##             :...Client_Programs_count_at_observation_cutoff > 0: 1 (8/1)
    ##             :   Client_Programs_count_at_observation_cutoff <= 0:
    ##             :   :...AverageCoreProgramHours <= 12.875: 1 (15/5)
    ##             :       AverageCoreProgramHours > 12.875: 0 (10/1)
    ##             ClientType = HAC:
    ##             :...default_contract_group = Disability: 1 (1)
    ##                 default_contract_group in {DVA/VHC,TransPac}: 0 (10)
    ##                 default_contract_group in {CHSP,Package,Private/Commercial}:
    ##                 :...Issues_Raised > 0.7666531: [S1]
    ##                     Issues_Raised <= 0.7666531:
    ##                     :...AverageCoreProgramHours > 11.25: 0 (98/11)
    ##                         AverageCoreProgramHours <= 11.25:
    ##                         :...HOME_STR_state in {ACT,VIC,WA}: 0 (21/6)
    ##                             HOME_STR_state = QLD: 1 (53/23)
    ##                             HOME_STR_state = SA: [S2]
    ##                             HOME_STR_state = NSW: [S3]
    ## 
    ## SubTree [S1]
    ## 
    ## default_contract_group = Private/Commercial: 0 (12/2)
    ## default_contract_group in {CHSP,Package}:
    ## :...HOME_STR_state in {ACT,QLD,SA,WA}: 0 (38/14)
    ##     HOME_STR_state in {NSW,VIC}: 1 (129/58)
    ## 
    ## SubTree [S2]
    ## 
    ## Client_Programs_count_at_observation_cutoff <= 0: 0 (86/32)
    ## Client_Programs_count_at_observation_cutoff > 0: 1 (5/1)
    ## 
    ## SubTree [S3]
    ## 
    ## default_contract_group in {CHSP,Package}: 0 (157/41)
    ## default_contract_group = Private/Commercial:
    ## :...AgeAtCreation <= 36: 0 (6/1)
    ##     AgeAtCreation > 36: 1 (29/9)
    ## 
    ## -----  Trial 1:  -----
    ## 
    ## Decision tree:
    ## 
    ## AverageCoreProgramHours > 37.5:
    ## :...ClientType in {Bro,CAC,CCP,Com,Dem,Dis,Dom,DVA,EAC,HAC,NRC,Nur,Per,Pri,Res,
    ## :   :              TAC,TCP,VHC}: 0 (1808.6/345.3)
    ## :   ClientType in {Soc,You}: 1 (22.3/4)
    ## AverageCoreProgramHours <= 37.5:
    ## :...Client_Initiated_Cancellations > 3: 0 (182.2/44.8)
    ##     Client_Initiated_Cancellations <= 3:
    ##     :...MostUsedBillingGrade in [BGrade2-BGrade9]:
    ##         :...AgeAtCreation > 73: 0 (625.4/277.9)
    ##         :   AgeAtCreation <= 73:
    ##         :   :...AgeAtCreation <= 58: 0 (78.1/31.6)
    ##         :       AgeAtCreation > 58: 1 (248.9/91.6)
    ##         MostUsedBillingGrade = BGrade1:
    ##         :...Issues_Raised > 0.7666531: 0 (449.1/209.4)
    ##             Issues_Raised <= 0.7666531:
    ##             :...PCNeedsFlag = Y: 1 (15.8/4.8)
    ##                 PCNeedsFlag = N:
    ##                 :...HCW_Ratio > 4: 0 (174.1/21.9)
    ##                     HCW_Ratio <= 4: [S1]
    ## 
    ## SubTree [S1]
    ## 
    ## Client_Programs_count_at_observation_cutoff <= 0.6800728: 0 (448.2/131)
    ## Client_Programs_count_at_observation_cutoff > 0.6800728: 1 (77.2/30)
    ## 
    ## -----  Trial 2:  -----
    ## 
    ## Decision tree:
    ## 
    ## AverageCoreProgramHours > 43:
    ## :...Issues_Raised <= 0: 0 (699/107.2)
    ## :   Issues_Raised > 0:
    ## :   :...AgeAtCreation > 78: 0 (436.1/101.6)
    ## :       AgeAtCreation <= 78:
    ## :       :...Client_Initiated_Cancellations > 4: 0 (117.8/29.4)
    ## :           Client_Initiated_Cancellations <= 4:
    ## :           :...Client_Initiated_Cancellations <= 1: 1 (159.9/63.9)
    ## :               Client_Initiated_Cancellations > 1: 0 (169.5/74.3)
    ## AverageCoreProgramHours <= 43:
    ## :...AverageCoreProgramHours <= 6.833333:
    ##     :...default_contract_group in {Disability,DVA/VHC,TransPac}: 0 (7.8)
    ##     :   default_contract_group in {CHSP,Package,Private/Commercial}:
    ##     :   :...Issues_Raised > 0.7666531: 1 (153.1/47.1)
    ##     :       Issues_Raised <= 0.7666531:
    ##     :       :...HOME_STR_state in {ACT,QLD,SA}: 1 (506.6/199.8)
    ##     :           HOME_STR_state in {NSW,VIC,WA}: 0 (314/147.8)
    ##     AverageCoreProgramHours > 6.833333:
    ##     :...Issues_Raised <= 0.7666531:
    ##         :...PCNeedsFlag = N: 0 (730.6/215.3)
    ##         :   PCNeedsFlag = Y: 1 (92.7/41.6)
    ##         Issues_Raised > 0.7666531:
    ##         :...AgeAtCreation <= 61: 1 (30.1/4.8)
    ##             AgeAtCreation > 61:
    ##             :...Client_Initiated_Cancellations > 3: 0 (118.3/39.1)
    ##                 Client_Initiated_Cancellations <= 3:
    ##                 :...default_contract_group in {CHSP,Package}: 1 (540.5/229.9)
    ##                     default_contract_group in {Disability,DVA/VHC,
    ##                                                Private/Commercial,
    ##                                                TransPac}: 0 (54/17.3)
    ## 
    ## -----  Trial 3:  -----
    ## 
    ## Decision tree:
    ## 
    ## AverageCoreProgramHours <= 9.666667:
    ## :...Client_Initiated_Cancellations > 2.241929: 0 (43.9/14)
    ## :   Client_Initiated_Cancellations <= 2.241929:
    ## :   :...default_contract_group in {Disability,
    ## :       :                          Private/Commercial}: 1 (388/170.8)
    ## :       default_contract_group in {DVA/VHC,Package,TransPac}: 0 (83.5/37.7)
    ## :       default_contract_group = CHSP:
    ## :       :...AgeAtCreation <= 84: 1 (531.9/237.8)
    ## :           AgeAtCreation > 84: 0 (213.9/89.8)
    ## AverageCoreProgramHours > 9.666667:
    ## :...HCW_Ratio > 41: 0 (64.8/3)
    ##     HCW_Ratio <= 41:
    ##     :...Issues_Raised > 0.7666531:
    ##         :...AgeAtCreation > 90: 0 (86.5/16.4)
    ##         :   AgeAtCreation <= 90:
    ##         :   :...HOME_STR_state in {ACT,WA}: 1 (213.6/89.7)
    ##         :       HOME_STR_state in {NSW,QLD,SA,VIC}: 0 (1191.2/482)
    ##         Issues_Raised <= 0.7666531:
    ##         :...Client_Initiated_Cancellations > 9: 0 (27.5)
    ##             Client_Initiated_Cancellations <= 9:
    ##             :...HCW_Ratio > 10:
    ##                 :...AgeAtCreation <= 42: 1 (17/5.9)
    ##                 :   AgeAtCreation > 42: 0 (258/29.1)
    ##                 HCW_Ratio <= 10:
    ##                 :...AgeAtCreation > 75:
    ##                     :...AverageCoreProgramHours > 24.125: 0 (374.1/69.1)
    ##                     :   AverageCoreProgramHours <= 24.125:
    ##                     :   :...ClientType in {Bro,CCP}: 1 (14.5/1.1)
    ##                     :       ClientType in {CAC,Com,Dem,Dis,Dom,DVA,EAC,HAC,NRC,
    ##                     :                      Nur,Per,Pri,Res,Soc,TAC,TCP,VHC,
    ##                     :                      You}: 0 (224.5/75.4)
    ##                     AgeAtCreation <= 75:
    ##                     :...default_contract_group in {Disability,
    ##                         :                          DVA/VHC}: 0 (6.5)
    ##                         default_contract_group = TransPac: 1 (5.7)
    ##                         default_contract_group in {CHSP,Package,
    ##                         :                          Private/Commercial}:
    ##                         :...MostUsedBillingGrade in [BGrade3-BGrade9]: 0 (44.7/9.6)
    ##                             MostUsedBillingGrade in [BGrade1-BGrade2]:
    ##                             :...PCNeedsFlag = Y: 1 (33.8/14.8)
    ##                                 PCNeedsFlag = N:
    ##                                 :...HCW_Ratio > 6: 0 (65.6/14.9)
    ##                                     HCW_Ratio <= 6: [S1]
    ## 
    ## SubTree [S1]
    ## 
    ## Client_Programs_count_at_observation_cutoff <= 0: 0 (122.1/43.6)
    ## Client_Programs_count_at_observation_cutoff > 0: 1 (118.7/51.3)
    ## 
    ## -----  Trial 4:  -----
    ## 
    ## Decision tree:
    ## 
    ## AverageCoreProgramHours > 46:
    ## :...HCW_Ratio > 41: 0 (58.1/3.3)
    ## :   HCW_Ratio <= 41:
    ## :   :...Issues_Raised <= 0: 0 (556.5/113)
    ## :       Issues_Raised > 0:
    ## :       :...AgeAtCreation > 78: 0 (347.8/107.2)
    ## :           AgeAtCreation <= 78:
    ## :           :...Client_Programs_count_at_observation_cutoff <= 0: 0 (46.3/13.1)
    ## :               Client_Programs_count_at_observation_cutoff > 0:
    ## :               :...default_contract_group in {CHSP,Package,
    ## :                   :                          Private/Commercial}: 0 (363.9/177.5)
    ## :                   default_contract_group in {Disability,DVA/VHC,
    ## :                                              TransPac}: 1 (19.7/3.6)
    ## AverageCoreProgramHours <= 46:
    ## :...AgeAtCreation > 92: 0 (61.4/14.4)
    ##     AgeAtCreation <= 92:
    ##     :...Client_Programs_count_at_observation_cutoff > 0:
    ##         :...Issues_Raised <= 0.7666531: 0 (442.5/207.3)
    ##         :   Issues_Raised > 0.7666531: 1 (388.3/170.7)
    ##         Client_Programs_count_at_observation_cutoff <= 0:
    ##         :...HCW_Ratio > 5: 0 (257.2/78.3)
    ##             HCW_Ratio <= 5:
    ##             :...MostUsedBillingGrade in [BGrade6-BGrade9]: 0 (187.6/66.8)
    ##                 MostUsedBillingGrade in [BGrade1-BGrade5]:
    ##                 :...MostUsedBillingGrade = BGrade1:
    ##                     :...AgeAtCreation <= 57: 1 (44/14.7)
    ##                     :   AgeAtCreation > 57: 0 (764.9/316.9)
    ##                     MostUsedBillingGrade in [BGrade2-BGrade5]:
    ##                     :...AgeAtCreation <= 70: 0 (143.1/58.3)
    ##                         AgeAtCreation > 70: 1 (448.5/194.8)
    ## 
    ## -----  Trial 5:  -----
    ## 
    ## Decision tree:
    ## 
    ## Client_Initiated_Cancellations > 5: 0 (314.7/91.5)
    ## Client_Initiated_Cancellations <= 5:
    ## :...HCW_Ratio > 40: 0 (30.8)
    ##     HCW_Ratio <= 40:
    ##     :...Issues_Raised > 0:
    ##         :...MostUsedBillingGrade in [BGrade4-BGrade9]: 1 (157.9/63.5)
    ##         :   MostUsedBillingGrade in [BGrade1-BGrade3]:
    ##         :   :...Issues_Raised <= 0.7666531: 0 (116.7/41.4)
    ##         :       Issues_Raised > 0.7666531:
    ##         :       :...AgeAtCreation > 78: 0 (741.8/314.5)
    ##         :           AgeAtCreation <= 78:
    ##         :           :...Issues_Raised <= 1: 1 (540.6/225.9)
    ##         :               Issues_Raised > 1: 0 (218.8/96.1)
    ##         Issues_Raised <= 0:
    ##         :...AverageCoreProgramHours > 35.5: 0 (563.7/145.8)
    ##             AverageCoreProgramHours <= 35.5:
    ##             :...MostUsedBillingGrade = BGrade9: 0 (8.1)
    ##                 MostUsedBillingGrade in [BGrade1-BGrade6]:
    ##                 :...HCW_Ratio > 4: 0 (200.2/70.5)
    ##                     HCW_Ratio <= 4: [S1]
    ## 
    ## SubTree [S1]
    ## 
    ## Client_Programs_count_at_observation_cutoff > 0: 1 (162.3/60.2)
    ## Client_Programs_count_at_observation_cutoff <= 0:
    ## :...HOME_STR_state in {ACT,NSW,VIC,WA}: 0 (532.5/220.8)
    ##     HOME_STR_state = QLD: 1 (120.6/51.8)
    ##     HOME_STR_state = SA:
    ##     :...MostUsedBillingGrade = BGrade1: 0 (120.4/53.3)
    ##         MostUsedBillingGrade in [BGrade2-BGrade6]: 1 (301/143)
    ## 
    ## -----  Trial 6:  -----
    ## 
    ## Decision tree:
    ## 
    ## AverageCoreProgramHours > 46:
    ## :...AgeAtCreation > 78: 0 (456.8/43.1)
    ## :   AgeAtCreation <= 78:
    ## :   :...ClientType in {Bro,CCP,Com,Dem,Dis,Dom,DVA,EAC,HAC,NRC,Nur,Per,Pri,Res,
    ## :       :              TAC,TCP,VHC}: 0 (671.8/231)
    ## :       ClientType in {CAC,Soc,You}: 1 (32/5.6)
    ## AverageCoreProgramHours <= 46:
    ## :...ClientType in {Bro,CAC,Com,Dem,Dis,Dom,DVA,NRC,Nur,Per,Pri,Soc,TAC,TCP,VHC,
    ##     :              You}: 0 (431.4/187.7)
    ##     ClientType in {CCP,EAC,Res}: 1 (98.5/38.1)
    ##     ClientType = HAC:
    ##     :...default_contract_group = Disability: 1 (1.7)
    ##         default_contract_group in {DVA/VHC,Package,TransPac}: 0 (194.1/70.8)
    ##         default_contract_group = CHSP:
    ##         :...Issues_Raised > 0:
    ##         :   :...AgeAtCreation <= 82: 0 (576.2/268.4)
    ##         :   :   AgeAtCreation > 82: 1 (359.3/150.7)
    ##         :   Issues_Raised <= 0:
    ##         :   :...AverageCoreProgramHours > 11.25: 0 (308/65.6)
    ##         :       AverageCoreProgramHours <= 11.25:
    ##         :       :...AgeAtCreation <= 66: 1 (55/15.2)
    ##         :           AgeAtCreation > 66: 0 (482/222.4)
    ##         default_contract_group = Private/Commercial:
    ##         :...HOME_STR_state = WA: 0 (8)
    ##             HOME_STR_state in {ACT,NSW,QLD,SA,VIC}:
    ##             :...HCW_Ratio > 5: 1 (29.8/6.6)
    ##                 HCW_Ratio <= 5:
    ##                 :...Issues_Raised > 0: 0 (38.8/11)
    ##                     Issues_Raised <= 0:
    ##                     :...MostUsedBillingGrade = BGrade1: 1 (80/27.2)
    ##                         MostUsedBillingGrade in [BGrade2-BGrade9]: [S1]
    ## 
    ## SubTree [S1]
    ## 
    ## Client_Programs_count_at_observation_cutoff <= 0: 0 (212.8/94.7)
    ## Client_Programs_count_at_observation_cutoff > 0: 1 (19.7/6.5)
    ## 
    ## -----  Trial 7:  -----
    ## 
    ## Decision tree:
    ## 
    ## AverageCoreProgramHours > 40: 0 (1039.9/149.5)
    ## AverageCoreProgramHours <= 40:
    ## :...Client_Programs_count_at_observation_cutoff > 2: 0 (29.7/5.8)
    ##     Client_Programs_count_at_observation_cutoff <= 2:
    ##     :...AgeAtCreation > 83:
    ##         :...HOME_STR_state in {NSW,QLD}: 0 (553.5/169.3)
    ##         :   HOME_STR_state in {ACT,SA,VIC,WA}:
    ##         :   :...Issues_Raised <= 0.7666531: 0 (215.8/96.9)
    ##         :       Issues_Raised > 0.7666531: 1 (87.4/34.2)
    ##         AgeAtCreation <= 83:
    ##         :...Client_Initiated_Cancellations > 3: 0 (106.3/31.8)
    ##             Client_Initiated_Cancellations <= 3:
    ##             :...AverageCoreProgramHours > 13.75:
    ##                 :...Client_Programs_count_at_observation_cutoff <= 0.6800728: 0 (285.8/97)
    ##                 :   Client_Programs_count_at_observation_cutoff > 0.6800728:
    ##                 :   :...ClientType in {Bro,CAC,CCP,Dis,EAC,Per,TAC,
    ##                 :       :              VHC}: 0 (12.1)
    ##                 :       ClientType in {Com,Dem,Dom,DVA,HAC,NRC,Nur,Pri,Res,Soc,
    ##                 :                      TCP,You}: 1 (258.4/112.9)
    ##                 AverageCoreProgramHours <= 13.75:
    ##                 :...Issues_Raised > 0.7666531: 1 (318.4/118.1)
    ##                     Issues_Raised <= 0.7666531:
    ##                     :...ClientType in {NRC,Soc,TAC,TCP,VHC,
    ##                         :              You}: 1 (0)
    ##                         ClientType in {CCP,Dem,Dis,DVA,Nur,
    ##                         :              Res}: 0 (14.3)
    ##                         ClientType in {Bro,CAC,Com,Dom,EAC,HAC,Per,Pri}:
    ##                         :...HCW_Ratio > 2: 0 (135.8/52.6)
    ##                             HCW_Ratio <= 2: [S1]
    ## 
    ## SubTree [S1]
    ## 
    ## default_contract_group in {Disability,DVA/VHC}: 1 (1.5)
    ## default_contract_group in {Package,TransPac}: 0 (23/10.8)
    ## default_contract_group = Private/Commercial:
    ## :...AgeAtCreation <= 40: 0 (29.2/9.1)
    ## :   AgeAtCreation > 40: 1 (348.1/137.1)
    ## default_contract_group = CHSP:
    ## :...ClientType in {Bro,CAC,Pri}: 1 (0)
    ##     ClientType = Com: 0 (5.7)
    ##     ClientType in {Dom,EAC,HAC,Per}:
    ##     :...Client_Programs_count_at_observation_cutoff > 0: 1 (134.9/44.3)
    ##         Client_Programs_count_at_observation_cutoff <= 0:
    ##         :...AverageCoreProgramHours <= 9.25: 1 (323.2/129)
    ##             AverageCoreProgramHours > 9.25: 0 (20.9/4.6)
    ## 
    ## -----  Trial 8:  -----
    ## 
    ## Decision tree:
    ## 
    ## AverageCoreProgramHours > 37.5:
    ## :...ClientType in {Bro,CAC,CCP,Com,Dem,Dis,Dom,DVA,EAC,HAC,NRC,Nur,Per,Pri,Res,
    ## :   :              TAC,TCP,VHC}: 0 (892.3/51.8)
    ## :   ClientType in {Soc,You}: 1 (32.6/6.4)
    ## AverageCoreProgramHours <= 37.5:
    ## :...Client_Initiated_Cancellations > 3: 0 (124.1/15.8)
    ##     Client_Initiated_Cancellations <= 3:
    ##     :...MostUsedBillingGrade in [BGrade4-BGrade9]:
    ##         :...HOME_STR_state in {NSW,VIC,WA}: 0 (125.6/34.4)
    ##         :   HOME_STR_state in {ACT,QLD,SA}:
    ##         :   :...AgeAtCreation <= 71: 1 (97.4/12.9)
    ##         :       AgeAtCreation > 71:
    ##         :       :...MostUsedBillingGrade in [BGrade4-BGrade6]: 1 (228.2/80.5)
    ##         :           MostUsedBillingGrade = BGrade9: 0 (71.6/29.3)
    ##         MostUsedBillingGrade in [BGrade1-BGrade3]:
    ##         :...PCNeedsFlag = Y:
    ##             :...Client_Initiated_Cancellations > 2: 0 (40.4/7.3)
    ##             :   Client_Initiated_Cancellations <= 2:
    ##             :   :...Issues_Raised <= 1: 1 (294.4/124.9)
    ##             :       Issues_Raised > 1: 0 (58.9/16.5)
    ##             PCNeedsFlag = N:
    ##             :...AgeAtCreation > 88: 0 (117.6/10.5)
    ##                 AgeAtCreation <= 88:
    ##                 :...AverageCoreProgramHours > 18.5: 0 (465.7/125.3)
    ##                     AverageCoreProgramHours <= 18.5:
    ##                     :...default_contract_group = Disability: 1 (1.4)
    ##                         default_contract_group in {DVA/VHC,Package,
    ##                         :                          TransPac}: 0 (66.1/18.3)
    ##                         default_contract_group in {CHSP,Private/Commercial}:
    ##                         :...HOME_STR_state = WA: 0 (16.8/1.4)
    ##                             HOME_STR_state in {ACT,NSW,QLD,SA,VIC}:
    ##                             :...Issues_Raised <= 0.7666531: 0 (782.9/307.5)
    ##                                 Issues_Raised > 0.7666531:
    ##                                 :...MostUsedBillingGrade = BGrade3: 1 (20.4/3.1)
    ##                                     MostUsedBillingGrade in [BGrade1-BGrade2]: [S1]
    ## 
    ## SubTree [S1]
    ## 
    ## default_contract_group = Private/Commercial: 0 (25.8/5.3)
    ## default_contract_group = CHSP:
    ## :...AgeAtCreation <= 65: 1 (20.5/3.3)
    ##     AgeAtCreation > 65:
    ##     :...Client_Programs_count_at_observation_cutoff <= 0: 0 (261/116.6)
    ##         Client_Programs_count_at_observation_cutoff > 0: 1 (60.2/21.8)
    ## 
    ## -----  Trial 9:  -----
    ## 
    ## Decision tree:
    ## 
    ## AverageCoreProgramHours > 18.75: 0 (1393.8/87.3)
    ## AverageCoreProgramHours <= 18.75:
    ## :...Client_Initiated_Cancellations > 3: 0 (34)
    ##     Client_Initiated_Cancellations <= 3:
    ##     :...ClientType in {CAC,CCP,Dem,Dis,DVA,NRC,Soc,TCP,VHC,
    ##         :              You}: 0 (46.8/2.8)
    ##         ClientType in {Bro,Com,Dom,EAC,HAC,Nur,Per,Pri,Res,TAC}:
    ##         :...AgeAtCreation > 83: 0 (554.2/167.4)
    ##             AgeAtCreation <= 83:
    ##             :...HCW_Ratio > 3: 0 (237.7/88.6)
    ##                 HCW_Ratio <= 3:
    ##                 :...HOME_STR_state = VIC: 0 (10.2)
    ##                     HOME_STR_state in {ACT,QLD,SA}:
    ##                     :...HOME_STR_state in {ACT,QLD}: 1 (347.8/114)
    ##                     :   HOME_STR_state = SA:
    ##                     :   :...MostUsedBillingGrade = BGrade1: 0 (104.7/25.4)
    ##                     :       MostUsedBillingGrade in [BGrade2-BGrade9]: 1 (274.1/63.6)
    ##                     HOME_STR_state in {NSW,WA}:
    ##                     :...AgeAtCreation <= 40: 0 (25.1)
    ##                         AgeAtCreation > 40:
    ##                         :...MostUsedBillingGrade = BGrade9: 0 (6.6)
    ##                             MostUsedBillingGrade in [BGrade1-BGrade6]:
    ##                             :...Issues_Raised > 0.7666531: 1 (222.2/64.1)
    ##                                 Issues_Raised <= 0.7666531:
    ##                                 :...AgeAtCreation <= 73: 1 (212.3/90.3)
    ##                                     AgeAtCreation > 73: 0 (174.5/39.2)
    ## 
    ## 
    ## Evaluation on training data (4130 cases):
    ## 
    ## Trial        Decision Tree   
    ## -----      ----------------  
    ##    Size      Errors  
    ## 
    ##    0     25  671(16.2%)
    ##    1     11  788(19.1%)
    ##    2     15  946(22.9%)
    ##    3     22  938(22.7%)
    ##    4     15 1026(24.8%)
    ##    5     15  881(21.3%)
    ##    6     18  920(22.3%)
    ##    7     20  823(19.9%)
    ##    8     21  734(17.8%)
    ##    9     14  745(18.0%)
    ## boost            641(15.5%)   <<
    ## 
    ## 
    ##     (a)   (b)    <-classified as
    ##    ----  ----
    ##    3191   113    (a): class 0
    ##     528   298    (b): class 1
    ## 
    ## 
    ##  Attribute usage:
    ## 
    ##  100.00% AgeAtCreation
    ##  100.00% MostUsedBillingGrade
    ##  100.00% ClientType
    ##  100.00% Client_Initiated_Cancellations
    ##  100.00% AverageCoreProgramHours
    ##   99.98% Issues_Raised
    ##   99.98% HCW_Ratio
    ##   67.92% HOME_STR_state
    ##   67.80% Client_Programs_count_at_observation_cutoff
    ##   66.80% default_contract_group
    ##   46.97% PCNeedsFlag
    ## 
    ## 
    ## Time: 0.1 secs

``` r
# myTree <- C50:::as.party.C5.0(kc.c50)
# plot(myTree)
# C50..plot.C5.0 (kc.c50,trial=0,subtree=NULL)

set.seed(1)
kc.c50.rules <-
  C50::C5.0(
    x = train[,-ndxLabel],
    y = train$Label,
    trial = C5.0Trials_param,
    rules = TRUE,
    control = C5.0Control(bands = 100, earlyStopping = TRUE)
  )
head(summary(kc.c50.rules),50)
```

    ## $output
    ## [1] "\nC5.0 [Release 2.07 GPL Edition]  \tMon Apr 17 21:04:47 2017\n-------------------------------\n    **  Warning (-u): rule ordering has no effect on boosting\n\nClass specified by attribute `outcome'\n\nRead 4130 cases (12 attributes) from undefined.data\n\n-----  Trial 0:  -----\n\nRules:\n\nRule 0/1: (2879/468, lift 1.0)\n\tHOME_STR_state in {ACT, NSW, VIC, WA}\n\t->  class 0  [0.837]\n\nRule 0/2: (2689/297, lift 1.1)\n\tMostUsedBillingGrade in [BGrade1-BGrade3]\n\tAverageCoreProgramHours > 17.875\n\t->  class 0  [0.889]\n\nRule 0/3: (1390/871, lift 1.9)\n\tAverageCoreProgramHours <= 17.875\n\t->  class 1  [0.374]\n\nRule 0/4: (2535/424, lift 1.0)\n\tMostUsedBillingGrade in [BGrade1-BGrade3]\n\tClientType = HAC\n\tdefault_contract_group in {CHSP, Package}\n\t->  class 0  [0.832]\n\nRule 0/5: (566/79, lift 1.1)\n\tMostUsedBillingGrade in [BGrade1-BGrade3]\n\tClientType in {Bro, CAC, CCP, Com, Dem, Dis, NRC, Per, Pri, Res, Soc,\n                       TCP, VHC, You}\n\t->  class 0  [0.859]\n\nRule 0/6: (29/9, lift 3.4)\n\tIssues_Raised <= 0.7666531\n\tAgeAtCreation > 36\n\tAgeAtCreation <= 84\n\tMostUsedBillingGrade in [BGrade1-BGrade3]\n\tClientType = HAC\n\tdefault_contract_group = Private/Commercial\n\tHOME_STR_state = NSW\n\tAverageCoreProgramHours <= 11.25\n\t->  class 1  [0.677]\n\nRule 0/7: (15/3, lift 3.8)\n\tAgeAtCreation <= 84\n\tClientType in {EAC, TAC}\n\tAverageCoreProgramHours <= 17.875\n\t->  class 1  [0.765]\n\nRule 0/8: (9, lift 1.1)\n\tIssues_Raised <= 0\n\tMostUsedBillingGrade = BGrade9\n\t->  class 0  [0.909]\n\nRule 0/9: (8/1, lift 4.0)\n\tAgeAtCreation <= 84\n\tClientType = Dom\n\tClient_Programs_count_at_observation_cutoff > 0\n\tAverageCoreProgramHours <= 17.875\n\t->  class 1  [0.800]\n\nRule 0/10: (79/30, lift 3.1)\n\tIssues_Raised <= 0.7666531\n\tAgeAtCreation <= 84\n\tClientType = HAC\n\tHOME_STR_state = QLD\n\tAverageCoreProgramHours <= 11.25\n\t->  class 1  [0.617]\n\nRule 0/11: (16/5, lift 3.3)\n\tAgeAtCreation <= 84\n\tMostUsedBillingGrade in [BGrade1-BGrade3]\n\tClientType = Dom\n\tAverageCoreProgramHours <= 12.875\n\t->  class 1  [0.667]\n\nRule 0/12: (1120/120, lift 1.1)\n\tAgeAtCreation > 84\n\tMostUsedBillingGrade in [BGrade1-BGrade3]\n\t->  class 0  [0.892]\n\nRule 0/13: (1193/79, lift 1.2)\n\tIssues_Raised <= 0.7666531\n\tClientType = HAC\n\tdefault_contract_group in {CHSP, Package, Private/Commercial}\n\tAverageCoreProgramHours > 11.25\n\t->  class 0  [0.933]\n\nRule 0/14: (120/10, lift 1.1)\n\tdefault_contract_group in {DVA/VHC, TransPac}\n\t->  class 0  [0.910]\n\nRule 0/15: (184/50, lift 3.6)\n\tMostUsedBillingGrade in [BGrade5-BGrade6]\n\tHOME_STR_state in {QLD, SA}\n\t->  class 1  [0.726]\n\nDefault class: 0\n\n-----  Trial 1:  -----\n\nRules:\n\nRule 1/1: (2417.7/671.4, lift 1.1)\n\tIssues_Raised <= 0.7666531\n\t->  class 0  [0.722]\n\nRule 1/2: (1803/337.8, lift 1.2)\n\tClientType in {Bro, CAC, CCP, Com, Dem, Dis, Dom, EAC, HAC, NRC, Per,\n                       Pri, Res, TAC, TCP}\n\tAverageCoreProgramHours > 37.5\n\t->  class 0  [0.812]\n\nRule 1/3: (783.7/354.5, lift 1.7)\n\tIssues_Raised > 0.7666531\n\tAverageCoreProgramHours <= 37.5\n\t->  class 1  [0.548]\n\nRule 1/4: (712.2/136.8, lift 1.2)\n\tClient_Initiated_Cancellations > 3\n\t->  class 0  [0.807]\n\nRule 1/5: (21.9/4, lift 2.4)\n\tClientType in {Soc, You}\n\tAverageCoreProgramHours > 37.5\n\t->  class 1  [0.791]\n\nDefault class: 0\n\n-----  Trial 2:  -----\n\nRules:\n\nRule 2/1: (2638.9/760.8, lift 1.1)\n\tAverageCoreProgramHours > 13.75\n\t->  class 0  [0.712]\n\nRule 2/2: (1491.1/688.4, lift 1.4)\n\tAverageCoreProgramHours <= 13.75\n\t->  class 1  [0.538]\n\nRule 2/3: (1294.6/362.7, lift 1.2)\n\tAgeAtCreation > 83\n\t->  class 0  [0.720]\n\nRule 2/4: (562.3/255.2, lift 1.4)\n\tIssues_Raised > 0.7666531\n\tIssues_Raised <= 1\n\tAgeAtCreation <= 78\n\t->  class 1  [0.546]\n\nRule 2/5: (325.6/122.2, lift 1.6)\n\tIssues_Raised > 0.7666531\n\tIssues_Raised <= 1\n\tAgeAtCreation <= 78\n\tClient_Initiated_Cancellations <= 1\n\t->  class 1  [0.624]\n\nRule 2/6: (1473.8/361.7, lift 1.2)\n\tIssues_Raised <= 0.7666531\n\tdefault_contract_group in {CHSP, Disability, Package,\n                                   Private/Commercial}\n\tAverageCoreProgramHours > 7.5\n\t->  class 0  [0.754]\n\nRule 2/7: (107.6/31.7, lift 1.9)\n\tAgeAtCreation <= 83\n\tdefault_contract_group in {CHSP, Package, Private/Commercial}\n\tClient_Programs_count_at_observation_cutoff > 0.6800728\n\tAverageCoreProgramHours <= 13.75\n\t->  class 1  [0.702]\n\nDefault class: 0\n\n-----  Trial 3:  -----\n\nRules:\n\nRule 3/1: (4025.9/1631.2, lift 1.0)\n\tdefault_contract_group in {CHSP, Disability, Package,\n                                   Private/Commercial}\n\t->  class 0  [0.595]\n\nRule 3/2: (453/178.3, lift 1.5)\n\tMostUsedBillingGrade in [BGrade2-BGrade9]\n\tHOME_STR_state = SA\n\tAverageCoreProgramHours <= 46\n\t->  class 1  [0.606]\n\nRule 3/3: (612.3/282, lift 1.3)\n\tAgeAtCreation <= 88\n\tdefault_contract_group in {CHSP, Disability, Package,\n                                   Private/Commercial}\n\tClient_Programs_count_at_observation_cutoff > 0.6800728\n\tAverageCoreProgramHours <= 46\n\t->  class 1  [0.539]\n\nRule 3/4: (483.9/214, lift 1.4)\n\tAgeAtCreation <= 88\n\tdefault_contract_group in {CHSP, Disability, Package,\n                                   Private/Commercial}\n\tClient_Programs_count_at_observation_cutoff > 0.6800728\n\tClient_Programs_count_at_observation_cutoff <= 1\n\tAverageCoreProgramHours <= 46\n\t->  class 1  [0.557]\n\nRule 3/5: (48.7/10.2, lift 1.3)\n\tdefault_contract_group in {DVA/VHC, TransPac}\n\tAverageCoreProgramHours <= 46\n\t->  class 0  [0.779]\n\nRule 3/6: (1053.2/267.6, lift 1.2)\n\tHOME_STR_state in {NSW, SA, VIC}\n\tAverageCoreProgramHours > 46\n\t->  class 0  [0.745]\n\nRule 3/7: (23.7/3.3, lift 2.1)\n\tClientType in {Soc, You}\n\tAverageCoreProgramHours > 46\n\t->  class 1  [0.832]\n\nRule 3/8: (41.2/13.5, lift 1.7)\n\tAgeAtCreation <= 57\n\tMostUsedBillingGrade = BGrade1\n\tClient_Programs_count_at_observation_cutoff <= 0.6800728\n\tHCW_Ratio <= 4\n\t->  class 1  [0.665]\n\nRule 3/9: (72.6/28, lift 1.5)\n\tMostUsedBillingGrade = BGrade1\n\tdefault_contract_group in {CHSP, Package, Private/Commercial}\n\tPCNeedsFlag = Y\n\tAverageCoreProgramHours <= 46\n\t->  class 1  [0.612]\n\nRule 3/10: (195.6/91.2, lift 1.3)\n\tIssues_Raised > 0\n\tHOME_STR_state in {ACT, QLD, WA}\n\tAverageCoreProgramHours > 46\n\tHCW_Ratio <= 36\n\t->  class 1  [0.533]\n\nDefault class: 0\n\n-----  Trial 4:  -----\n\nRules:\n\nRule 4/1: (2002.5/902.4, lift 1.0)\n\tIssues_Raised > 0\n\t->  class 0  [0.549]\n\nRule 4/2: (1010.5/284, lift 1.2)\n\tIssues_Raised <= 0\n\tAverageCoreProgramHours > 17\n\t->  class 0  [0.719]\n\nRule 4/3: (1768.9/862.1, lift 1.2)\n\tAverageCoreProgramHours <= 17\n\t->  class 1  [0.513]\n\nRule 4/4: (442.8/111.6, lift 1.3)\n\tIssues_Raised <= 0\n\tAgeAtCreation > 73\n\tdefault_contract_group = CHSP\n\tHOME_STR_state = NSW\n\t->  class 0  [0.747]\n\nRule 4/5: (267.4/106.8, lift 1.4)\n\tIssues_Raised > 0\n\tAgeAtCreation <= 74\n\tClient_Initiated_Cancellations <= 8\n\tClient_Programs_count_at_observation_cutoff > 0\n\tAverageCoreProgramHours > 17\n\tAverageCoreProgramHours <= 188.75\n\t->  class 1  [0.600]\n\nRule 4/6: (308.8/131.3, lift 1.4)\n\tIssues_Raised > 0\n\tAgeAtCreation <= 92\n\tdefault_contract_group = CHSP\n\tHOME_STR_state = NSW\n\tAverageCoreProgramHours <= 17\n\t->  class 1  [0.574]\n\nRule 4/7: (366.6/142.5, lift 1.4)\n\tAgeAtCreation <= 73\n\tdefault_contract_group = CHSP\n\tAverageCoreProgramHours <= 17\n\t->  class 1  [0.611]\n\nRule 4/8: (84.8/31.3, lift 1.5)\n\tIssues_Raised > 0\n\tAgeAtCreation <= 74\n\tClientType in {CAC, Com, Dis, Dom, EAC, HAC, Per}\n\tClient_Initiated_Cancellations <= 8\n\tClient_Programs_count_at_observation_cutoff > 0\n\tPCNeedsFlag = Y\n\tAverageCoreProgramHours > 17\n\tHCW_Ratio <= 41\n\t->  class 1  [0.628]\n\nRule 4/9: (26.7/6.1, lift 1.8)\n\tClientType in {Soc, You}\n\tAverageCoreProgramHours > 17\n\t->  class 1  [0.753]\n\nRule 4/10: (93.4/19.8, lift 1.4)\n\tAgeAtCreation > 92\n\t->  class 0  [0.782]\n\nDefault class: 0\n\n-----  Trial 5:  -----\n\nRules:\n\nRule 5/1: (4073.5/1735.8, lift 1.0)\n\tHCW_Ratio <= 40\n\t->  class 0  [0.574]\n\nRule 5/2: (1153.9/572.4, lift 1.2)\n\tMostUsedBillingGrade in [BGrade2-BGrade9]\n\tClientType = HAC\n\tClient_Initiated_Cancellations <= 5\n\tHCW_Ratio <= 40\n\t->  class 1  [0.504]\n\nRule 5/3: (603.3/264.1, lift 1.3)\n\tAgeAtCreation > 59\n\tMostUsedBillingGrade in [BGrade2-BGrade9]\n\tClientType = HAC\n\tClient_Initiated_Cancellations <= 5\n\tdefault_contract_group = CHSP\n\tAverageCoreProgramHours <= 149\n\t->  class 1  [0.562]\n\nRule 5/4: (30.5, lift 1.7)\n\tClient_Initiated_Cancellations <= 5\n\tHCW_Ratio > 40\n\t->  class 0  [0.969]\n\nRule 5/5: (67.5/16.5, lift 1.8)\n\tMostUsedBillingGrade in [BGrade2-BGrade6]\n\tClientType = HAC\n\tClient_Initiated_Cancellations <= 5\n\tdefault_contract_group in {CHSP, Package, Private/Commercial}\n\tAverageCoreProgramHours <= 149\n\tHCW_Ratio > 12\n\t->  class 1  [0.748]\n\nRule 5/6: (303.6/92.9, lift 1.2)\n\tClient_Initiated_Cancellations > 5\n\t->  class 0  [0.693]\n\nRule 5/7: (25.7/4.5, lift 1.9)\n\tMostUsedBillingGrade in [BGrade2-BGrade6]\n\tClientType = HAC\n\tClient_Initiated_Cancellations <= 5\n\tdefault_contract_group = Private/Commercial\n\tAverageCoreProgramHours <= 149\n\tHCW_Ratio > 5\n\t->  class 1  [0.800]\n\nRule 5/8: (6.7/0.9, lift 1.8)\n\tClientType = HAC\n\tdefault_contract_group = Disability\n\t->  class 1  [0.776]\n\nDefault class: 0\n\n-----  Trial 6:  -----\n\nRules:\n\nRule 6/1: (1129/261.6, lift 1.3)\n\tAverageCoreProgramHours > 43\n\t->  class 0  [0.768]\n\nRule 6/2: (1108.8/518, lift 1.3)\n\tIssues_Raised > 0\n\tClient_Initiated_Cancellations <= 5\n\tdefault_contract_group in {CHSP, Package}\n\tAverageCoreProgramHours <= 43\n\t->  class 1  [0.533]\n\nRule 6/3: (1208.7/362.7, lift 1.2)\n\tIssues_Raised <= 0\n\tdefault_contract_group in {CHSP, Package}\n\tPCNeedsFlag = N\n\t->  class 0  [0.700]\n\nRule 6/4: (538.3/231.8, lift 1.4)\n\tIssues_Raised <= 0\n\tClient_Initiated_Cancellations <= 5\n\tdefault_contract_group = Private/Commercial\n\tHOME_STR_state in {ACT, NSW, QLD, SA, VIC}\n\tAverageCoreProgramHours <= 43\n\t->  class 1  [0.569]\n\nRule 6/5: (181.2/63, lift 1.1)\n\tIssues_Raised > 0\n\tdefault_contract_group = Private/Commercial\n\t->  class 0  [0.651]\n\nRule 6/6: (246.1/48.9, lift 1.4)\n\tClient_Initiated_Cancellations > 5\n\t->  class 0  [0.799]\n\nRule 6/7: (55.3/19.1, lift 1.6)\n\tIssues_Raised <= 0\n\tAgeAtCreation <= 84\n\tdefault_contract_group in {CHSP, Package}\n\tClient_Programs_count_at_observation_cutoff <= 0\n\tPCNeedsFlag = Y\n\tAverageCoreProgramHours <= 16.25\n\tHCW_Ratio <= 4\n\t->  class 1  [0.649]\n\nRule 6/8: (79.1/22.3, lift 1.8)\n\tIssues_Raised <= 0\n\tMostUsedBillingGrade in [BGrade1-BGrade2]\n\tClient_Initiated_Cancellations <= 5\n\tdefault_contract_group in {CHSP, Package}\n\tClient_Programs_count_at_observation_cutoff > 0\n\tAverageCoreProgramHours <= 43\n\tHCW_Ratio <= 4\n\t->  class 1  [0.713]\n\nRule 6/9: (44.6/10.4, lift 1.3)\n\tdefault_contract_group in {Disability, DVA/VHC, TransPac}\n\tAverageCoreProgramHours <= 43\n\t->  class 0  [0.756]\n\nRule 6/10: (351.7/69.3, lift 1.4)\n\tIssues_Raised <= 0\n\tAgeAtCreation > 84\n\tdefault_contract_group in {CHSP, Package}\n\t->  class 0  [0.801]\n\nRule 6/11: (469.9/56.9, lift 1.5)\n\tIssues_Raised <= 0\n\tdefault_contract_group in {CHSP, Package}\n\tHCW_Ratio > 4\n\t->  class 0  [0.877]\n\nRule 6/12: (9.8, lift 1.6)\n\tdefault_contract_group = Private/Commercial\n\tHOME_STR_state = WA\n\tAverageCoreProgramHours <= 43\n\t->  class 0  [0.915]\n\nDefault class: 0\n\n-----  Trial 7:  -----\n\nRules:\n\nRule 7/1: (2005.6/460.7, lift 1.2)\n\tAverageCoreProgramHours > 13.5\n\t->  class 0  [0.770]\n\nRule 7/2: (839.1/385.6, lift 1.7)\n\tMostUsedBillingGrade in [BGrade1-BGrade6]\n\tClient_Initiated_Cancellations <= 2\n\tdefault_contract_group in {CHSP, Disability, Package,\n                                   Private/Commercial}\n\tHOME_STR_state in {ACT, QLD, SA, WA}\n\tAverageCoreProgramHours <= 13.5\n\t->  class 1  [0.540]\n\nRule 7/3: (1001.6/227.4, lift 1.2)\n\tIssues_Raised <= 0.7666531\n\tHOME_STR_state in {NSW, VIC}\n\t->  class 0  [0.772]\n\nRule 7/4: (931.8/193, lift 1.3)\n\tClient_Initiated_Cancellations > 2\n\t->  class 0  [0.792]\n\nRule 7/5: (777.5/158.8, lift 1.3)\n\tAgeAtCreation > 81\n\tHOME_STR_state in {NSW, VIC}\n\t->  class 0  [0.795]\n\nRule 7/6: (191.9/83, lift 1.8)\n\tIssues_Raised > 0.7666531\n\tAgeAtCreation <= 81\n\tClient_Initiated_Cancellations <= 5\n\tdefault_contract_group in {CHSP, Package, Private/Commercial}\n\tHOME_STR_state = NSW\n\tAverageCoreProgramHours <= 13.5\n\t->  class 1  [0.567]\n\nRule 7/7: (26.8/5.2, lift 1.2)\n\tMostUsedBillingGrade = BGrade9\n\tClient_Initiated_Cancellations <= 2\n\t->  class 0  [0.783]\n\nRule 7/8: (31.5/4.8, lift 2.6)\n\tClientType in {Soc, You}\n\tAverageCoreProgramHours > 43.75\n\t->  class 1  [0.826]\n\nDefault class: 0\n\n-----  Trial 8:  -----\n\nRules:\n\nRule 8/1: (3732/1204, lift 1.1)\n\tdefault_contract_group in {CHSP, Disability, Package,\n                                   Private/Commercial}\n\t->  class 0  [0.677]\n\nRule 8/2: (345.3/96.3, lift 2.5)\n\tMostUsedBillingGrade in [BGrade5-BGrade9]\n\tdefault_contract_group in {CHSP, Disability, Package,\n                                   Private/Commercial}\n\tHOME_STR_state in {ACT, QLD, SA}\n\tAverageCoreProgramHours <= 37.5\n\t->  class 1  [0.720]\n\nRule 8/3: (50, lift 1.6)\n\tdefault_contract_group in {DVA/VHC, TransPac}\n\t->  class 0  [0.981]\n\nRule 8/4: (100.3/31.8, lift 2.3)\n\tIssues_Raised > 0.7666531\n\tAgeAtCreation <= 84\n\tClient_Initiated_Cancellations <= 3\n\tdefault_contract_group in {CHSP, Package, Private/Commercial}\n\tClient_Programs_count_at_observation_cutoff <= 1\n\tAverageCoreProgramHours > 30.75\n\tAverageCoreProgramHours <= 37.5\n\t->  class 1  [0.679]\n\nRule 8/5: (64.6/14.6, lift 2.6)\n\tIssues_Raised <= 0.7666531\n\tAgeAtCreation > 70\n\tAgeAtCreation <= 73\n\tMostUsedBillingGrade in [BGrade1-BGrade3]\n\tClient_Initiated_Cancellations <= 3\n\tPCNeedsFlag = N\n\tAverageCoreProgramHours <= 7.5\n\t->  class 1  [0.765]\n\nRule 8/6: (286.4/127.7, lift 1.9)\n\tAgeAtCreation <= 84\n\tClient_Initiated_Cancellations <= 3\n\tdefault_contract_group in {CHSP, Disability, Package,\n                                   Private/Commercial}\n\tPCNeedsFlag = Y\n\tAverageCoreProgramHours <= 37.5\n\t->  class 1  [0.554]\n\nRule 8/7: (259.5/108.5, lift 2.0)\n\tAgeAtCreation <= 84\n\tClient_Initiated_Cancellations <= 2\n\tdefault_contract_group in {CHSP, Disability, Package,\n                                   Private/Commercial}\n\tPCNeedsFlag = Y\n\tAverageCoreProgramHours <= 37.5\n\t->  class 1  [0.581]\n\nRule 8/8: (25.4/4, lift 2.8)\n\tIssues_Raised > 0.7666531\n\tAgeAtCreation <= 58\n\tMostUsedBillingGrade in [BGrade1-BGrade3]\n\tClient_Initiated_Cancellations <= 3\n\tPCNeedsFlag = N\n\tAverageCoreProgramHours <= 37.5\n\t->  class 1  [0.819]\n\nRule 8/9: (24.7/6.8, lift 2.4)\n\tClientType in {Soc, You}\n\tAverageCoreProgramHours > 37.5\n\t->  class 1  [0.709]\n\nRule 8/10: (344.2/15.1, lift 1.5)\n\tClient_Initiated_Cancellations > 3\n\t->  class 0  [0.954]\n\nDefault class: 0\n\n-----  Trial 9:  -----\n\nRules:\n\nRule 9/1: (1499.5/109.4, lift 1.5)\n\tClientType in {Bro, CAC, CCP, Com, Dem, Dis, Dom, EAC, HAC, NRC, Per,\n                       Pri, Res, Soc, TAC, TCP, VHC}\n\tAverageCoreProgramHours > 15.75\n\t->  class 0  [0.926]\n\nRule 9/2: (849.5/83.8, lift 1.4)\n\tIssues_Raised <= 0\n\tdefault_contract_group in {CHSP, Package}\n\tClient_Programs_count_at_observation_cutoff <= 0\n\t->  class 0  [0.900]\n\nRule 9/3: (365.8/99.9, lift 3.2)\n\tIssues_Raised > 0\n\tAgeAtCreation <= 83\n\tClientType in {Dom, EAC, HAC, Per}\n\tClient_Initiated_Cancellations <= 1\n\tdefault_contract_group in {CHSP, Package}\n\tAverageCoreProgramHours <= 13.75\n\t->  class 1  [0.726]\n\nRule 9/4: (906.1/50.3, lift 1.5)\n\tAgeAtCreation > 83\n\t->  class 0  [0.943]\n\nRule 9/5: (278.6/98.4, lift 2.8)\n\tIssues_Raised <= 0\n\tAgeAtCreation <= 83\n\tClientType in {Com, HAC, Pri}\n\tdefault_contract_group = Private/Commercial\n\tAverageCoreProgramHours <= 9.25\n\t->  class 1  [0.646]\n\nRule 9/6: (281.3/52.2, lift 3.6)\n\tIssues_Raised > 0\n\tAgeAtCreation <= 73\n\tClientType in {Com, EAC, HAC}\n\tdefault_contract_group in {CHSP, Package}\n\tHOME_STR_state in {NSW, QLD, SA, WA}\n\tAverageCoreProgramHours <= 13.75\n\t->  class 1  [0.812]\n\nRule 9/7: (612.5/40.1, lift 1.5)\n\tAgeAtCreation > 73\n\tClientType in {Com, EAC, HAC, Per}\n\tClient_Initiated_Cancellations > 1\n\tAverageCoreProgramHours > 0.6666667\n\t->  class 0  [0.933]\n\nRule 9/8: (111.1/27.6, lift 3.3)\n\tIssues_Raised <= 0\n\tAgeAtCreation <= 83\n\tdefault_contract_group in {CHSP, Private/Commercial}\n\tClient_Programs_count_at_observation_cutoff > 0\n\tAverageCoreProgramHours <= 15.75\n\t->  class 1  [0.747]\n\nRule 9/9: (126.7/16.8, lift 1.4)\n\tIssues_Raised > 0\n\tClientType in {Com, EAC, HAC}\n\tdefault_contract_group = Private/Commercial\n\tHOME_STR_state in {NSW, QLD, SA, WA}\n\t->  class 0  [0.862]\n\nRule 9/10: (37.1/0.6, lift 4.2)\n\tIssues_Raised > 0\n\tAgeAtCreation <= 83\n\tClientType in {Dom, Pri}\n\tAverageCoreProgramHours <= 15.75\n\t->  class 1  [0.958]\n\nRule 9/11: (62.9, lift 1.5)\n\tIssues_Raised <= 0\n\tdefault_contract_group = Private/Commercial\n\tClient_Programs_count_at_observation_cutoff <= 0\n\tAverageCoreProgramHours > 9.25\n\t->  class 0  [0.985]\n\nRule 9/12: (1419/103.2, lift 1.5)\n\tHOME_STR_state in {NSW, QLD, SA, WA}\n\tAverageCoreProgramHours > 13.75\n\t->  class 0  [0.927]\n\nRule 9/13: (42.8, lift 1.5)\n\tClientType in {Bro, CAC, CCP, Dem, Dis, DVA, Nur, Res, Soc, VHC}\n\tAverageCoreProgramHours <= 15.75\n\t->  class 0  [0.978]\n\nRule 9/14: (47.9/3.4, lift 4.0)\n\tAgeAtCreation <= 83\n\tAverageCoreProgramHours <= 0.6666667\n\t->  class 1  [0.911]\n\nRule 9/15: (21.4, lift 4.2)\n\tIssues_Raised <= 0\n\tAgeAtCreation <= 83\n\tClientType in {Com, HAC, Per, Pri}\n\tHOME_STR_state in {ACT, QLD, SA}\n\tPCNeedsFlag = Y\n\tAverageCoreProgramHours <= 15.75\n\t->  class 1  [0.957]\n\nRule 9/16: (8.2, lift 4.0)\n\tClientType = You\n\tClient_Initiated_Cancellations <= 3\n\t->  class 1  [0.902]\n\nRule 9/17: (301, lift 1.6)\n\tClient_Initiated_Cancellations > 3\n\t->  class 0  [0.997]\n\nRule 9/18: (5.2, lift 3.8)\n\tdefault_contract_group = Disability\n\tAverageCoreProgramHours <= 15.75\n\t->  class 1  [0.861]\n\nRule 9/19: (45.7, lift 1.5)\n\tdefault_contract_group in {DVA/VHC, TransPac}\n\t->  class 0  [0.979]\n\nDefault class: 0\n\n\nEvaluation on training data (4130 cases):\n\nTrial\t        Rules     \n-----\t  ----------------\n\t    No      Errors\n\n   0\t    15  691(16.7%)\n   1\t     5  956(23.1%)\n   2\t     7  822(19.9%)\n   3\t    10  915(22.2%)\n   4\t    10 1013(24.5%)\n   5\t     8  957(23.2%)\n   6\t    12  960(23.2%)\n   7\t     8  786(19.0%)\n   8\t    10  721(17.5%)\n   9\t    19  709(17.2%)\nboost\t        663(16.1%)   <<\n\n\n\t   (a)   (b)    <-classified as\n\t  ----  ----\n\t  3200   104    (a): class 0\n\t   559   267    (b): class 1\n\n\n\tAttribute usage:\n\n\t100.00%\tIssues_Raised\n\t100.00%\tdefault_contract_group\n\t100.00%\tAverageCoreProgramHours\n\t 99.56%\tHOME_STR_state\n\t 99.10%\tClientType\n\t 99.10%\tHCW_Ratio\n\t 98.72%\tMostUsedBillingGrade\n\t 81.65%\tAgeAtCreation\n\t 74.29%\tClient_Initiated_Cancellations\n\t 43.68%\tClient_Programs_count_at_observation_cutoff\n\t 43.12%\tPCNeedsFlag\n\n\nTime: 0.2 secs\n"
    ## 
    ## $call
    ## C5.0.default(x = train[, -ndxLabel], y = train$Label, trials = C5.0Trials_param, 
    ##     rules = TRUE, control = C5.0Control(bands = 100, earlyStopping = TRUE))

``` r
# make predictions
pr.kc.c50 <-
  predict(kc.c50, type = "prob", newdata = test[-ndxLabel])[, 2]
fitted.results.c50 <- ifelse(pr.kc.c50 > fitThreshold, 1, 0)
confusion_maxtix <- table(test$Label, fitted.results.c50)
#print.table(confusion_maxtix)
kc.c50.eval.results <- computePerformanceMeasures(confusion_maxtix, fScoreB_param)


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
```

![](kc_client_churn_FINAL_F_files/figure-markdown_github/cars-5.png)

``` r
#Compare AUC of 2 Curves using prediction probabilities
rocGLM <- roc(test.scaled$Label, pr.kc.glm)
rocRF <- roc(test$Label, pr.kc.rf)
rocC50 <- roc(test$Label, pr.kc.c50)
kc.roc.eval.results <- c(rocGLM$auc,rocRF$auc,rocC50$auc)

roctest.eval.GLM.RF <- roc.test(rocGLM, rocRF,method="delong",paired=TRUE)
roctest.eval.GLM.c50 <- roc.test(rocGLM, rocC50,method="delong",paired=TRUE)
roctest.eval.RF.c50 <- roc.test (rocRF, rocC50, method="delong",paired=TRUE)
roc.test.eval <- c(roctest.eval.GLM.RF$p.value, roctest.eval.GLM.c50$p.value, roctest.eval.RF.c50$p.value)

#======================
## 10 FOLD CROSS VALIDATION

kFoldValidation_param  <- 10
k <- kFoldValidation_param

data <- rbind(train, test)
set.seed(1)
data$id <- sample(1:kFoldValidation_param, nrow(data), replace = TRUE)
ndxId <- which(names(data)=='id')

list <- 1:kFoldValidation_param

# prediction and testset data frames that we add to with each iteration over the folds
cv.prediction.glm <- data.frame()
cv.prediction.rf <- data.frame()
cv.prediction.c50 <- data.frame()

testsetCopy <- data.frame()

#Creating a progress bar to know the status of CV
progress.bar <- create_progress_bar("text")
progress.bar$init(k)
```

    ## 
      |                                                                       
      |                                                                 |   0%

``` r
# options(warn=-1)
for (i in 1:k) {
  # remove rows with id i from dataframe to create training set
  # select rows with id i to create test set
  trainingset <- subset(data, id %in% list[-i])
  testset <- subset(data, id %in% c(i))
  trainingset.scaled <- scaleNumbers(trainingset[,-ndxId])
  testset.scaled <- scaleNumbers(testset[,-ndxId])

  for (feature in model_features) {
    if (is.factor(trainingset[[feature]]) | is.ordered(trainingset[[feature]])) {
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
```

    ## 
      |                                                                       
      |======                                                           |  10%
      |                                                                       
      |=============                                                    |  20%
      |                                                                       
      |====================                                             |  30%
      |                                                                       
      |==========================                                       |  40%
      |                                                                       
      |================================                                 |  50%
      |                                                                       
      |=======================================                          |  60%
      |                                                                       
      |==============================================                   |  70%
      |                                                                       
      |====================================================             |  80%
      |                                                                       
      |==========================================================       |  90%
      |                                                                       
      |=================================================================| 100%

``` r
# options(warn=-0)

# add predictions and actual churn labels
result.glm <- cbind(cv.prediction.glm, testsetCopy)
result.rf  <- cbind(cv.prediction.rf, testsetCopy)
result.c50 <- cbind(cv.prediction.c50, testsetCopy)

names(result.glm) <- c("Predicted", "Actual")
names(result.rf)  <- c("Predicted", "Actual")
names(result.c50) <- c("Predicted", "Actual")

# AUC as CV criterion
cv.rocGLM <- roc(result.glm$Actual, result.glm$Predicted)
cv.rocRF  <- roc(result.rf$Actual, result.rf$Predicted)
cv.rocC50 <- roc(result.c50$Actual, result.c50$Predicted)

cv.roctest.eval.GLM.RF <- roc.test(cv.rocGLM, cv.rocRF,method="delong",paired=TRUE)
cv.roctest.eval.GLM.c50 <- roc.test(cv.rocGLM, cv.rocC50,method="delong",paired=TRUE)
cv.roctest.eval.RF.c50 <- roc.test (cv.rocRF, cv.rocC50, method="delong",paired=TRUE)

## Render 10-X Validation ROC graphs

pred.kc.glm <-  prediction(result.glm$Predicted, result.glm$Actual)
pred.kc.rf <-  prediction(result.rf$Predicted, result.rf$Actual)
pred.kc.c50 <-  prediction(result.c50$Predicted, result.c50$Actual)

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
```

![](kc_client_churn_FINAL_F_files/figure-markdown_github/cars-6.png)

``` r
# use Mean Prediction Accuracy as CV criteria
print (paste ('CV Probability fitThreshold used is ',fitThreshold))
```

    ## [1] "CV Probability fitThreshold used is  0.5"

``` r
cv.fitted.result.glm <-
  ifelse(result.glm$Predicted > fitThreshold, 1, 0)
result.glm$Difference <-
  ifelse(result.glm$Actual == cv.fitted.result.glm, 1, 0)

cv.fitted.result.rf <-
  ifelse(result.rf$Predicted > fitThreshold, 1, 0)
result.rf$Difference <-
  ifelse(result.rf$Actual == cv.fitted.result.rf, 1, 0)

cv.fitted.result.c50 <-
  ifelse(result.c50$Predicted > fitThreshold, 1, 0)
result.c50$Difference <-
  ifelse(result.c50$Actual == cv.fitted.result.c50, 1, 0)

kc.cv.roc.eval.results <- c(mean(result.glm$Difference), mean(result.rf$Difference),mean(result.c50$Difference))

# Tweak probability threshold of selected model - RF
pr.kc.rf <- predict(kc.rf, newdata = test, type = 'prob')[, 2]
x <- cbind(test$Label,pr.kc.rf)
y <- subset(x,x[,1]==2)
summary(y[,2])
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##  0.0020  0.1088  0.2100  0.3148  0.4185  0.9870

``` r
fitThreshold.adj <- summary(y[,2])[4] #Mean Probability value
print(paste('New RF prediction prob. threshold set to min probabilty of churn class',fitThreshold.adj))
```

    ## [1] "New RF prediction prob. threshold set to min probabilty of churn class 0.3148"

``` r
fitted.results.rf <- ifelse(pr.kc.rf > fitThreshold.adj, 1, 0)
confusion_maxtix <- table(test$Label, fitted.results.rf)
kc.rf.eval.results.adj <- c(computePerformanceMeasures(confusion_maxtix, fScoreB_param),NA,NA)

# Tweak probability threshold of selected model - C50
pr.kc.c50.train <- predict(kc.c50, type = "prob", newdata = train[-ndxLabel])[, 2]
x <- cbind(test$Label,pr.kc.c50)
y <- subset(x,x[,1]==2)
summary(y[,2])
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ## 0.00000 0.07283 0.17390 0.24590 0.38900 0.82680

``` r
fitThreshold.adj <- summary(y[,2])[4]  #Mean Probability value
print(paste('New C5.0 prediction prob. threshold set to min probabilty of churn class',fitThreshold.adj))
```

    ## [1] "New C5.0 prediction prob. threshold set to min probabilty of churn class 0.2459"

``` r
fitted.results.c50 <- ifelse(pr.kc.c50 > fitThreshold.adj, 1, 0)
confusion_maxtix <- table(test$Label, fitted.results.c50)
kc.c50.eval.results.adj <- c(computePerformanceMeasures(confusion_maxtix, fScoreB_param),NA,NA)

### Construct the Performance Matrix
kc.eval.matrix <- data.frame()
kc.eval.matrix <- rbind(kc.glm.eval.results,kc.rf.eval.results,kc.c50.eval.results)
kc.eval.matrix <- cbind(kc.eval.matrix,kc.roc.eval.results, kc.cv.roc.eval.results)
kc.eval.matrix <- rbind(kc.eval.matrix,kc.rf.eval.results.adj,kc.c50.eval.results.adj )

rownames(kc.eval.matrix)  <-  c('Logit', 'RF','C50','Adjusted RF','Adjusted C5.0')
colnames(kc.eval.matrix)  <- c('Accuracy','Precision','Recall','F-score','AUC', '10X-Val Accuracy')

roc.test.eval <- data.frame()
cv.roc.test.eval <- data.frame()

roc.test.eval <- c(roctest.eval.GLM.RF$p.value, roctest.eval.GLM.c50$p.value, roctest.eval.RF.c50$p.value)
cv.roc.test.eval <- c(cv.roctest.eval.GLM.RF$p.value, cv.roctest.eval.GLM.c50$p.value, cv.roctest.eval.RF.c50$p.value)
kc.roc.test.eval <- cbind(roc.test.eval, cv.roc.test.eval)

rownames(kc.roc.test.eval)  <-  c('Logit to RF', 'Logit to C50','RF to C50')
colnames(kc.roc.test.eval) <- c('Test','10-X-Val Test' )

print ('Performance Matrix')
```

    ## [1] "Performance Matrix"

``` r
print (kc.eval.matrix)
```

    ##               Accuracy Precision Recall   F-score       AUC
    ## Logit            0.832 0.7352941  0.250 0.3731343 0.7993187
    ## RF               0.831 0.8039216  0.205 0.3266932 0.7817437
    ## C50              0.838 0.9523810  0.200 0.3305785 0.7793375
    ## Adjusted RF      0.827 0.6173913  0.355 0.4507937        NA
    ## Adjusted C5.0    0.812 0.5405405  0.400 0.4597701        NA
    ##               10X-Val Accuracy
    ## Logit                0.8161793
    ## RF                   0.8259259
    ## C50                  0.8282651
    ## Adjusted RF                 NA
    ## Adjusted C5.0               NA

``` r
print ('Test for Significance(delong) between Paired Models p-Value')
```

    ## [1] "Test for Significance(delong) between Paired Models p-Value"

``` r
print (kc.roc.test.eval)
```

    ##                   Test 10-X-Val Test
    ## Logit to RF  0.2912408  0.0003466975
    ## Logit to C50 0.2074172  0.0046163407
    ## RF to C50    0.8668279  0.3040373435
