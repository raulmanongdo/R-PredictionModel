kc\_client\_churn\_FINAL1A
================

``` r
# ***********************************************************************************************
#  R  Program : kc_client_churn_FINAL1A.R
#
#  Author     : Raul Manongdo, University of Technology, Sydney
#               Advance Analytics Institute
#  Date       : April 2017

#  Program description:
#     Birary client churn prediction modelling for anonymous company using 3 statistical models;
#     Logistic Regression, Random Forest and C5.0 Decison Trees.
#     Feature selection using GLM + RF models and Pearson correlation analysis.
#     Train/Test set is based on pre-defined observation and labeling window time periods, as separate datasets.
#     Threshold for prediction accuracy is set at .6 instead of default .5
#     Model comparison by area under ROC and test of signifance using Delong method
#     Subsequent model selection by F2 and Accuracy score.
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
library(stargazer)
```

    ## 
    ## Please cite as:

    ##  Hlavac, Marek (2015). stargazer: Well-Formatted Regression and Summary Statistics Tables.

    ##  R package version 5.2. http://CRAN.R-project.org/package=stargazer

``` r
#**************************
#returns model performance measures
#**************************

computePerformanceMeasures <- function (confusion_maxtix, b = 2) {
  tn <- confusion_maxtix[1, 1]
  fp <- confusion_maxtix[1, 2]
  fn <- confusion_maxtix[2, 1]
  tp <- confusion_maxtix[2, 2]
  acc <- (tp + tn) / (tp + tn + fp + fn)
  prec <- (tp / (tp + fp))
  rec  <- (tp / (tp + fn))
  b < ifelse(is.null(b), 1, b)
  fScore <-  (1 + (b * b)) * (prec + rec) / ((b * b * prec) + rec)
  return (c(acc, prec, rec, fScore))
}

selectRawDataVariables <- function (raw.data) {
  # Remove unique key values = 1
  raw.data <-
    raw.data[-which(names(raw.data) == 'myUniqueClientID')]
  
  # Remove attributes that have more than 50% missing values
  n50pcnt <- round(nrow(raw.data) * .50)
  x <- sapply(raw.data, function(x) {
    sum(is.na(x)) + sum(is.nan(x))
  })
  drops <- x[x > n50pcnt]
  raw.data <- raw.data[,!(names(raw.data) %in% names(drops))]
  
  # Remove attributes with only 1 value fo entire population
  x1 <- sapply(raw.data, function(x)
    length(unique(x)))
  drops <- x1[x1 == 1]
  raw.data <- raw.data[,!(names(raw.data) %in% names(drops))]
  
  return (names(raw.data))
}

#**************************
#returns a cleansed raw dataset
#**************************
cleanRawData <- function(raw.data, toScale = FALSE) {
  x <- sapply(raw.data, function(x)
    class(x))
  char_vars <- x[x == "character"]
  all_num_vars <- x[x %in% c("integer", "numeric", "matrix")]
  
  # Impute missing values that takes into account of correlations between the missings
  # or the correlations of the measured, but won't seriously inflate the significance of the results.
  for (var in names(all_num_vars)) {
    raw.data[[var]][is.na(raw.data[[var]])] <-
      mean(raw.data[[var]], na.rm = TRUE) + rnorm(sum(is.na(raw.data[[var]]))) * sd(raw.data[[var]], na.rm = T)
  }
  
  # Assign lower and upper quantile values to numeric outliers
  for (var in names(all_num_vars)) {
    lowerq <- quantile(raw.data[[var]], na.rm = TRUE)[1]
    upperq <- quantile(raw.data[[var]], na.rm = TRUE)[4]
    iqr <-
      quantile(raw.data[[var]], na.rm = TRUE)[4] - quantile(raw.data[[var]], na.rm =
                                                              TRUE)[2]
    extreme.threshold.upper <- upperq + (iqr * 1.5)
    extreme.threshold.lower <- lowerq - (iqr * 1.5)
    raw.data[[var]][raw.data[[var]] > extreme.threshold.upper] <-
      extreme.threshold.upper
    raw.data[[var]][raw.data[[var]] < extreme.threshold.lower] <-
      extreme.threshold.lower
  }
  
  # z-score normalization of numeric variables
  # For this particular program, toScale param is always 'No', no normalisation occurs
  if (toScale) {
    options(max.print = 100000)
    for (var in names(all_num_vars))
      raw.data[[var]] <-
        scale(raw.data[[var]], center = TRUE, scale = TRUE)
  }
  
  #  Assign most used value to missing categorical attributes
  for (var in names(char_vars)) {
    tmp <- table(raw.data[[var]], useNA = "no")
    default_val <- names(tmp[which(tmp == max(tmp))])
    raw.data[[var]][is.na(raw.data[[var]]) |
                      is.nan(raw.data[[var]])] <- default_val
  }
  
  # Convert categorical variables to factors
  for (var in names(char_vars)) {
    raw.data[[var]] <- as.factor(as.character(raw.data[[var]]))
  }
  
  raw.data$Label <- ifelse (raw.data$Label == 'Churn', 1, 0)
  raw.data$Label <- as.factor(raw.data$Label)
  
  sapply(raw.data, function(x) {
    sum(is.na(x)) + sum(is.nan(x))
  })
  return(raw.data)
}

# -----------------------------------------------------------------------------
# LOAD RAW DATA
setwd("/Users/raulmanongdo/Documents/R-KinCare-FINAL/")
raw.data.train  <-
  read.csv(
    "KinCareTraing.csv",
    header = TRUE,
    na.strings = c("", "NA", "<NA>"),
    stringsAsFactors = FALSE
  )
raw.data.test  <-
  read.csv(
    "KinCareTesting.csv",
    header = TRUE,
    na.strings = c("", "NA", "<NA>"),
    stringsAsFactors = FALSE
  )
raw.data.train$role <- 'Train'
raw.data.test$role <-  'Test'
raw.data <- rbind(raw.data.train, raw.data.test)
summary(raw.data)
```

    ##  myUniqueClientID HOME_STR_state         Sex             ClientType       
    ##  Min.   :  105    Length:5130        Length:5130        Length:5130       
    ##  1st Qu.:10039    Class :character   Class :character   Class :character  
    ##  Median :15866    Mode  :character   Mode  :character   Mode  :character  
    ##  Mean   :17069                                                            
    ##  3rd Qu.:22335                                                            
    ##  Max.   :48723                                                            
    ##                                                                           
    ##     Grade           SmokerAccepted     GenderRequired    
    ##  Length:5130        Length:5130        Length:5130       
    ##  Class :character   Class :character   Class :character  
    ##  Mode  :character   Mode  :character   Mode  :character  
    ##                                                          
    ##                                                          
    ##                                                          
    ##                                                          
    ##  EthnicGroupRequired SpokenLanguageRequired ClientAge     
    ##  Length:5130         Length:5130            Mode:logical  
    ##  Class :character    Class :character       NA's:5130     
    ##  Mode  :character    Mode  :character                     
    ##                                                           
    ##                                                           
    ##                                                           
    ##                                                           
    ##  AssDaysAfterCreation AgeAtCreation     Responses_1    Responses_2   
    ##  Min.   : 0.00        Min.   :  0.00   Min.   :1.00   Min.   :2.000  
    ##  1st Qu.: 0.00        1st Qu.: 73.00   1st Qu.:4.00   1st Qu.:4.000  
    ##  Median : 0.00        Median : 82.00   Median :4.00   Median :4.000  
    ##  Mean   :16.50        Mean   : 78.83   Mean   :4.33   Mean   :4.406  
    ##  3rd Qu.:20.75        3rd Qu.: 87.00   3rd Qu.:5.00   3rd Qu.:5.000  
    ##  Max.   :89.00        Max.   :115.00   Max.   :5.00   Max.   :5.000  
    ##  NA's   :4780                          NA's   :4576   NA's   :4576   
    ##   Responses_3     Responses_4     Responses_5     Responses_6   
    ##  Min.   :2.000   Min.   :1.000   Min.   :1.000   Min.   :1.000  
    ##  1st Qu.:4.000   1st Qu.:4.000   1st Qu.:4.000   1st Qu.:4.000  
    ##  Median :4.000   Median :4.000   Median :5.000   Median :4.000  
    ##  Mean   :4.388   Mean   :4.359   Mean   :4.471   Mean   :4.071  
    ##  3rd Qu.:5.000   3rd Qu.:5.000   3rd Qu.:5.000   3rd Qu.:5.000  
    ##  Max.   :5.000   Max.   :5.000   Max.   :5.000   Max.   :5.000  
    ##  NA's   :4576    NA's   :4576    NA's   :4576    NA's   :4598   
    ##   Responses_7    Responses_8        Responses_9         Responses_10 
    ##  Min.   :1.000   Length:5130        Length:5130        Min.   :1.00  
    ##  1st Qu.:4.000   Class :character   Class :character   1st Qu.:4.00  
    ##  Median :4.000   Mode  :character   Mode  :character   Median :4.00  
    ##  Mean   :4.005                                         Mean   :4.33  
    ##  3rd Qu.:5.000                                         3rd Qu.:5.00  
    ##  Max.   :5.000                                         Max.   :5.00  
    ##  NA's   :4711                                          NA's   :4576  
    ##  AllRecordsNums    CoreProgramsNums CoreRecordNums   
    ##  Min.   :    1.0   Min.   :1.000    Min.   :    1.0  
    ##  1st Qu.:    9.0   1st Qu.:1.000    1st Qu.:    9.0  
    ##  Median :   35.0   Median :1.000    Median :   34.0  
    ##  Mean   :  138.4   Mean   :1.309    Mean   :  136.5  
    ##  3rd Qu.:  100.0   3rd Qu.:1.000    3rd Qu.:   99.0  
    ##  Max.   :11064.0   Max.   :7.000    Max.   :11064.0  
    ##                                                      
    ##  TotalCoreProgramHours MaxCoreProgramHours MinCoreProgramHours
    ##  Min.   :    0.25      Min.   :  0.250     Min.   :-22.0000   
    ##  1st Qu.:   12.50      1st Qu.:  2.000     1st Qu.:  0.5000   
    ##  Median :   48.50      Median :  3.000     Median :  0.5000   
    ##  Mean   :  214.87      Mean   :  3.607     Mean   :  0.7868   
    ##  3rd Qu.:  143.00      3rd Qu.:  3.000     3rd Qu.:  1.0000   
    ##  Max.   :73077.33      Max.   :728.000     Max.   : 24.0000   
    ##  NA's   :4             NA's   :4           NA's   :4          
    ##  AverageCoreProgramHours AverageCoreServiceHours CoreRecordsRate 
    ##  Min.   :    0.25        Min.   : 0.250          Min.   :0.0000  
    ##  1st Qu.:   11.00        1st Qu.: 1.016          1st Qu.:1.0000  
    ##  Median :   38.00        Median : 1.464          Median :1.0000  
    ##  Mean   :  175.90        Mean   : 1.655          Mean   :0.9567  
    ##  3rd Qu.:  107.23        3rd Qu.: 1.875          3rd Qu.:1.0000  
    ##  Max.   :73077.33        Max.   :24.000          Max.   :1.0000  
    ##  NA's   :4               NA's   :4                               
    ##   CoreTotalKM       FirstCoreServiceDelayDays FirstCoreServiceHours
    ##  Min.   :    0.00   Min.   : -995.0           Min.   :0            
    ##  1st Qu.:   22.99   1st Qu.:    0.0           1st Qu.:0            
    ##  Median :  130.47   Median :    3.0           Median :0            
    ##  Mean   :  634.03   Mean   :  156.4           Mean   :0            
    ##  3rd Qu.:  442.86   3rd Qu.:    8.0           3rd Qu.:0            
    ##  Max.   :44144.31   Max.   :32153.0           Max.   :0            
    ##                                                                    
    ##  LastCoreServiceHours NoneCoreProgramsNums NoneCoreRecordNums
    ##  Min.   :0            Min.   :1.000        Min.   :  1.00    
    ##  1st Qu.:0            1st Qu.:1.000        1st Qu.:  6.00    
    ##  Median :0            Median :1.000        Median : 27.50    
    ##  Mean   :0            Mean   :1.027        Mean   : 43.66    
    ##  3rd Qu.:0            3rd Qu.:1.000        3rd Qu.: 65.00    
    ##  Max.   :0            Max.   :2.000        Max.   :195.00    
    ##                       NA's   :4908         NA's   :4908      
    ##  TotalNoneCoreProgramHours MaxNoneCoreProgramHours MinNoneCoreProgramHours
    ##  Min.   :   0.333          Min.   : 0.333          Min.   : 0.000         
    ##  1st Qu.:   8.562          1st Qu.: 1.500          1st Qu.: 0.500         
    ##  Median :  36.500          Median : 2.500          Median : 0.500         
    ##  Mean   :  60.305          Mean   : 2.988          Mean   : 1.102         
    ##  3rd Qu.:  87.688          3rd Qu.: 3.000          3rd Qu.: 1.000         
    ##  Max.   :1710.983          Max.   :24.000          Max.   :24.000         
    ##  NA's   :4908              NA's   :4908            NA's   :4908           
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
    ##  FrequentschedStatusGroup MostUsedBillingGrade MostUsedPayGrade  
    ##  Length:5130              Length:5130          Length:5130       
    ##  Class :character         Class :character     Class :character  
    ##  Mode  :character         Mode  :character     Mode  :character  
    ##                                                                  
    ##                                                                  
    ##                                                                  
    ##                                                                  
    ##  RespiteNeedsFlag   DANeedsFlag        NCNeedsFlag       
    ##  Length:5130        Length:5130        Length:5130       
    ##  Class :character   Class :character   Class :character  
    ##  Mode  :character   Mode  :character   Mode  :character  
    ##                                                          
    ##                                                          
    ##                                                          
    ##                                                          
    ##  PCNeedsFlag        SocialNeedsFlag    TransportNeedsFlag
    ##  Length:5130        Length:5130        Length:5130       
    ##  Class :character   Class :character   Class :character  
    ##  Mode  :character   Mode  :character   Mode  :character  
    ##                                                          
    ##                                                          
    ##                                                          
    ##                                                          
    ##  RequiredWorkersFlag PreferredWorkersFlag
    ##  Length:5130         Length:5130         
    ##  Class :character    Class :character    
    ##  Mode  :character    Mode  :character    
    ##                                          
    ##                                          
    ##                                          
    ##                                          
    ##  Client_Programs_count_at_observation_cutoff default_contract_group
    ##  Min.   :0.0000                              Length:5130           
    ##  1st Qu.:0.0000                              Class :character      
    ##  Median :1.0000                              Mode  :character      
    ##  Mean   :0.6801                                                    
    ##  3rd Qu.:1.0000                                                    
    ##  Max.   :8.0000                                                    
    ##  NA's   :182                                                       
    ##  TotalDaysWithKincare_all_Programs complainttier      Issues_Raised    
    ##  Min.   :   0.0                    Length:5130        Min.   : 0.0000  
    ##  1st Qu.:  83.0                    Class :character   1st Qu.: 0.0000  
    ##  Median : 223.0                    Mode  :character   Median : 0.0000  
    ##  Mean   : 459.5                                       Mean   : 0.7667  
    ##  3rd Qu.: 518.0                                       3rd Qu.: 1.0000  
    ##  Max.   :5939.0                                       Max.   :33.0000  
    ##  NA's   :2857                                         NA's   :236      
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
    ##  avg_first_service_days_all_programs   HCW_Ratio          Label          
    ##  Min.   :    1.0                     Min.   :  0.000   Length:5130       
    ##  1st Qu.:    4.0                     1st Qu.:  2.000   Class :character  
    ##  Median :   12.0                     Median :  5.000   Mode  :character  
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
vars <-  selectRawDataVariables (raw.data)
d1 <- cleanRawData(raw.data[vars], toScale = FALSE)

#=========================
#FEATURE SELECTION
ndx <- which(names(d1) == 'role')
d2 <- d1[, -ndx]

set.seed(1)
mod_d2 <-
  glm(Label ~ ., family = binomial(link = "logit"), data = d2)
summary(mod_d2)
```

    ## 
    ## Call:
    ## glm(formula = Label ~ ., family = binomial(link = "logit"), data = d2)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -2.3063  -0.6481  -0.4503  -0.2057   2.9462  
    ## 
    ## Coefficients: (1 not defined because of singularities)
    ##                                               Estimate Std. Error z value
    ## (Intercept)                                  1.801e+01  9.101e+02   0.020
    ## HOME_STR_stateNSW                           -6.449e-01  2.359e-01  -2.734
    ## HOME_STR_stateQLD                           -8.300e-02  2.699e-01  -0.307
    ## HOME_STR_stateSA                            -1.494e-01  2.604e-01  -0.574
    ## HOME_STR_stateVIC                           -1.117e+00  4.432e-01  -2.521
    ## HOME_STR_stateWA                            -3.764e-01  3.300e-01  -1.141
    ## ClientTypeCAC                                1.401e+00  1.169e+00   1.199
    ## ClientTypeCCP                                1.286e+00  1.133e+00   1.135
    ## ClientTypeCom                               -2.973e-01  1.104e+00  -0.269
    ## ClientTypeDem                               -1.350e+01  3.639e+02  -0.037
    ## ClientTypeDis                               -4.704e-02  1.405e+00  -0.033
    ## ClientTypeDom                                1.084e+00  1.125e+00   0.963
    ## ClientTypeDVA                               -9.597e+00  7.941e+02  -0.012
    ## ClientTypeEAC                                1.469e+00  1.120e+00   1.312
    ## ClientTypeHAC                                9.402e-01  1.092e+00   0.861
    ## ClientTypeNRC                                1.624e+00  1.582e+00   1.026
    ## ClientTypeNur                               -1.133e+00  1.633e+00  -0.694
    ## ClientTypePer                                7.482e-01  1.181e+00   0.634
    ## ClientTypePri                                6.435e-01  1.128e+00   0.571
    ## ClientTypeRes                                9.227e-01  1.402e+00   0.658
    ## ClientTypeSoc                                1.565e+00  1.220e+00   1.283
    ## ClientTypeTAC                                5.612e-01  1.364e+00   0.411
    ## ClientTypeTCP                               -1.265e+01  7.597e+02  -0.017
    ## ClientTypeVHC                               -1.241e+01  9.604e+02  -0.013
    ## ClientTypeYou                                3.795e+00  1.450e+00   2.617
    ## GradeGrade2                                 -1.066e-01  1.446e-01  -0.737
    ## GradeGrade3                                 -1.074e-01  2.399e-01  -0.448
    ## GradeGrade4                                 -5.971e-01  8.328e-01  -0.717
    ## GradeGrade5                                 -1.907e-01  4.405e-01  -0.433
    ## GradeGrade6                                  2.995e-01  2.829e-01   1.059
    ## SmokerAcceptedYes                           -8.547e-02  8.735e-02  -0.978
    ## GenderRequiredFemale                         1.539e-01  8.481e-02   1.815
    ## GenderRequiredMale                          -5.103e-02  2.775e-01  -0.184
    ## AgeAtCreation                               -2.108e-02  2.889e-03  -7.295
    ## AllRecordsNums                              -1.089e-02  7.843e-03  -1.389
    ## CoreProgramsNums                                    NA         NA      NA
    ## CoreRecordNums                               1.018e-03  8.052e-03   0.126
    ## TotalCoreProgramHours                        6.524e-04  7.589e-04   0.860
    ## MaxCoreProgramHours                         -9.742e-02  5.734e-02  -1.699
    ## MinCoreProgramHours                          2.644e-01  1.238e-01   2.136
    ## AverageCoreProgramHours                     -7.532e-04  1.101e-03  -0.684
    ## AverageCoreServiceHours                     -1.388e-01  1.047e-01  -1.326
    ## CoreRecordsRate                              3.721e-01  4.976e-01   0.748
    ## CoreTotalKM                                 -1.734e-04  2.696e-04  -0.643
    ## FirstCoreServiceDelayDays                   -6.785e-04  1.091e-03  -0.622
    ## FrequentschedStatusGroupCancelled            5.936e-01  2.972e-01   1.997
    ## FrequentschedStatusGroupKincareInitiated    -4.328e-01  5.752e-01  -0.752
    ## MostUsedBillingGradeBGrade2                  4.530e-01  1.455e-01   3.112
    ## MostUsedBillingGradeBGrade3                  2.723e-01  1.646e-01   1.655
    ## MostUsedBillingGradeBGrade4                  1.443e+01  1.021e+03   0.014
    ## MostUsedBillingGradeBGrade5                  1.784e+00  5.900e-01   3.023
    ## MostUsedBillingGradeBGrade6                  1.328e+00  3.510e-01   3.783
    ## MostUsedBillingGradeBGrade9                  8.632e-01  3.041e-01   2.839
    ## MostUsedPayGradePGrade1                     -1.838e+01  9.101e+02  -0.020
    ## MostUsedPayGradePGrade2                     -1.842e+01  9.101e+02  -0.020
    ## MostUsedPayGradePGrade3                     -1.819e+01  9.101e+02  -0.020
    ## MostUsedPayGradePGrade4                     -1.646e+01  9.101e+02  -0.018
    ## MostUsedPayGradePGrade5                     -1.834e+01  9.101e+02  -0.020
    ## MostUsedPayGradePGrade6                     -1.903e+01  9.101e+02  -0.021
    ## RespiteNeedsFlagY                           -4.540e-01  2.041e-01  -2.225
    ## DANeedsFlagY                                -2.244e-02  1.103e-01  -0.203
    ## NCNeedsFlagY                                -1.045e-01  2.319e-01  -0.451
    ## PCNeedsFlagY                                 3.093e-01  1.322e-01   2.340
    ## SocialNeedsFlagY                             2.343e-02  1.262e-01   0.186
    ## TransportNeedsFlagY                         -1.876e-01  1.877e-01  -0.999
    ## RequiredWorkersFlagY                        -3.341e-01  2.852e-01  -1.171
    ## PreferredWorkersFlagY                       -2.980e-01  1.953e-01  -1.526
    ## Client_Programs_count_at_observation_cutoff  2.681e-01  7.377e-02   3.634
    ## default_contract_groupDisability             3.307e-01  8.165e-01   0.405
    ## default_contract_groupDVA/VHC               -1.057e+01  6.885e+02  -0.015
    ## default_contract_groupPackage               -2.961e-02  1.584e-01  -0.187
    ## default_contract_groupPrivate/Commercial     4.495e-01  1.537e-01   2.924
    ## default_contract_groupTransPac               5.446e-01  6.059e-01   0.899
    ## Issues_Raised                                2.219e-01  9.848e-02   2.253
    ## Issues_Requiring_Action                      1.400e+00  1.152e+00   1.215
    ## Escalated_Issues                             1.050e+00  8.267e-01   1.270
    ## Closed_Issues                                2.423e-01  9.790e-02   2.475
    ## Client_Initiated_Cancellations              -7.081e-02  2.626e-02  -2.697
    ## Kincare_Initiated_Cancellations             -7.898e-02  6.484e-02  -1.218
    ## Canned_Appointments                         -9.199e-03  2.047e-02  -0.449
    ## HCW_Ratio                                   -9.848e-03  1.123e-02  -0.877
    ##                                             Pr(>|z|)    
    ## (Intercept)                                 0.984207    
    ## HOME_STR_stateNSW                           0.006250 ** 
    ## HOME_STR_stateQLD                           0.758467    
    ## HOME_STR_stateSA                            0.566166    
    ## HOME_STR_stateVIC                           0.011710 *  
    ## HOME_STR_stateWA                            0.253941    
    ## ClientTypeCAC                               0.230635    
    ## ClientTypeCCP                               0.256238    
    ## ClientTypeCom                               0.787626    
    ## ClientTypeDem                               0.970412    
    ## ClientTypeDis                               0.973289    
    ## ClientTypeDom                               0.335677    
    ## ClientTypeDVA                               0.990358    
    ## ClientTypeEAC                               0.189627    
    ## ClientTypeHAC                               0.389182    
    ## ClientTypeNRC                               0.304773    
    ## ClientTypeNur                               0.487607    
    ## ClientTypePer                               0.526403    
    ## ClientTypePri                               0.568274    
    ## ClientTypeRes                               0.510522    
    ## ClientTypeSoc                               0.199354    
    ## ClientTypeTAC                               0.680761    
    ## ClientTypeTCP                               0.986712    
    ## ClientTypeVHC                               0.989694    
    ## ClientTypeYou                               0.008879 ** 
    ## GradeGrade2                                 0.461153    
    ## GradeGrade3                                 0.654512    
    ## GradeGrade4                                 0.473435    
    ## GradeGrade5                                 0.665081    
    ## GradeGrade6                                 0.289743    
    ## SmokerAcceptedYes                           0.327835    
    ## GenderRequiredFemale                        0.069531 .  
    ## GenderRequiredMale                          0.854095    
    ## AgeAtCreation                               2.99e-13 ***
    ## AllRecordsNums                              0.164935    
    ## CoreProgramsNums                                  NA    
    ## CoreRecordNums                              0.899405    
    ## TotalCoreProgramHours                       0.389977    
    ## MaxCoreProgramHours                         0.089314 .  
    ## MinCoreProgramHours                         0.032670 *  
    ## AverageCoreProgramHours                     0.493879    
    ## AverageCoreServiceHours                     0.184841    
    ## CoreRecordsRate                             0.454582    
    ## CoreTotalKM                                 0.520107    
    ## FirstCoreServiceDelayDays                   0.533961    
    ## FrequentschedStatusGroupCancelled           0.045830 *  
    ## FrequentschedStatusGroupKincareInitiated    0.451801    
    ## MostUsedBillingGradeBGrade2                 0.001857 ** 
    ## MostUsedBillingGradeBGrade3                 0.098014 .  
    ## MostUsedBillingGradeBGrade4                 0.988721    
    ## MostUsedBillingGradeBGrade5                 0.002504 ** 
    ## MostUsedBillingGradeBGrade6                 0.000155 ***
    ## MostUsedBillingGradeBGrade9                 0.004528 ** 
    ## MostUsedPayGradePGrade1                     0.983889    
    ## MostUsedPayGradePGrade2                     0.983850    
    ## MostUsedPayGradePGrade3                     0.984056    
    ## MostUsedPayGradePGrade4                     0.985566    
    ## MostUsedPayGradePGrade5                     0.983920    
    ## MostUsedPayGradePGrade6                     0.983317    
    ## RespiteNeedsFlagY                           0.026091 *  
    ## DANeedsFlagY                                0.838780    
    ## NCNeedsFlagY                                0.652292    
    ## PCNeedsFlagY                                0.019269 *  
    ## SocialNeedsFlagY                            0.852791    
    ## TransportNeedsFlagY                         0.317625    
    ## RequiredWorkersFlagY                        0.241454    
    ## PreferredWorkersFlagY                       0.127120    
    ## Client_Programs_count_at_observation_cutoff 0.000279 ***
    ## default_contract_groupDisability            0.685484    
    ## default_contract_groupDVA/VHC               0.987751    
    ## default_contract_groupPackage               0.851741    
    ## default_contract_groupPrivate/Commercial    0.003451 ** 
    ## default_contract_groupTransPac              0.368707    
    ## Issues_Raised                               0.024230 *  
    ## Issues_Requiring_Action                     0.224301    
    ## Escalated_Issues                            0.204042    
    ## Closed_Issues                               0.013328 *  
    ## Client_Initiated_Cancellations              0.006999 ** 
    ## Kincare_Initiated_Cancellations             0.223207    
    ## Canned_Appointments                         0.653072    
    ## HCW_Ratio                                   0.380593    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 5134.1  on 5129  degrees of freedom
    ## Residual deviance: 4230.6  on 5050  degrees of freedom
    ## AIC: 4390.6
    ## 
    ## Number of Fisher Scoring iterations: 14

``` r
# Chosen significant variables are manually placed into variable GLM_signfcnt_features above.
# Only p-Values <- .05 are chosen (*** and **)
GLM_signficant_features <-  c(
  'HOME_STR_state',
  'ClientType',
  'AgeAtCreation',
  'MostUsedBillingGrade',
  'Client_Programs_count_at_observation_cutoff',
  'default_contract_group',
  'Client_Initiated_Cancellations'
#  'Issues_Raised'
)
print (GLM_signficant_features)
```

    ## [1] "HOME_STR_state"                             
    ## [2] "ClientType"                                 
    ## [3] "AgeAtCreation"                              
    ## [4] "MostUsedBillingGrade"                       
    ## [5] "Client_Programs_count_at_observation_cutoff"
    ## [6] "default_contract_group"                     
    ## [7] "Client_Initiated_Cancellations"

``` r
write (GLM_signficant_features, file = "GLM_signifcant_features.txt")

RFnTree_param <- 1000
ndxLabel <- which(names(d2) == 'Label')
set.seed(1)
bestmtry <-
  tuneRF(
    d2[,-ndxLabel],
    d2$Label,
    ntreeTry = RFnTree_param,
    stepFactor = 1.5,
    improve = 0.01,
    trace = TRUE,
    plot = TRUE,
    dobest = FALSE
  )
```

    ## mtry = 6  OOB error = 17.31% 
    ## Searching left ...
    ## mtry = 4     OOB error = 17.02% 
    ## 0.01689189 0.01 
    ## mtry = 3     OOB error = 17.17% 
    ## -0.009163803 0.01 
    ## Searching right ...
    ## mtry = 9     OOB error = 16.8% 
    ## 0.01260023 0.01 
    ## mtry = 13    OOB error = 16.92% 
    ## -0.006960557 0.01

![](kc_client_churn_FINAL1A_files/figure-markdown_github/cars-1.png)

``` r
RFmtry_param <- bestmtry[min(bestmtry[, 2]) == bestmtry[, 2]][1]
print (paste(
  'Feature Selection RF mtry value with least OOB error is ',
  RFmtry_param
))
```

    ## [1] "Feature Selection RF mtry value with least OOB error is  9"

``` r
set.seed(1)
mod_d2 <-
  randomForest(
    Label ~ .,
    data = d2,
    mtry = RFmtry_param,
    ntree = RFnTree_param,
    keep.forest = TRUE,
    importance = TRUE
  )
varImpPlot(
  mod_d2,
  sort = TRUE,
#  n.var = 15,
  type = 1,
  scale = FALSE
)
```

![](kc_client_churn_FINAL1A_files/figure-markdown_github/cars-2.png)

``` r
tmp <- mod_d2$importance [, 3]
tmp <- sort(tmp, decreasing = TRUE)
RF_imp_vars <- names(tmp)[1:15]
print (RF_imp_vars)
```

    ##  [1] "TotalCoreProgramHours"                      
    ##  [2] "AllRecordsNums"                             
    ##  [3] "AverageCoreProgramHours"                    
    ##  [4] "CoreRecordNums"                             
    ##  [5] "CoreTotalKM"                                
    ##  [6] "HCW_Ratio"                                  
    ##  [7] "HOME_STR_state"                             
    ##  [8] "AgeAtCreation"                              
    ##  [9] "Client_Programs_count_at_observation_cutoff"
    ## [10] "MostUsedBillingGrade"                       
    ## [11] "MaxCoreProgramHours"                        
    ## [12] "AverageCoreServiceHours"                    
    ## [13] "Issues_Raised"                              
    ## [14] "Closed_Issues"                              
    ## [15] "default_contract_group"

``` r
write(RF_imp_vars, file = "RF_imp_top15_variables.txt")

#Combine the RF and GLM features.
model_features <- union (GLM_signficant_features, RF_imp_vars)
model_features <- unique(model_features)

d2 <- d2[model_features]

#Convert categorical features (i.e factors) into integers
x <- sapply(d2, function(x)
  class(x))
factor_vars <- x[x == "factor"]
for (var in names(factor_vars)) {
  d2[[var]] <- as.integer(d2[[var]])
}

for (var in names(d2)) {
  d2[[var]] <- scale(d2[[var]], center = TRUE, scale = TRUE)
}

# Perform Correlation Analysis with abs(.8) as threshold
d2_corr <- cor(d2, method = "pearson")
print (d2_corr)
```

    ##                                             HOME_STR_state   ClientType
    ## HOME_STR_state                                  1.00000000 -0.200269152
    ## ClientType                                     -0.20026915  1.000000000
    ## AgeAtCreation                                   0.07008933 -0.074028656
    ## MostUsedBillingGrade                            0.18037771  0.055417978
    ## Client_Programs_count_at_observation_cutoff    -0.01785506 -0.080130886
    ## default_contract_group                          0.30838450 -0.273237595
    ## Client_Initiated_Cancellations                 -0.05398831  0.005577203
    ## TotalCoreProgramHours                           0.03673526 -0.128394774
    ## AllRecordsNums                                  0.07346519 -0.141876526
    ## AverageCoreProgramHours                         0.04670507 -0.154187799
    ## CoreRecordNums                                  0.08141893 -0.152571223
    ## CoreTotalKM                                     0.07392703 -0.117552453
    ## HCW_Ratio                                      -0.03547427 -0.043816554
    ## MaxCoreProgramHours                            -0.21669767  0.074894610
    ## AverageCoreServiceHours                        -0.14839521  0.043462144
    ## Issues_Raised                                  -0.08009269  0.008327817
    ## Closed_Issues                                  -0.08057211  0.023508704
    ##                                             AgeAtCreation
    ## HOME_STR_state                                0.070089330
    ## ClientType                                   -0.074028656
    ## AgeAtCreation                                 1.000000000
    ## MostUsedBillingGrade                         -0.088367924
    ## Client_Programs_count_at_observation_cutoff   0.012381330
    ## default_contract_group                        0.041147949
    ## Client_Initiated_Cancellations               -0.020852121
    ## TotalCoreProgramHours                        -0.025513056
    ## AllRecordsNums                                0.028390503
    ## AverageCoreProgramHours                      -0.046307146
    ## CoreRecordNums                                0.021010634
    ## CoreTotalKM                                   0.026330093
    ## HCW_Ratio                                     0.041347930
    ## MaxCoreProgramHours                          -0.005870537
    ## AverageCoreServiceHours                      -0.112805618
    ## Issues_Raised                                 0.046832535
    ## Closed_Issues                                 0.042850687
    ##                                             MostUsedBillingGrade
    ## HOME_STR_state                                        0.18037771
    ## ClientType                                            0.05541798
    ## AgeAtCreation                                        -0.08836792
    ## MostUsedBillingGrade                                  1.00000000
    ## Client_Programs_count_at_observation_cutoff          -0.12221781
    ## default_contract_group                                0.20372788
    ## Client_Initiated_Cancellations                       -0.10787059
    ## TotalCoreProgramHours                                -0.10272163
    ## AllRecordsNums                                       -0.07982880
    ## AverageCoreProgramHours                              -0.11480853
    ## CoreRecordNums                                       -0.07988958
    ## CoreTotalKM                                          -0.08116090
    ## HCW_Ratio                                            -0.18373325
    ## MaxCoreProgramHours                                  -0.29633684
    ## AverageCoreServiceHours                              -0.13572747
    ## Issues_Raised                                        -0.07632969
    ## Closed_Issues                                        -0.05433874
    ##                                             Client_Programs_count_at_observation_cutoff
    ## HOME_STR_state                                                              -0.01785506
    ## ClientType                                                                  -0.08013089
    ## AgeAtCreation                                                                0.01238133
    ## MostUsedBillingGrade                                                        -0.12221781
    ## Client_Programs_count_at_observation_cutoff                                  1.00000000
    ## default_contract_group                                                       0.15560648
    ## Client_Initiated_Cancellations                                               0.20705538
    ## TotalCoreProgramHours                                                        0.48769360
    ## AllRecordsNums                                                               0.53352748
    ## AverageCoreProgramHours                                                      0.43403655
    ## CoreRecordNums                                                               0.53047823
    ## CoreTotalKM                                                                  0.47782924
    ## HCW_Ratio                                                                    0.45766986
    ## MaxCoreProgramHours                                                          0.20200951
    ## AverageCoreServiceHours                                                     -0.02380233
    ## Issues_Raised                                                                0.23790936
    ## Closed_Issues                                                                0.22621570
    ##                                             default_contract_group
    ## HOME_STR_state                                          0.30838450
    ## ClientType                                             -0.27323759
    ## AgeAtCreation                                           0.04114795
    ## MostUsedBillingGrade                                    0.20372788
    ## Client_Programs_count_at_observation_cutoff             0.15560648
    ## default_contract_group                                  1.00000000
    ## Client_Initiated_Cancellations                          0.08142300
    ## TotalCoreProgramHours                                   0.22030600
    ## AllRecordsNums                                          0.30424103
    ## AverageCoreProgramHours                                 0.22144614
    ## CoreRecordNums                                          0.27429445
    ## CoreTotalKM                                             0.23851407
    ## HCW_Ratio                                               0.17389024
    ## MaxCoreProgramHours                                    -0.21534331
    ## AverageCoreServiceHours                                -0.19605905
    ## Issues_Raised                                           0.06236820
    ## Closed_Issues                                           0.05181580
    ##                                             Client_Initiated_Cancellations
    ## HOME_STR_state                                                -0.053988309
    ## ClientType                                                     0.005577203
    ## AgeAtCreation                                                 -0.020852121
    ## MostUsedBillingGrade                                          -0.107870589
    ## Client_Programs_count_at_observation_cutoff                    0.207055383
    ## default_contract_group                                         0.081422998
    ## Client_Initiated_Cancellations                                 1.000000000
    ## TotalCoreProgramHours                                          0.342271206
    ## AllRecordsNums                                                 0.411409926
    ## AverageCoreProgramHours                                        0.340624358
    ## CoreRecordNums                                                 0.401625648
    ## CoreTotalKM                                                    0.381386082
    ## HCW_Ratio                                                      0.421021555
    ## MaxCoreProgramHours                                            0.205827427
    ## AverageCoreServiceHours                                       -0.042396445
    ## Issues_Raised                                                  0.322586808
    ## Closed_Issues                                                  0.328318131
    ##                                             TotalCoreProgramHours
    ## HOME_STR_state                                         0.03673526
    ## ClientType                                            -0.12839477
    ## AgeAtCreation                                         -0.02551306
    ## MostUsedBillingGrade                                  -0.10272163
    ## Client_Programs_count_at_observation_cutoff            0.48769360
    ## default_contract_group                                 0.22030600
    ## Client_Initiated_Cancellations                         0.34227121
    ## TotalCoreProgramHours                                  1.00000000
    ## AllRecordsNums                                         0.84476671
    ## AverageCoreProgramHours                                0.83904711
    ## CoreRecordNums                                         0.85167319
    ## CoreTotalKM                                            0.72448554
    ## HCW_Ratio                                              0.65863616
    ## MaxCoreProgramHours                                    0.42082689
    ## AverageCoreServiceHours                                0.11330095
    ## Issues_Raised                                          0.25199010
    ## Closed_Issues                                          0.24213090
    ##                                             AllRecordsNums
    ## HOME_STR_state                                  0.07346519
    ## ClientType                                     -0.14187653
    ## AgeAtCreation                                   0.02839050
    ## MostUsedBillingGrade                           -0.07982880
    ## Client_Programs_count_at_observation_cutoff     0.53352748
    ## default_contract_group                          0.30424103
    ## Client_Initiated_Cancellations                  0.41140993
    ## TotalCoreProgramHours                           0.84476671
    ## AllRecordsNums                                  1.00000000
    ## AverageCoreProgramHours                         0.85081261
    ## CoreRecordNums                                  0.99230463
    ## CoreTotalKM                                     0.83394521
    ## HCW_Ratio                                       0.75038115
    ## MaxCoreProgramHours                             0.26130829
    ## AverageCoreServiceHours                        -0.12544158
    ## Issues_Raised                                   0.31844456
    ## Closed_Issues                                   0.30154321
    ##                                             AverageCoreProgramHours
    ## HOME_STR_state                                           0.04670507
    ## ClientType                                              -0.15418780
    ## AgeAtCreation                                           -0.04630715
    ## MostUsedBillingGrade                                    -0.11480853
    ## Client_Programs_count_at_observation_cutoff              0.43403655
    ## default_contract_group                                   0.22144614
    ## Client_Initiated_Cancellations                           0.34062436
    ## TotalCoreProgramHours                                    0.83904711
    ## AllRecordsNums                                           0.85081261
    ## AverageCoreProgramHours                                  1.00000000
    ## CoreRecordNums                                           0.85895084
    ## CoreTotalKM                                              0.73474396
    ## HCW_Ratio                                                0.65964953
    ## MaxCoreProgramHours                                      0.33910018
    ## AverageCoreServiceHours                                  0.11824291
    ## Issues_Raised                                            0.22182898
    ## Closed_Issues                                            0.20343252
    ##                                             CoreRecordNums CoreTotalKM
    ## HOME_STR_state                                  0.08141893  0.07392703
    ## ClientType                                     -0.15257122 -0.11755245
    ## AgeAtCreation                                   0.02101063  0.02633009
    ## MostUsedBillingGrade                           -0.07988958 -0.08116090
    ## Client_Programs_count_at_observation_cutoff     0.53047823  0.47782924
    ## default_contract_group                          0.27429445  0.23851407
    ## Client_Initiated_Cancellations                  0.40162565  0.38138608
    ## TotalCoreProgramHours                           0.85167319  0.72448554
    ## AllRecordsNums                                  0.99230463  0.83394521
    ## AverageCoreProgramHours                         0.85895084  0.73474396
    ## CoreRecordNums                                  1.00000000  0.84100842
    ## CoreTotalKM                                     0.84100842  1.00000000
    ## HCW_Ratio                                       0.75400550  0.76245895
    ## MaxCoreProgramHours                             0.26252175  0.24655819
    ## AverageCoreServiceHours                        -0.12493450 -0.12417716
    ## Issues_Raised                                   0.30691116  0.27081826
    ## Closed_Issues                                   0.29002531  0.25863269
    ##                                               HCW_Ratio
    ## HOME_STR_state                              -0.03547427
    ## ClientType                                  -0.04381655
    ## AgeAtCreation                                0.04134793
    ## MostUsedBillingGrade                        -0.18373325
    ## Client_Programs_count_at_observation_cutoff  0.45766986
    ## default_contract_group                       0.17389024
    ## Client_Initiated_Cancellations               0.42102155
    ## TotalCoreProgramHours                        0.65863616
    ## AllRecordsNums                               0.75038115
    ## AverageCoreProgramHours                      0.65964953
    ## CoreRecordNums                               0.75400550
    ## CoreTotalKM                                  0.76245895
    ## HCW_Ratio                                    1.00000000
    ## MaxCoreProgramHours                          0.28687015
    ## AverageCoreServiceHours                     -0.12536551
    ## Issues_Raised                                0.30288956
    ## Closed_Issues                                0.28470554
    ##                                             MaxCoreProgramHours
    ## HOME_STR_state                                     -0.216697675
    ## ClientType                                          0.074894610
    ## AgeAtCreation                                      -0.005870537
    ## MostUsedBillingGrade                               -0.296336838
    ## Client_Programs_count_at_observation_cutoff         0.202009515
    ## default_contract_group                             -0.215343314
    ## Client_Initiated_Cancellations                      0.205827427
    ## TotalCoreProgramHours                               0.420826886
    ## AllRecordsNums                                      0.261308287
    ## AverageCoreProgramHours                             0.339100184
    ## CoreRecordNums                                      0.262521749
    ## CoreTotalKM                                         0.246558190
    ## HCW_Ratio                                           0.286870146
    ## MaxCoreProgramHours                                 1.000000000
    ## AverageCoreServiceHours                             0.517220199
    ## Issues_Raised                                       0.174174399
    ## Closed_Issues                                       0.161378415
    ##                                             AverageCoreServiceHours
    ## HOME_STR_state                                          -0.14839521
    ## ClientType                                               0.04346214
    ## AgeAtCreation                                           -0.11280562
    ## MostUsedBillingGrade                                    -0.13572747
    ## Client_Programs_count_at_observation_cutoff             -0.02380233
    ## default_contract_group                                  -0.19605905
    ## Client_Initiated_Cancellations                          -0.04239644
    ## TotalCoreProgramHours                                    0.11330095
    ## AllRecordsNums                                          -0.12544158
    ## AverageCoreProgramHours                                  0.11824291
    ## CoreRecordNums                                          -0.12493450
    ## CoreTotalKM                                             -0.12417716
    ## HCW_Ratio                                               -0.12536551
    ## MaxCoreProgramHours                                      0.51722020
    ## AverageCoreServiceHours                                  1.00000000
    ## Issues_Raised                                           -0.07112997
    ## Closed_Issues                                           -0.06588375
    ##                                             Issues_Raised Closed_Issues
    ## HOME_STR_state                               -0.080092687   -0.08057211
    ## ClientType                                    0.008327817    0.02350870
    ## AgeAtCreation                                 0.046832535    0.04285069
    ## MostUsedBillingGrade                         -0.076329690   -0.05433874
    ## Client_Programs_count_at_observation_cutoff   0.237909361    0.22621570
    ## default_contract_group                        0.062368197    0.05181580
    ## Client_Initiated_Cancellations                0.322586808    0.32831813
    ## TotalCoreProgramHours                         0.251990105    0.24213090
    ## AllRecordsNums                                0.318444558    0.30154321
    ## AverageCoreProgramHours                       0.221828983    0.20343252
    ## CoreRecordNums                                0.306911158    0.29002531
    ## CoreTotalKM                                   0.270818263    0.25863269
    ## HCW_Ratio                                     0.302889560    0.28470554
    ## MaxCoreProgramHours                           0.174174399    0.16137842
    ## AverageCoreServiceHours                      -0.071129969   -0.06588375
    ## Issues_Raised                                 1.000000000    0.90223577
    ## Closed_Issues                                 0.902235768    1.00000000

``` r
write.csv (d2_corr, file = "kc_correlation_pearson_model_features.csv")

# By manual inspection of the correlation map, following  variables are to be removed
# due to collinearity (Pearson correlation  +/-0.816

collinear_features_to_be_removed <- c(
  'Closed_Issues',
  'AllRecordsNums',
  'CoreRecordNums',
  'TotalCoreProgramHours',
  'CoreTotalKM',
  'MaxCoreProgramHours',
  'AverageCoreServiceHours'
)
print (collinear_features_to_be_removed)
```

    ## [1] "Closed_Issues"           "AllRecordsNums"         
    ## [3] "CoreRecordNums"          "TotalCoreProgramHours"  
    ## [5] "CoreTotalKM"             "MaxCoreProgramHours"    
    ## [7] "AverageCoreServiceHours"

``` r
write(collinear_features_to_be_removed, file = "collinear_features_to_be_removed.txt")

model_features <-
  setdiff(model_features, collinear_features_to_be_removed)
print (model_features)
```

    ##  [1] "HOME_STR_state"                             
    ##  [2] "ClientType"                                 
    ##  [3] "AgeAtCreation"                              
    ##  [4] "MostUsedBillingGrade"                       
    ##  [5] "Client_Programs_count_at_observation_cutoff"
    ##  [6] "default_contract_group"                     
    ##  [7] "Client_Initiated_Cancellations"             
    ##  [8] "AverageCoreProgramHours"                    
    ##  [9] "HCW_Ratio"                                  
    ## [10] "Issues_Raised"

``` r
write(model_features, file = "selected_model_features.txt")

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
  if (class(train[[feature]]) == "factor") {
    all_levels <-
      union(levels(train[[feature]]), levels(test[[feature]]))
    levels(train[[feature]]) <- all_levels
    levels(test[[feature]]) <- all_levels
  }
}

#Prediction threshold for rounding changed from default of .5
#Threshold is set from ROC where specifty is more or less same as sensitvity
fitThreshold <- 0.6

#This weights recall twice more than precison as a measure of predction accuracy
fScoreB_param <- 2

#========================
#GLM model and performance
set.seed(1)
kc.glm <-
  glm(Label ~ ., family = binomial(link = "logit"), data = train)
kc.glm
```

    ## 
    ## Call:  glm(formula = Label ~ ., family = binomial(link = "logit"), data = train)
    ## 
    ## Coefficients:
    ##                                 (Intercept)  
    ##                                   -0.071629  
    ##                           HOME_STR_stateNSW  
    ##                                   -0.699939  
    ##                           HOME_STR_stateQLD  
    ##                                   -0.220580  
    ##                            HOME_STR_stateSA  
    ##                                   -0.118296  
    ##                           HOME_STR_stateVIC  
    ##                                   -1.333457  
    ##                            HOME_STR_stateWA  
    ##                                   -0.284973  
    ##                               ClientTypeCAC  
    ##                                    1.425988  
    ##                               ClientTypeCCP  
    ##                                    0.906875  
    ##                               ClientTypeCom  
    ##                                   -0.213161  
    ##                               ClientTypeDem  
    ##                                  -13.443714  
    ##                               ClientTypeDis  
    ##                                   -0.641594  
    ##                               ClientTypeDom  
    ##                                    1.151105  
    ##                               ClientTypeDVA  
    ##                                   -3.373480  
    ##                               ClientTypeEAC  
    ##                                    1.191865  
    ##                               ClientTypeHAC  
    ##                                    0.890359  
    ##                               ClientTypeNRC  
    ##                                  -12.019739  
    ##                               ClientTypeNur  
    ##                                   -0.747651  
    ##                               ClientTypePer  
    ##                                    0.609980  
    ##                               ClientTypePri  
    ##                                    0.839934  
    ##                               ClientTypeRes  
    ##                                   -0.047731  
    ##                               ClientTypeSoc  
    ##                                    1.848582  
    ##                               ClientTypeTAC  
    ##                                    0.838799  
    ##                               ClientTypeTCP  
    ##                                  -11.605234  
    ##                               ClientTypeVHC  
    ##                                  -12.384916  
    ##                               ClientTypeYou  
    ##                                    3.369091  
    ##                               AgeAtCreation  
    ##                                   -0.020300  
    ##                 MostUsedBillingGradeBGrade2  
    ##                                    0.467177  
    ##                 MostUsedBillingGradeBGrade3  
    ##                                    0.308766  
    ##                 MostUsedBillingGradeBGrade4  
    ##                                   16.658687  
    ##                 MostUsedBillingGradeBGrade5  
    ##                                    1.518491  
    ##                 MostUsedBillingGradeBGrade6  
    ##                                    0.879755  
    ##                 MostUsedBillingGradeBGrade9  
    ##                                    0.889901  
    ## Client_Programs_count_at_observation_cutoff  
    ##                                    0.155471  
    ##            default_contract_groupDisability  
    ##                                    0.670219  
    ##               default_contract_groupDVA/VHC  
    ##                                  -11.301715  
    ##               default_contract_groupPackage  
    ##                                    0.019674  
    ##    default_contract_groupPrivate/Commercial  
    ##                                    0.422243  
    ##              default_contract_groupTransPac  
    ##                                   -0.630018  
    ##              Client_Initiated_Cancellations  
    ##                                   -0.098284  
    ##                     AverageCoreProgramHours  
    ##                                   -0.006664  
    ##                                   HCW_Ratio  
    ##                                   -0.045153  
    ##                               Issues_Raised  
    ##                                    0.376394  
    ## 
    ## Degrees of Freedom: 4129 Total (i.e. Null);  4088 Residual
    ## Null Deviance:       4133 
    ## Residual Deviance: 3509  AIC: 3593

``` r
summary(kc.glm)
```

    ## 
    ## Call:
    ## glm(formula = Label ~ ., family = binomial(link = "logit"), data = train)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -3.6434  -0.6610  -0.4785  -0.2360   2.9994  
    ## 
    ## Coefficients:
    ##                                               Estimate Std. Error z value
    ## (Intercept)                                 -7.163e-02  1.160e+00  -0.062
    ## HOME_STR_stateNSW                           -6.999e-01  2.543e-01  -2.752
    ## HOME_STR_stateQLD                           -2.206e-01  2.819e-01  -0.783
    ## HOME_STR_stateSA                            -1.183e-01  2.764e-01  -0.428
    ## HOME_STR_stateVIC                           -1.333e+00  4.647e-01  -2.870
    ## HOME_STR_stateWA                            -2.850e-01  3.516e-01  -0.810
    ## ClientTypeCAC                                1.426e+00  1.189e+00   1.199
    ## ClientTypeCCP                                9.069e-01  1.147e+00   0.791
    ## ClientTypeCom                               -2.132e-01  1.110e+00  -0.192
    ## ClientTypeDem                               -1.344e+01  3.525e+02  -0.038
    ## ClientTypeDis                               -6.416e-01  1.410e+00  -0.455
    ## ClientTypeDom                                1.151e+00  1.138e+00   1.012
    ## ClientTypeDVA                               -3.373e+00  1.726e+03  -0.002
    ## ClientTypeEAC                                1.192e+00  1.132e+00   1.053
    ## ClientTypeHAC                                8.904e-01  1.098e+00   0.811
    ## ClientTypeNRC                               -1.202e+01  4.045e+02  -0.030
    ## ClientTypeNur                               -7.477e-01  1.627e+00  -0.460
    ## ClientTypePer                                6.100e-01  1.204e+00   0.507
    ## ClientTypePri                                8.399e-01  1.130e+00   0.744
    ## ClientTypeRes                               -4.773e-02  1.576e+00  -0.030
    ## ClientTypeSoc                                1.849e+00  1.276e+00   1.449
    ## ClientTypeTAC                                8.388e-01  1.555e+00   0.539
    ## ClientTypeTCP                               -1.161e+01  9.566e+02  -0.012
    ## ClientTypeVHC                               -1.238e+01  9.284e+02  -0.013
    ## ClientTypeYou                                3.369e+00  1.455e+00   2.316
    ## AgeAtCreation                               -2.030e-02  3.084e-03  -6.582
    ## MostUsedBillingGradeBGrade2                  4.672e-01  1.224e-01   3.816
    ## MostUsedBillingGradeBGrade3                  3.088e-01  1.522e-01   2.029
    ## MostUsedBillingGradeBGrade4                  1.666e+01  1.027e+03   0.016
    ## MostUsedBillingGradeBGrade5                  1.518e+00  3.241e-01   4.686
    ## MostUsedBillingGradeBGrade6                  8.798e-01  1.693e-01   5.195
    ## MostUsedBillingGradeBGrade9                  8.899e-01  2.991e-01   2.975
    ## Client_Programs_count_at_observation_cutoff  1.555e-01  7.352e-02   2.115
    ## default_contract_groupDisability             6.702e-01  8.363e-01   0.801
    ## default_contract_groupDVA/VHC               -1.130e+01  9.284e+02  -0.012
    ## default_contract_groupPackage                1.967e-02  1.675e-01   0.117
    ## default_contract_groupPrivate/Commercial     4.222e-01  1.486e-01   2.841
    ## default_contract_groupTransPac              -6.300e-01  3.671e-01  -1.716
    ## Client_Initiated_Cancellations              -9.828e-02  2.340e-02  -4.199
    ## AverageCoreProgramHours                     -6.664e-03  9.795e-04  -6.804
    ## HCW_Ratio                                   -4.515e-02  1.075e-02  -4.199
    ## Issues_Raised                                3.764e-01  5.649e-02   6.663
    ##                                             Pr(>|z|)    
    ## (Intercept)                                 0.950755    
    ## HOME_STR_stateNSW                           0.005917 ** 
    ## HOME_STR_stateQLD                           0.433900    
    ## HOME_STR_stateSA                            0.668643    
    ## HOME_STR_stateVIC                           0.004108 ** 
    ## HOME_STR_stateWA                            0.417697    
    ## ClientTypeCAC                               0.230387    
    ## ClientTypeCCP                               0.429130    
    ## ClientTypeCom                               0.847649    
    ## ClientTypeDem                               0.969573    
    ## ClientTypeDis                               0.649171    
    ## ClientTypeDom                               0.311639    
    ## ClientTypeDVA                               0.998441    
    ## ClientTypeEAC                               0.292336    
    ## ClientTypeHAC                               0.417581    
    ## ClientTypeNRC                               0.976295    
    ## ClientTypeNur                               0.645807    
    ## ClientTypePer                               0.612440    
    ## ClientTypePri                               0.457144    
    ## ClientTypeRes                               0.975834    
    ## ClientTypeSoc                               0.147290    
    ## ClientTypeTAC                               0.589607    
    ## ClientTypeTCP                               0.990321    
    ## ClientTypeVHC                               0.989357    
    ## ClientTypeYou                               0.020559 *  
    ## AgeAtCreation                               4.63e-11 ***
    ## MostUsedBillingGradeBGrade2                 0.000136 ***
    ## MostUsedBillingGradeBGrade3                 0.042429 *  
    ## MostUsedBillingGradeBGrade4                 0.987058    
    ## MostUsedBillingGradeBGrade5                 2.79e-06 ***
    ## MostUsedBillingGradeBGrade6                 2.04e-07 ***
    ## MostUsedBillingGradeBGrade9                 0.002930 ** 
    ## Client_Programs_count_at_observation_cutoff 0.034466 *  
    ## default_contract_groupDisability            0.422879    
    ## default_contract_groupDVA/VHC               0.990287    
    ## default_contract_groupPackage               0.906475    
    ## default_contract_groupPrivate/Commercial    0.004497 ** 
    ## default_contract_groupTransPac              0.086128 .  
    ## Client_Initiated_Cancellations              2.68e-05 ***
    ## AverageCoreProgramHours                     1.02e-11 ***
    ## HCW_Ratio                                   2.68e-05 ***
    ## Issues_Raised                               2.69e-11 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 4133.3  on 4129  degrees of freedom
    ## Residual deviance: 3509.4  on 4088  degrees of freedom
    ## AIC: 3593.4
    ## 
    ## Number of Fisher Scoring iterations: 14

``` r
pr.kc.glm <- predict(kc.glm, newdata = test, type = 'response')
fitted.results.glm <- ifelse(pr.kc.glm > fitThreshold, 1, 0)
confusion_maxtix <- table(test$Label, fitted.results.glm)
print.table(confusion_maxtix)
```

    ##    fitted.results.glm
    ##       0   1
    ##   0 796   4
    ##   1 174  26

``` r
print(paste(
  'GLM Accuracy',
  computePerformanceMeasures(confusion_maxtix, fScoreB_param)[1]
))
```

    ## [1] "GLM Accuracy 0.822"

``` r
print(paste(
  'GLM Precision',
  computePerformanceMeasures(confusion_maxtix, fScoreB_param)[2]
))
```

    ## [1] "GLM Precision 0.866666666666667"

``` r
print(paste(
  'GLM Recall',
  computePerformanceMeasures(confusion_maxtix, fScoreB_param)[3]
))
```

    ## [1] "GLM Recall 0.13"

``` r
print(paste(
  'GLM F2 Score',
  computePerformanceMeasures(confusion_maxtix, fScoreB_param)[4]
))
```

    ## [1] "GLM F2 Score 1.3855421686747"

``` r
#==========================
# Random Forest model and performance

RFnTree_param <- 1000
ndxLabel <- which(names(train) == "Label")
set.seed(1)
bestmtry <-
  tuneRF(
    train[,-ndxLabel],
    train$Label,
    ntreeTry = RFnTree_param,
    stepFactor = 1.5,
    improve = 0.01,
    trace = TRUE,
    plot = TRUE,
    dobest = FALSE
  )
```

    ## mtry = 3  OOB error = 17.97% 
    ## Searching left ...
    ## mtry = 2     OOB error = 17.94% 
    ## 0.001347709 0.01 
    ## Searching right ...
    ## mtry = 4     OOB error = 18.18% 
    ## -0.01212938 0.01

![](kc_client_churn_FINAL1A_files/figure-markdown_github/cars-3.png)

``` r
RFmtry_param <- bestmtry[min(bestmtry[, 2]) == bestmtry[, 2]][1]
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
    ##         OOB estimate of  error rate: 17.87%
    ## Confusion matrix:
    ##      0   1 class.error
    ## 0 3153 151  0.04570218
    ## 1  587 239  0.71065375

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
    ## importance        40   -none- numeric  
    ## importanceSD      30   -none- numeric  
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
varImpPlot(kc.rf, type = 1)
```

![](kc_client_churn_FINAL1A_files/figure-markdown_github/cars-4.png)

``` r
# make predictions
pr.kc.rf <- predict(kc.rf, newdata = test, type = 'prob')[, 2]
fitted.results.rf <- ifelse(pr.kc.rf > fitThreshold, 1, 0)
confusion_maxtix <- table(test$Label, fitted.results.rf)
print.table(confusion_maxtix)
```

    ##    fitted.results.rf
    ##       0   1
    ##   0 798   2
    ##   1 165  35

``` r
print(paste(
  'RF Accuracy',
  computePerformanceMeasures(confusion_maxtix, fScoreB_param)[1]
))
```

    ## [1] "RF Accuracy 0.833"

``` r
print(paste(
  'RF Precision',
  computePerformanceMeasures(confusion_maxtix, fScoreB_param)[2]
))
```

    ## [1] "RF Precision 0.945945945945946"

``` r
print(paste(
  'RF Recall',
  computePerformanceMeasures(confusion_maxtix, fScoreB_param)[3]
))
```

    ## [1] "RF Recall 0.175"

``` r
print(paste(
  'RF F2 Score',
  computePerformanceMeasures(confusion_maxtix, fScoreB_param)[4]
))
```

    ## [1] "RF F2 Score 1.415770609319"

``` r
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
    ## C5.0 [Release 2.07 GPL Edition]      Sun Apr  9 18:15:05 2017
    ## -------------------------------
    ## 
    ## Class specified by attribute `outcome'
    ## 
    ## Read 4130 cases (11 attributes) from undefined.data
    ## 
    ## -----  Trial 0:  -----
    ## 
    ## Decision tree:
    ## 
    ## MostUsedBillingGrade in {BGrade4,BGrade5,BGrade6,BGrade9}:
    ## :...HOME_STR_state in {ACT,NSW,VIC,WA}:
    ## :   :...Issues_Raised <= 0.5258737: 0 (86/10)
    ## :   :   Issues_Raised > 0.5258737:
    ## :   :   :...AverageCoreProgramHours <= 5.5: 1 (9/2)
    ## :   :       AverageCoreProgramHours > 5.5: 0 (34/11)
    ## :   HOME_STR_state in {QLD,SA}:
    ## :   :...default_contract_group in {Disability,Package}: 0 (3/1)
    ## :       default_contract_group in {DVA/VHC,Private/Commercial,
    ## :       :                          TransPac}: 1 (157/42)
    ## :       default_contract_group = CHSP:
    ## :       :...ClientType in {Bro,CAC,CCP,Dem,Dis,Dom,DVA,EAC,NRC,Nur,Per,Res,Soc,
    ## :           :              TAC,TCP,VHC,You}: 1 (0)
    ## :           ClientType in {Com,Pri}: 0 (3)
    ## :           ClientType = HAC:
    ## :           :...HCW_Ratio > 0: 0 (10/3)
    ## :               HCW_Ratio <= 0:
    ## :               :...HOME_STR_state = SA: 1 (21/3)
    ## :                   HOME_STR_state = QLD:
    ## :                   :...AverageCoreProgramHours <= 2.5: 0 (5)
    ## :                       AverageCoreProgramHours > 2.5: 1 (29/7)
    ## MostUsedBillingGrade in {BGrade1,BGrade2,BGrade3}:
    ## :...AverageCoreProgramHours > 17.875: 0 (2686/297)
    ##     AverageCoreProgramHours <= 17.875:
    ##     :...Client_Initiated_Cancellations > 5.047571: 0 (27)
    ##         Client_Initiated_Cancellations <= 5.047571:
    ##         :...AgeAtCreation > 84: 0 (332/62)
    ##             AgeAtCreation <= 84:
    ##             :...ClientType in {Bro,CAC,CCP,Com,Dem,Dis,DVA,NRC,Nur,Per,Pri,Res,
    ##                 :              Soc,TCP,VHC,You}: 0 (49/18)
    ##                 ClientType in {EAC,TAC}: 1 (14/3)
    ##                 ClientType = Dom:
    ##                 :...Client_Programs_count_at_observation_cutoff > 0.4849309: 1 (8/1)
    ##                 :   Client_Programs_count_at_observation_cutoff <= 0.4849309:
    ##                 :   :...Client_Initiated_Cancellations > 1.497191: 1 (5)
    ##                 :       Client_Initiated_Cancellations <= 1.497191:
    ##                 :       :...AgeAtCreation <= 73: 1 (6/1)
    ##                 :           AgeAtCreation > 73: 0 (12/1)
    ##                 ClientType = HAC:
    ##                 :...Issues_Raised > 0.554844: [S1]
    ##                     Issues_Raised <= 0.554844:
    ##                     :...AverageCoreProgramHours > 11.25: 0 (96/11)
    ##                         AverageCoreProgramHours <= 11.25:
    ##                         :...default_contract_group = Disability: 1 (1)
    ##                             default_contract_group in {DVA/VHC,Package,
    ##                             :                          TransPac}: 0 (25/4)
    ##                             default_contract_group = Private/Commercial:
    ##                             :...AgeAtCreation <= 24: 0 (8/1)
    ##                             :   AgeAtCreation > 24:
    ##                             :   :...HOME_STR_state in {ACT,NSW,QLD,SA,
    ##                             :       :                  VIC}: 1 (49/18)
    ##                             :       HOME_STR_state = WA: 0 (6)
    ##                             default_contract_group = CHSP:
    ##                             :...MostUsedBillingGrade = BGrade2: 1 (26/10)
    ##                                 MostUsedBillingGrade in {BGrade1,BGrade3}:
    ##                                 :...HOME_STR_state in {NSW,SA,VIC,
    ##                                     :                  WA}: 0 (183/50)
    ##                                     HOME_STR_state in {ACT,QLD}: [S2]
    ## 
    ## SubTree [S1]
    ## 
    ## default_contract_group in {Disability,DVA/VHC,Private/Commercial,
    ## :                          TransPac}: 0 (18/2)
    ## default_contract_group in {CHSP,Package}:
    ## :...HOME_STR_state = VIC: 1 (0)
    ##     HOME_STR_state in {ACT,QLD,SA,WA}: 0 (38/16)
    ##     HOME_STR_state = NSW:
    ##     :...Issues_Raised <= 1.496931: 1 (102/43)
    ##         Issues_Raised > 1.496931:
    ##         :...AverageCoreProgramHours <= 6.833333: 1 (6/1)
    ##             AverageCoreProgramHours > 6.833333: 0 (24/9)
    ## 
    ## SubTree [S2]
    ## 
    ## MostUsedBillingGrade = BGrade3: 0 (8/2)
    ## MostUsedBillingGrade = BGrade1:
    ## :...AverageCoreProgramHours <= 9.666667: 1 (38/13)
    ##     AverageCoreProgramHours > 9.666667: 0 (6/1)
    ## 
    ## -----  Trial 1:  -----
    ## 
    ## Decision tree:
    ## 
    ## AverageCoreProgramHours > 37.5:
    ## :...ClientType in {Bro,CAC,CCP,Com,Dem,Dis,Dom,DVA,EAC,HAC,NRC,Nur,Per,Pri,Res,
    ## :   :              TAC,TCP,VHC}: 0 (1814.6/358)
    ## :   ClientType in {Soc,You}: 1 (22.9/4)
    ## AverageCoreProgramHours <= 37.5:
    ## :...AverageCoreProgramHours <= 6.833333:
    ##     :...default_contract_group in {Disability,DVA/VHC,TransPac}: 0 (7.2)
    ##     :   default_contract_group in {CHSP,Package,Private/Commercial}:
    ##     :   :...HOME_STR_state in {ACT,SA}: 1 (389.5/181.1)
    ##     :       HOME_STR_state in {QLD,VIC,WA}: 0 (124.5/56.9)
    ##     :       HOME_STR_state = NSW:
    ##     :       :...AgeAtCreation <= 60: 0 (32.5/5.3)
    ##     :           AgeAtCreation > 60:
    ##     :           :...AgeAtCreation <= 73: 1 (83.5/29.4)
    ##     :               AgeAtCreation > 73: 0 (248.3/100.2)
    ##     AverageCoreProgramHours > 6.833333:
    ##     :...Issues_Raised <= 0.3290106:
    ##         :...Client_Programs_count_at_observation_cutoff <= 0.7405742: 0 (572.8/137.3)
    ##         :   Client_Programs_count_at_observation_cutoff > 0.7405742:
    ##         :   :...HCW_Ratio <= 4: 1 (93.8/40)
    ##         :       HCW_Ratio > 4: 0 (86.9/25.6)
    ##         Issues_Raised > 0.3290106:
    ##         :...default_contract_group in {Disability,DVA/VHC,
    ##             :                          TransPac}: 0 (22.8/2.1)
    ##             default_contract_group in {CHSP,Package,Private/Commercial}:
    ##             :...Client_Initiated_Cancellations > 3.149487: 0 (95.6/26.3)
    ##                 Client_Initiated_Cancellations <= 3.149487:
    ##                 :...MostUsedBillingGrade in {BGrade2,BGrade6}: 1 (128.6/52.8)
    ##                     MostUsedBillingGrade in {BGrade3,BGrade4,BGrade5,
    ##                     :                        BGrade9}: 0 (10.9/2.9)
    ##                     MostUsedBillingGrade = BGrade1: [S1]
    ## 
    ## SubTree [S1]
    ## 
    ## default_contract_group = Private/Commercial: 0 (20.9/4.2)
    ## default_contract_group in {CHSP,Package}:
    ## :...Client_Programs_count_at_observation_cutoff <= 0.974571: 0 (222.4/97.3)
    ##     Client_Programs_count_at_observation_cutoff > 0.974571: 1 (152.4/68.4)
    ## 
    ## -----  Trial 2:  -----
    ## 
    ## Decision tree:
    ## 
    ## AverageCoreProgramHours > 46:
    ## :...Issues_Raised <= 0.6818216: 0 (677.5/101)
    ## :   Issues_Raised > 0.6818216:
    ## :   :...AgeAtCreation > 78: 0 (405.1/91.3)
    ## :       AgeAtCreation <= 78:
    ## :       :...Issues_Raised > 1.705531: 0 (172.8/59.7)
    ## :           Issues_Raised <= 1.705531:
    ## :           :...Client_Initiated_Cancellations <= 1.181425: 1 (121.8/36.6)
    ## :               Client_Initiated_Cancellations > 1.181425: 0 (126.7/50.1)
    ## AverageCoreProgramHours <= 46:
    ## :...Issues_Raised <= 0.5329809:
    ##     :...HCW_Ratio > 2: 0 (638.5/187.9)
    ##     :   HCW_Ratio <= 2:
    ##     :   :...AgeAtCreation > 83: 0 (260.5/88.7)
    ##     :       AgeAtCreation <= 83:
    ##     :       :...HOME_STR_state in {ACT,SA,VIC,WA}: 0 (342.7/145.1)
    ##     :           HOME_STR_state = QLD: 1 (81.3/33.3)
    ##     :           HOME_STR_state = NSW:
    ##     :           :...Client_Initiated_Cancellations <= 0.5011531: 0 (186.3/83.1)
    ##     :               Client_Initiated_Cancellations > 0.5011531: 1 (66.8/25.3)
    ##     Issues_Raised > 0.5329809:
    ##     :...default_contract_group in {Disability,DVA/VHC,
    ##         :                          TransPac}: 0 (26.4/2.6)
    ##         default_contract_group in {CHSP,Package,Private/Commercial}:
    ##         :...AverageCoreProgramHours <= 4.583333: 1 (142.7/52.2)
    ##             AverageCoreProgramHours > 4.583333:
    ##             :...MostUsedBillingGrade in {BGrade2,BGrade4,BGrade5,
    ##                 :                        BGrade9}: 0 (194.7/78.2)
    ##                 MostUsedBillingGrade in {BGrade3,BGrade6}: 1 (50.9/22.8)
    ##                 MostUsedBillingGrade = BGrade1:
    ##                 :...default_contract_group in {Package,
    ##                     :                          Private/Commercial}: 0 (86.6/36.1)
    ##                     default_contract_group = CHSP:
    ##                     :...HCW_Ratio > 11: 0 (42.2/11.1)
    ##                         HCW_Ratio <= 11:
    ##                         :...Issues_Raised > 1.496931: 1 (145.3/66.5)
    ##                             Issues_Raised <= 1.496931:
    ##                             :...Client_Initiated_Cancellations <= 2.428169: 1 (292.5/134)
    ##                                 Client_Initiated_Cancellations > 2.428169: 0 (68.6/19)
    ## 
    ## -----  Trial 3:  -----
    ## 
    ## Decision tree:
    ## 
    ## AverageCoreProgramHours <= 13.75:
    ## :...HOME_STR_state in {ACT,WA}: 1 (100.9/47)
    ## :   HOME_STR_state = VIC: 0 (19.3/6.9)
    ## :   HOME_STR_state = QLD:
    ## :   :...AverageCoreProgramHours <= 2.25: 0 (11.1/1.1)
    ## :   :   AverageCoreProgramHours > 2.25: 1 (182.3/74.7)
    ## :   HOME_STR_state = SA:
    ## :   :...default_contract_group in {Disability,DVA/VHC,Private/Commercial,
    ## :   :   :                          TransPac}: 1 (295.7/128.9)
    ## :   :   default_contract_group = Package: 0 (2.2)
    ## :   :   default_contract_group = CHSP:
    ## :   :   :...Client_Programs_count_at_observation_cutoff <= 0.9451866: 0 (208.9/92.9)
    ## :   :       Client_Programs_count_at_observation_cutoff > 0.9451866: 1 (27.8/5.8)
    ## :   HOME_STR_state = NSW:
    ## :   :...MostUsedBillingGrade in {BGrade2,BGrade5,BGrade6,
    ## :       :                        BGrade9}: 0 (183.5/78.7)
    ## :       MostUsedBillingGrade in {BGrade3,BGrade4}: 1 (182.6/80.2)
    ## :       MostUsedBillingGrade = BGrade1:
    ## :       :...Issues_Raised > 1.496931: 1 (42.5/17.9)
    ## :           Issues_Raised <= 1.496931:
    ## :           :...AgeAtCreation <= 68: 1 (47.3/17.7)
    ## :               AgeAtCreation > 68: 0 (225.5/70.7)
    ## AverageCoreProgramHours > 13.75:
    ## :...Issues_Raised <= 0.960986: 0 (1162.4/307.3)
    ##     Issues_Raised > 0.960986:
    ##     :...AgeAtCreation > 77: 0 (778.8/257.1)
    ##         AgeAtCreation <= 77:
    ##         :...Issues_Raised > 1.010425: 0 (243.5/89.3)
    ##             Issues_Raised <= 1.010425:
    ##             :...AgeAtCreation <= 45: 1 (52.3/12.5)
    ##                 AgeAtCreation > 45:
    ##                 :...Client_Initiated_Cancellations <= 4.402288: 1 (320.8/154.6)
    ##                     Client_Initiated_Cancellations > 4.402288: 0 (42.7/13.9)
    ## 
    ## -----  Trial 4:  -----
    ## 
    ## Decision tree:
    ## 
    ## AgeAtCreation > 91: 0 (151.9/32)
    ## AgeAtCreation <= 91:
    ## :...AverageCoreProgramHours <= 46:
    ##     :...MostUsedBillingGrade in {BGrade1,BGrade3,BGrade6,
    ##     :   :                        BGrade9}: 0 (2110.1/908.6)
    ##     :   MostUsedBillingGrade in {BGrade2,BGrade4,BGrade5}: 1 (539.9/240.7)
    ##     AverageCoreProgramHours > 46:
    ##     :...ClientType in {Bro,Dem,DVA,NRC,Nur,Res,TAC,TCP,VHC}: 0 (28)
    ##         ClientType in {Soc,You}: 1 (22.8/4.1)
    ##         ClientType in {CAC,CCP,Com,Dis,Dom,EAC,HAC,Per,Pri}:
    ##         :...Issues_Raised <= 0.6818216: 0 (534.4/124.8)
    ##             Issues_Raised > 0.6818216:
    ##             :...AverageCoreProgramHours > 176.5: 0 (248.9/68.7)
    ##                 AverageCoreProgramHours <= 176.5:
    ##                 :...HOME_STR_state in {ACT,QLD,WA}: 1 (127.2/48.3)
    ##                     HOME_STR_state in {SA,VIC}: 0 (47.8/15.5)
    ##                     HOME_STR_state = NSW:
    ##                     :...default_contract_group = Disability: 1 (2)
    ##                         default_contract_group in {DVA/VHC,
    ##                         :                          TransPac}: 0 (10.3)
    ##                         default_contract_group in {CHSP,Package,
    ##                         :                          Private/Commercial}:
    ##                         :...Client_Initiated_Cancellations > 4.787849: 0 (56.4/10.1)
    ##                             Client_Initiated_Cancellations <= 4.787849:
    ##                             :...AgeAtCreation <= 79: 1 (170/73.7)
    ##                                 AgeAtCreation > 79: 0 (80.2/27)
    ## 
    ## -----  Trial 5:  -----
    ## 
    ## Decision tree:
    ## 
    ## AverageCoreProgramHours <= 46:
    ## :...ClientType in {Bro,CCP,Com,Dom,EAC,Nur,Pri,Res}: 1 (432.5/204.3)
    ## :   ClientType in {CAC,Dem,Dis,DVA,NRC,Per,Soc,TAC,TCP,VHC,
    ## :   :              You}: 0 (68.3/22.8)
    ## :   ClientType = HAC:
    ## :   :...AgeAtCreation > 92: 0 (36.8/7)
    ## :       AgeAtCreation <= 92:
    ## :       :...default_contract_group = Disability: 1 (1.8)
    ## :           default_contract_group in {DVA/VHC,TransPac}: 0 (32.1/8.7)
    ## :           default_contract_group = Private/Commercial:
    ## :           :...HOME_STR_state = WA: 0 (7)
    ## :           :   HOME_STR_state in {ACT,NSW,QLD,SA,VIC}:
    ## :           :   :...HCW_Ratio > 5: 1 (25.4/5.1)
    ## :           :       HCW_Ratio <= 5:
    ## :           :       :...Issues_Raised <= 0.4970248: 1 (281.1/123.8)
    ## :           :           Issues_Raised > 0.4970248: 0 (33.2/10.5)
    ## :           default_contract_group in {CHSP,Package}:
    ## :           :...HOME_STR_state in {ACT,SA}: 1 (364.3/165.1)
    ## :               HOME_STR_state in {VIC,WA}: 0 (17/5.8)
    ## :               HOME_STR_state = QLD:
    ## :               :...default_contract_group = Package: 1 (24.9/5.2)
    ## :               :   default_contract_group = CHSP:
    ## :               :   :...HCW_Ratio <= 5: 1 (264/126.6)
    ## :               :       HCW_Ratio > 5: 0 (49.6/12.4)
    ## :               HOME_STR_state = NSW:
    ## :               :...Issues_Raised <= 0.554844: 0 (519.5/196.7)
    ## :                   Issues_Raised > 0.554844:
    ## :                   :...MostUsedBillingGrade in {BGrade2,BGrade3,
    ## :                       :                        BGrade5}: 0 (139.6/62.5)
    ## :                       MostUsedBillingGrade in {BGrade4,BGrade6,
    ## :                       :                        BGrade9}: 1 (27.1/11.8)
    ## :                       MostUsedBillingGrade = BGrade1:
    ## :                       :...default_contract_group = Package: 1 (42.6/15.5)
    ## :                           default_contract_group = CHSP:
    ## :                           :...AverageCoreProgramHours <= 6.833333: 1 (32.5/8.3)
    ## :                               AverageCoreProgramHours > 6.833333: 0 (391.2/180.7)
    ## AverageCoreProgramHours > 46:
    ## :...HOME_STR_state in {ACT,WA}:
    ##     :...ClientType in {Bro,CCP,Com,Dem,Dis,DVA,NRC,Nur,Res,TAC,TCP,
    ##     :   :              VHC}: 0 (14.9)
    ##     :   ClientType in {CAC,Dom,EAC,HAC,Per,Pri,Soc,You}:
    ##     :   :...AgeAtCreation <= 63: 1 (48.6/11.9)
    ##     :       AgeAtCreation > 63:
    ##     :       :...ClientType in {Dom,You}: 0 (11)
    ##     :           ClientType in {CAC,EAC,HAC,Per,Pri,Soc}:
    ##     :           :...AverageCoreProgramHours <= 248.5: 1 (110.1/48.3)
    ##     :               AverageCoreProgramHours > 248.5: 0 (51.5/12.4)
    ##     HOME_STR_state in {NSW,QLD,SA,VIC}:
    ##     :...AgeAtCreation > 77: 0 (516.8/116.7)
    ##         AgeAtCreation <= 77:
    ##         :...AgeAtCreation <= 32: 1 (32.6/11.2)
    ##             AgeAtCreation > 32:
    ##             :...ClientType in {CAC,CCP}: 1 (36.6/12.5)
    ##                 ClientType in {Bro,Dem,Dis,Dom,DVA,NRC,Nur,Per,Pri,Res,Soc,TAC,
    ##                 :              TCP,VHC,You}: 0 (25.8)
    ##                 ClientType in {Com,EAC,HAC}:
    ##                 :...MostUsedBillingGrade in {BGrade2,BGrade3,BGrade4,BGrade5,
    ##                     :                        BGrade6,BGrade9}: 0 (141.9/56.3)
    ##                     MostUsedBillingGrade = BGrade1:
    ##                     :...default_contract_group = DVA/VHC: 0 (0)
    ##                         default_contract_group in {Disability,
    ##                         :                          TransPac}: 1 (12.5/2.2)
    ##                         default_contract_group in {CHSP,Package,
    ##                         :                          Private/Commercial}:
    ##                         :...ClientType in {Com,EAC}: 0 (20/3.6)
    ##                             ClientType = HAC:
    ##                             :...HOME_STR_state in {NSW,QLD,VIC}: 0 (287.9/82.3)
    ##                                 HOME_STR_state = SA: 1 (29.6/9.2)
    ## 
    ## -----  Trial 6:  -----
    ## 
    ## Decision tree:
    ## 
    ## AverageCoreProgramHours > 46: 0 (1182.3/306.3)
    ## AverageCoreProgramHours <= 46:
    ## :...AgeAtCreation > 83: 0 (884.7/325.4)
    ##     AgeAtCreation <= 83:
    ##     :...HCW_Ratio > 14: 0 (29.8/5.4)
    ##         HCW_Ratio <= 14:
    ##         :...ClientType in {Bro,CAC,Com,Dom,EAC,Per,Res,TAC}: 1 (259.8/124.1)
    ##             ClientType in {CCP,Dem,Dis,DVA,NRC,Nur,Pri,Soc,TCP,VHC,
    ##             :              You}: 0 (117.5/49.7)
    ##             ClientType = HAC:
    ##             :...MostUsedBillingGrade in {BGrade3,BGrade4,BGrade5,BGrade6,
    ##                 :                        BGrade9}: 1 (418.1/194.6)
    ##                 MostUsedBillingGrade = BGrade2:
    ##                 :...Client_Initiated_Cancellations > 4.555237: 0 (14)
    ##                 :   Client_Initiated_Cancellations <= 4.555237: [S1]
    ##                 MostUsedBillingGrade = BGrade1:
    ##                 :...Issues_Raised <= 0.7355129: 0 (488.2/185.5)
    ##                     Issues_Raised > 0.7355129:
    ##                     :...AgeAtCreation <= 61: 1 (26.1/5.1)
    ##                         AgeAtCreation > 61:
    ##                         :...HOME_STR_state in {ACT,SA,VIC}: 0 (27.9/6.7)
    ##                             HOME_STR_state = WA: 1 (2.7)
    ##                             HOME_STR_state in {NSW,QLD}:
    ##                             :...HOME_STR_state = QLD: 0 (85.9/39.2)
    ##                                 HOME_STR_state = NSW: [S2]
    ## 
    ## SubTree [S1]
    ## 
    ## Client_Programs_count_at_observation_cutoff <= 0.7902139: 0 (186.2/83.1)
    ## Client_Programs_count_at_observation_cutoff > 0.7902139: 1 (32.7/4.4)
    ## 
    ## SubTree [S2]
    ## 
    ## Client_Initiated_Cancellations <= 3.399661: 1 (284.1/126.5)
    ## Client_Initiated_Cancellations > 3.399661: 0 (30/8.2)
    ## 
    ## -----  Trial 7:  -----
    ## 
    ## Decision tree:
    ## 
    ## AverageCoreProgramHours <= 18:
    ## :...ClientType in {Bro,CAC,CCP,Com,Dem,Dis,Dom,DVA,NRC,Per,Res,Soc,TCP,VHC,
    ## :   :              You}: 0 (227.6/88.2)
    ## :   ClientType in {EAC,Nur,Pri,TAC}: 1 (147.3/65.7)
    ## :   ClientType = HAC:
    ## :   :...default_contract_group = Disability: 1 (1.5)
    ## :       default_contract_group in {DVA/VHC,Package,TransPac}: 0 (87.4/20.8)
    ## :       default_contract_group in {CHSP,Private/Commercial}:
    ## :       :...HOME_STR_state = WA: 0 (7.5)
    ## :           HOME_STR_state in {ACT,NSW,QLD,SA,VIC}:
    ## :           :...HCW_Ratio > 6: 0 (31.6/7.3)
    ## :               HCW_Ratio <= 6:
    ## :               :...MostUsedBillingGrade in {BGrade2,BGrade4,
    ## :                   :                        BGrade5}: 1 (298.1/135.4)
    ## :                   MostUsedBillingGrade in {BGrade3,BGrade9}: 0 (331.3/142.6)
    ## :                   MostUsedBillingGrade = BGrade6: [S1]
    ## :                   MostUsedBillingGrade = BGrade1:
    ## :                   :...AgeAtCreation <= 57: 1 (38.8/11.1)
    ## :                       AgeAtCreation > 57:
    ## :                       :...AverageCoreProgramHours <= 6.833333: 1 (245.7/108.7)
    ## :                           AverageCoreProgramHours > 6.833333:
    ## :                           :...AgeAtCreation > 87: 0 (23.7)
    ## :                               AgeAtCreation <= 87:
    ## :                               :...AverageCoreProgramHours <= 7.416667: 0 (19.3)
    ## :                                   AverageCoreProgramHours > 7.416667:
    ## :                                   :...Issues_Raised <= 0.576956: 0 (156.4/46.1)
    ## :                                       Issues_Raised > 0.576956:
    ## :                                       :...HCW_Ratio <= 3: 0 (102/43.3)
    ## :                                           HCW_Ratio > 3: 1 (86.4/29.6)
    ## AverageCoreProgramHours > 18:
    ## :...Issues_Raised <= 0.5498743: 0 (717.8/71.5)
    ##     Issues_Raised > 0.5498743:
    ##     :...AverageCoreProgramHours > 187.5: 0 (189.4/34.9)
    ##         AverageCoreProgramHours <= 187.5:
    ##         :...AgeAtCreation > 90: 0 (42.9/3.4)
    ##             AgeAtCreation <= 90:
    ##             :...ClientType in {Bro,Dem,Dom,DVA,NRC,Nur,Per,Pri,Res,TAC,TCP,VHC,
    ##                 :              You}: 0 (64.7/4.1)
    ##                 ClientType in {CAC,CCP,Com,Dis,EAC,HAC,Soc}:
    ##                 :...HOME_STR_state in {ACT,VIC,WA}: 1 (133.7/54.7)
    ##                     HOME_STR_state in {NSW,QLD,SA}:
    ##                     :...Issues_Raised > 1.010425: 0 (242.8/59.7)
    ##                         Issues_Raised <= 1.010425:
    ##                         :...AgeAtCreation > 77: 0 (212.2/65.2)
    ##                             AgeAtCreation <= 77:
    ##                             :...AgeAtCreation <= 45: 1 (27.5/6)
    ##                                 AgeAtCreation > 45:
    ##                                 :...MostUsedBillingGrade in {BGrade4,BGrade5,
    ##                                     :                        BGrade9}: 0 (0)
    ##                                     MostUsedBillingGrade in {BGrade3,
    ##                                     :                        BGrade6}: 1 (9.7/1.1)
    ##                                     MostUsedBillingGrade in {BGrade1,BGrade2}:
    ##                                     :...AverageCoreProgramHours <= 30.16667: 0 (73.5/19.1)
    ##                                         AverageCoreProgramHours > 30.16667: [S2]
    ## 
    ## SubTree [S1]
    ## 
    ## Client_Programs_count_at_observation_cutoff > 1: 1 (16.3/1)
    ## Client_Programs_count_at_observation_cutoff <= 1:
    ## :...AverageCoreProgramHours > 10.75: 0 (15.2)
    ##     AverageCoreProgramHours <= 10.75:
    ##     :...AverageCoreProgramHours <= 2.5: 0 (139/47)
    ##         AverageCoreProgramHours > 2.5: 1 (44.5/15.8)
    ## 
    ## SubTree [S2]
    ## 
    ## AverageCoreProgramHours <= 32: 1 (13.8)
    ## AverageCoreProgramHours > 32:
    ## :...Client_Programs_count_at_observation_cutoff > 1.476779: 0 (14/2.4)
    ##     Client_Programs_count_at_observation_cutoff <= 1.476779:
    ##     :...Client_Programs_count_at_observation_cutoff <= 0.7600639: 0 (37.8/12.4)
    ##         Client_Programs_count_at_observation_cutoff > 0.7600639: 1 (157.4/69.8)
    ## 
    ## -----  Trial 8:  -----
    ## 
    ## Decision tree:
    ## 
    ## AverageCoreProgramHours <= 17.875:
    ## :...MostUsedBillingGrade in {BGrade4,BGrade5}: 1 (60/11.1)
    ## :   MostUsedBillingGrade in {BGrade1,BGrade2,BGrade3,BGrade6,BGrade9}:
    ## :   :...AgeAtCreation > 78: 0 (903.9/325.7)
    ## :       AgeAtCreation <= 78:
    ## :       :...HOME_STR_state in {ACT,QLD}: 1 (270.7/109.7)
    ## :           HOME_STR_state in {NSW,VIC,WA}: 0 (572.9/250.2)
    ## :           HOME_STR_state = SA:
    ## :           :...MostUsedBillingGrade in {BGrade1,BGrade9}: 0 (82.8/23.8)
    ## :               MostUsedBillingGrade in {BGrade2,BGrade3,
    ## :                                        BGrade6}: 1 (249.6/104.9)
    ## AverageCoreProgramHours > 17.875:
    ## :...ClientType in {DVA,Nur}: 0 (0)
    ##     ClientType in {Soc,You}: 1 (34.6/7.5)
    ##     ClientType in {Bro,CAC,CCP,Com,Dem,Dis,Dom,EAC,HAC,NRC,Per,Pri,Res,TAC,TCP,
    ##     :              VHC}:
    ##     :...Issues_Raised <= 0.6829327: 0 (594.3/7.2)
    ##         Issues_Raised > 0.6829327:
    ##         :...default_contract_group = DVA/VHC: 0 (0)
    ##             default_contract_group = Disability: 1 (15.2/5.4)
    ##             default_contract_group in {CHSP,Package,Private/Commercial,
    ##             :                          TransPac}:
    ##             :...AverageCoreProgramHours > 187.5: 0 (131.7)
    ##                 AverageCoreProgramHours <= 187.5:
    ##                 :...AgeAtCreation > 77: 0 (386.4/43.4)
    ##                     AgeAtCreation <= 77:
    ##                     :...Client_Initiated_Cancellations > 2.428169: 0 (170.4/25.6)
    ##                         Client_Initiated_Cancellations <= 2.428169:
    ##                         :...Issues_Raised > 1.059588: 0 (62/3.3)
    ##                             Issues_Raised <= 1.059588: [S1]
    ## 
    ## SubTree [S1]
    ## 
    ## default_contract_group in {Package,TransPac}: 1 (53/16.5)
    ## default_contract_group in {CHSP,Private/Commercial}:
    ## :...MostUsedBillingGrade = BGrade3: 0 (6.1)
    ##     MostUsedBillingGrade in {BGrade4,BGrade5,BGrade6,BGrade9}: 1 (5.3)
    ##     MostUsedBillingGrade in {BGrade1,BGrade2}:
    ##     :...AgeAtCreation <= 45: 1 (19.8/2.3)
    ##         AgeAtCreation > 45: 0 (220.3/82.9)
    ## 
    ## -----  Trial 9:  -----
    ## 
    ## Decision tree:
    ## 
    ## AverageCoreProgramHours > 17.875: 0 (1472.2/120)
    ## AverageCoreProgramHours <= 17.875:
    ## :...MostUsedBillingGrade in {BGrade4,BGrade5,BGrade6,BGrade9}:
    ##     :...HOME_STR_state in {ACT,NSW,VIC,WA}: 0 (148.8/51.5)
    ##     :   HOME_STR_state in {QLD,SA}: 1 (360.9/51.8)
    ##     MostUsedBillingGrade in {BGrade1,BGrade2,BGrade3}:
    ##     :...AgeAtCreation > 84: 0 (311.2/56.2)
    ##         AgeAtCreation <= 84:
    ##         :...ClientType in {Dem,Dis,DVA,NRC,Nur,TCP,VHC,You}: 0 (0)
    ##             ClientType in {Bro,EAC,TAC}: 1 (50.3/4.8)
    ##             ClientType in {CAC,CCP,Com,Dom,HAC,Per,Pri,Res,Soc}:
    ##             :...default_contract_group in {DVA/VHC,Package,
    ##                 :                          TransPac}: 0 (58.2/9.9)
    ##                 default_contract_group in {CHSP,Disability,Private/Commercial}:
    ##                 :...Client_Initiated_Cancellations > 3.071223: 0 (75.7/6.6)
    ##                     Client_Initiated_Cancellations <= 3.071223:
    ##                     :...AverageCoreProgramHours > 17.375: 1 (25.4/3.5)
    ##                         AverageCoreProgramHours <= 17.375:
    ##                         :...Issues_Raised > 0.554844: [S1]
    ##                             Issues_Raised <= 0.554844: [S2]
    ## 
    ## SubTree [S1]
    ## 
    ## Client_Programs_count_at_observation_cutoff > 1.01361: 0 (24.7/5.5)
    ## Client_Programs_count_at_observation_cutoff <= 1.01361:
    ## :...Issues_Raised > 2.016513: 1 (24.7/4.1)
    ##     Issues_Raised <= 2.016513:
    ##     :...HOME_STR_state in {VIC,WA}: 1 (0)
    ##         HOME_STR_state in {QLD,SA}: 0 (85.2/33.3)
    ##         HOME_STR_state in {ACT,NSW}:
    ##         :...default_contract_group in {CHSP,Disability}: 1 (328.2/113.4)
    ##             default_contract_group = Private/Commercial: 0 (9.5)
    ## 
    ## SubTree [S2]
    ## 
    ## Client_Programs_count_at_observation_cutoff > 0.2730071: 1 (52.9/15.8)
    ## Client_Programs_count_at_observation_cutoff <= 0.2730071:
    ## :...AverageCoreProgramHours > 9: 0 (139.5/11.6)
    ##     AverageCoreProgramHours <= 9:
    ##     :...MostUsedBillingGrade = BGrade3: 0 (129.5/26.4)
    ##         MostUsedBillingGrade in {BGrade1,BGrade2}:
    ##         :...HCW_Ratio > 2: 0 (70.9/22.4)
    ##             HCW_Ratio <= 2:
    ##             :...Client_Initiated_Cancellations > 0.4943928: 1 (78.9/14.1)
    ##                 Client_Initiated_Cancellations <= 0.4943928:
    ##                 :...AverageCoreProgramHours > 7.25: 1 (35.6/5.5)
    ##                     AverageCoreProgramHours <= 7.25: [S3]
    ## 
    ## SubTree [S3]
    ## 
    ## default_contract_group in {CHSP,Disability}: 0 (100.9/33.8)
    ## default_contract_group = Private/Commercial: 1 (65.8/21.6)
    ## 
    ## 
    ## Evaluation on training data (4130 cases):
    ## 
    ## Trial        Decision Tree   
    ## -----      ----------------  
    ##    Size      Errors  
    ## 
    ##    0     35  643(15.6%)
    ##    1     18  829(20.1%)
    ##    2     20  916(22.2%)
    ##    3     19  907(22.0%)
    ##    4     14 1009(24.4%)
    ##    5     34 1145(27.7%)
    ##    6     16  868(21.0%)
    ##    7     33  866(21.0%)
    ##    8     18  726(17.6%)
    ##    9     21  701(17.0%)
    ## boost            615(14.9%)   <<
    ## 
    ## 
    ##     (a)   (b)    <-classified as
    ##    ----  ----
    ##    3215    89    (a): class 0
    ##     526   300    (b): class 1
    ## 
    ## 
    ##  Attribute usage:
    ## 
    ##  100.00% ClientType
    ##  100.00% AgeAtCreation
    ##  100.00% MostUsedBillingGrade
    ##  100.00% AverageCoreProgramHours
    ##  100.00% Issues_Raised
    ##   96.34% HOME_STR_state
    ##   81.28% default_contract_group
    ##   53.95% HCW_Ratio
    ##   49.66% Client_Initiated_Cancellations
    ##   41.86% Client_Programs_count_at_observation_cutoff
    ## 
    ## 
    ## Time: 0.1 secs

``` r
myTree <- C50:::as.party.C5.0(kc.c50)
plot(myTree)
```

![](kc_client_churn_FINAL1A_files/figure-markdown_github/cars-5.png)

``` r
set.seed(1)
kc.c50.rules <-
  C50::C5.0(
    x = train[,-ndxLabel],
    y = train$Label,
    trial = C5.0Trials_param,
    rules = TRUE,
    control = C5.0Control(bands = 100, earlyStopping = TRUE)
  )
summary(kc.c50.rules)
```

    ## 
    ## Call:
    ## C5.0.default(x = train[, -ndxLabel], y = train$Label, trials
    ##  = C5.0Trials_param, rules = TRUE, control = C5.0Control(bands =
    ##  100, earlyStopping = TRUE))
    ## 
    ## 
    ## C5.0 [Release 2.07 GPL Edition]      Sun Apr  9 18:15:09 2017
    ## -------------------------------
    ##     **  Warning (-u): rule ordering has no effect on boosting
    ## 
    ## Class specified by attribute `outcome'
    ## 
    ## Read 4130 cases (11 attributes) from undefined.data
    ## 
    ## -----  Trial 0:  -----
    ## 
    ## Rules:
    ## 
    ## Rule 0/1: (2879/468, lift 1.0)
    ##  HOME_STR_state in {ACT, NSW, VIC, WA}
    ##  ->  class 0  [0.837]
    ## 
    ## Rule 0/2: (2686/297, lift 1.1)
    ##  MostUsedBillingGrade in {BGrade1, BGrade2, BGrade3}
    ##  AverageCoreProgramHours > 17.875
    ##  ->  class 0  [0.889]
    ## 
    ## Rule 0/3: (1393/874, lift 1.9)
    ##  AverageCoreProgramHours <= 17.875
    ##  ->  class 1  [0.373]
    ## 
    ## Rule 0/4: (945/96, lift 1.1)
    ##  HOME_STR_state in {NSW, SA, VIC}
    ##  ClientType = HAC
    ##  MostUsedBillingGrade in {BGrade1, BGrade3}
    ##  default_contract_group = CHSP
    ##  Issues_Raised <= 0.554844
    ##  ->  class 0  [0.898]
    ## 
    ## Rule 0/5: (1120/120, lift 1.1)
    ##  AgeAtCreation > 84
    ##  MostUsedBillingGrade in {BGrade1, BGrade2, BGrade3}
    ##  ->  class 0  [0.892]
    ## 
    ## Rule 0/6: (107/45, lift 2.9)
    ##  HOME_STR_state = NSW
    ##  AgeAtCreation <= 84
    ##  MostUsedBillingGrade in {BGrade1, BGrade2, BGrade3}
    ##  default_contract_group in {CHSP, Package}
    ##  Client_Initiated_Cancellations <= 5.047571
    ##  AverageCoreProgramHours <= 17.875
    ##  Issues_Raised > 0.554844
    ##  Issues_Raised <= 1.496931
    ##  ->  class 1  [0.578]
    ## 
    ## Rule 0/7: (457/107, lift 1.0)
    ##  HOME_STR_state = QLD
    ##  ->  class 0  [0.765]
    ## 
    ## Rule 0/8: (35/7, lift 3.9)
    ##  HOME_STR_state in {QLD, SA}
    ##  MostUsedBillingGrade in {BGrade6, BGrade9}
    ##  AverageCoreProgramHours > 2.5
    ##  HCW_Ratio <= 0
    ##  ->  class 1  [0.784]
    ## 
    ## Rule 0/9: (41/14, lift 3.3)
    ##  HOME_STR_state in {ACT, QLD}
    ##  ClientType = HAC
    ##  AgeAtCreation <= 84
    ##  MostUsedBillingGrade = BGrade1
    ##  Client_Initiated_Cancellations <= 5.047571
    ##  AverageCoreProgramHours <= 9.666667
    ##  Issues_Raised <= 0.554844
    ##  ->  class 1  [0.651]
    ## 
    ## Rule 0/10: (11/1, lift 4.2)
    ##  ClientType = Dom
    ##  AgeAtCreation <= 73
    ##  Client_Initiated_Cancellations <= 5.047571
    ##  AverageCoreProgramHours <= 17.875
    ##  ->  class 1  [0.846]
    ## 
    ## Rule 0/11: (15/3, lift 3.8)
    ##  ClientType in {EAC, TAC}
    ##  AgeAtCreation <= 84
    ##  AverageCoreProgramHours <= 17.875
    ##  ->  class 1  [0.765]
    ## 
    ## Rule 0/12: (566/79, lift 1.1)
    ##  ClientType in {Bro, CAC, CCP, Com, Dem, Dis, NRC, Per, Pri, Res, Soc,
    ##                        TCP, VHC, You}
    ##  MostUsedBillingGrade in {BGrade1, BGrade2, BGrade3}
    ##  ->  class 0  [0.859]
    ## 
    ## Rule 0/13: (9/2, lift 3.6)
    ##  HOME_STR_state in {ACT, NSW, VIC}
    ##  MostUsedBillingGrade in {BGrade4, BGrade6, BGrade9}
    ##  AverageCoreProgramHours <= 5.5
    ##  Issues_Raised > 0.5258737
    ##  ->  class 1  [0.727]
    ## 
    ## Rule 0/14: (7/1, lift 3.9)
    ##  HOME_STR_state = NSW
    ##  ClientType = HAC
    ##  AgeAtCreation <= 84
    ##  default_contract_group = CHSP
    ##  AverageCoreProgramHours <= 6.833333
    ##  Issues_Raised > 1.496931
    ##  ->  class 1  [0.778]
    ## 
    ## Rule 0/15: (5, lift 4.3)
    ##  ClientType = Dom
    ##  AgeAtCreation <= 84
    ##  Client_Programs_count_at_observation_cutoff <= 0.4849309
    ##  Client_Initiated_Cancellations > 1.497191
    ##  Client_Initiated_Cancellations <= 5.047571
    ##  AverageCoreProgramHours <= 17.875
    ##  ->  class 1  [0.857]
    ## 
    ## Rule 0/16: (27, lift 1.2)
    ##  MostUsedBillingGrade in {BGrade1, BGrade2, BGrade3}
    ##  Client_Initiated_Cancellations > 5.047571
    ##  AverageCoreProgramHours <= 17.875
    ##  ->  class 0  [0.966]
    ## 
    ## Rule 0/17: (1186/79, lift 1.2)
    ##  ClientType = HAC
    ##  AverageCoreProgramHours > 11.25
    ##  Issues_Raised <= 0.554844
    ##  ->  class 0  [0.933]
    ## 
    ## Default class: 0
    ## 
    ## -----  Trial 1:  -----
    ## 
    ## Rules:
    ## 
    ## Rule 1/1: (3261.6/987.8, lift 1.0)
    ##  default_contract_group in {CHSP, Disability, Package}
    ##  ->  class 0  [0.697]
    ## 
    ## Rule 1/2: (1807.9/349.3, lift 1.2)
    ##  ClientType in {Bro, CAC, CCP, Com, Dem, Dis, Dom, EAC, HAC, NRC, Per,
    ##                        Pri, Res, TAC, TCP}
    ##  AverageCoreProgramHours > 37.5
    ##  ->  class 0  [0.806]
    ## 
    ## Rule 1/3: (467.8/227.1, lift 1.6)
    ##  default_contract_group = Private/Commercial
    ##  AverageCoreProgramHours <= 37.5
    ##  ->  class 1  [0.515]
    ## 
    ## Rule 1/4: (107.1/19.3, lift 1.2)
    ##  default_contract_group in {DVA/VHC, TransPac}
    ##  ->  class 0  [0.814]
    ## 
    ## Rule 1/5: (22.5/4, lift 2.5)
    ##  ClientType in {Soc, You}
    ##  AverageCoreProgramHours > 37.5
    ##  ->  class 1  [0.796]
    ## 
    ## Rule 1/6: (8, lift 1.3)
    ##  HOME_STR_state = WA
    ##  default_contract_group = Private/Commercial
    ##  AverageCoreProgramHours <= 37.5
    ##  ->  class 0  [0.900]
    ## 
    ## Default class: 0
    ## 
    ## -----  Trial 2:  -----
    ## 
    ## Rules:
    ## 
    ## Rule 2/1: (2261.6/739.8, lift 1.1)
    ##  Issues_Raised <= 0.5329809
    ##  ->  class 0  [0.673]
    ## 
    ## Rule 2/2: (1584.3/400.8, lift 1.2)
    ##  AverageCoreProgramHours > 43
    ##  ->  class 0  [0.747]
    ## 
    ## Rule 2/3: (877.7/371.6, lift 1.5)
    ##  default_contract_group in {CHSP, Package}
    ##  AverageCoreProgramHours <= 43
    ##  Issues_Raised > 0.5329809
    ##  ->  class 1  [0.576]
    ## 
    ## Rule 2/4: (118.2/31.3, lift 1.2)
    ##  default_contract_group in {Disability, Private/Commercial, TransPac}
    ##  AverageCoreProgramHours <= 43
    ##  Issues_Raised > 0.5329809
    ##  ->  class 0  [0.731]
    ## 
    ## Rule 2/5: (86.2/26.4, lift 1.8)
    ##  AgeAtCreation <= 84
    ##  MostUsedBillingGrade = BGrade1
    ##  AverageCoreProgramHours <= 4.25
    ##  ->  class 1  [0.690]
    ## 
    ## Default class: 0
    ## 
    ## -----  Trial 3:  -----
    ## 
    ## Rules:
    ## 
    ## Rule 3/1: (1336.9/365.9, lift 1.2)
    ##  AgeAtCreation > 78
    ##  AverageCoreProgramHours > 13.75
    ##  ->  class 0  [0.726]
    ## 
    ## Rule 3/2: (1536.1/729.8, lift 1.3)
    ##  AverageCoreProgramHours <= 13.75
    ##  ->  class 1  [0.525]
    ## 
    ## Rule 3/3: (1152.9/292.8, lift 1.3)
    ##  AverageCoreProgramHours > 13.75
    ##  Issues_Raised <= 0.6829327
    ##  ->  class 0  [0.746]
    ## 
    ## Rule 3/4: (555/250.5, lift 1.3)
    ##  AgeAtCreation <= 78
    ##  Client_Initiated_Cancellations <= 4.628976
    ##  AverageCoreProgramHours > 13.75
    ##  Issues_Raised > 0.6829327
    ##  ->  class 1  [0.548]
    ## 
    ## Rule 3/5: (472.6/112.5, lift 1.3)
    ##  Client_Initiated_Cancellations > 4.628976
    ##  AverageCoreProgramHours > 13.75
    ##  ->  class 0  [0.761]
    ## 
    ## Rule 3/6: (633.3/196.9, lift 1.2)
    ##  AgeAtCreation > 83
    ##  AgeAtCreation <= 86
    ##  ->  class 0  [0.688]
    ## 
    ## Rule 3/7: (201.7/69.6, lift 1.6)
    ##  HOME_STR_state in {ACT, QLD}
    ##  AgeAtCreation <= 83
    ##  AverageCoreProgramHours <= 13.75
    ##  ->  class 1  [0.653]
    ## 
    ## Rule 3/8: (2377.8/870.9, lift 1.1)
    ##  MostUsedBillingGrade = BGrade1
    ##  ->  class 0  [0.634]
    ## 
    ## Rule 3/9: (283.1/119.7, lift 1.4)
    ##  AgeAtCreation <= 78
    ##  default_contract_group = CHSP
    ##  Client_Initiated_Cancellations <= 4.628976
    ##  AverageCoreProgramHours > 13.75
    ##  AverageCoreProgramHours <= 127
    ##  HCW_Ratio > 2
    ##  Issues_Raised > 0.6829327
    ##  ->  class 1  [0.576]
    ## 
    ## Rule 3/10: (145.9/52.2, lift 1.6)
    ##  AgeAtCreation <= 78
    ##  default_contract_group in {Disability, Package, TransPac}
    ##  Client_Initiated_Cancellations <= 4.628976
    ##  Issues_Raised > 0.6829327
    ##  ->  class 1  [0.640]
    ## 
    ## Rule 3/11: (398.5/105.2, lift 1.2)
    ##  AgeAtCreation > 88
    ##  ->  class 0  [0.735]
    ## 
    ## Rule 3/12: (53.6/11.9, lift 1.9)
    ##  AgeAtCreation <= 45
    ##  Client_Initiated_Cancellations <= 4.628976
    ##  AverageCoreProgramHours > 13.75
    ##  Issues_Raised > 0.6829327
    ##  ->  class 1  [0.767]
    ## 
    ## Rule 3/13: (62.1/24.3, lift 1.5)
    ##  HOME_STR_state in {NSW, SA}
    ##  AgeAtCreation <= 83
    ##  AverageCoreProgramHours <= 13.75
    ##  Issues_Raised > 1.496931
    ##  ->  class 1  [0.606]
    ## 
    ## Rule 3/14: (325.1/102.5, lift 1.2)
    ##  AgeAtCreation > 45
    ##  default_contract_group = Private/Commercial
    ##  AverageCoreProgramHours > 13.75
    ##  ->  class 0  [0.684]
    ## 
    ## Default class: 0
    ## 
    ## -----  Trial 4:  -----
    ## 
    ## Rules:
    ## 
    ## Rule 4/1: (1028.9/335.9, lift 1.2)
    ##  ClientType = HAC
    ##  HCW_Ratio > 6
    ##  ->  class 0  [0.673]
    ## 
    ## Rule 4/2: (818/362.9, lift 1.0)
    ##  default_contract_group = Private/Commercial
    ##  ->  class 0  [0.556]
    ## 
    ## Rule 4/3: (769.5/246.4, lift 1.2)
    ##  ClientType = HAC
    ##  Client_Programs_count_at_observation_cutoff <= 0.6227208
    ##  AverageCoreProgramHours > 11
    ##  ->  class 0  [0.679]
    ## 
    ## Rule 4/4: (397.1/170.8, lift 1.4)
    ##  HOME_STR_state in {ACT, QLD, SA, WA}
    ##  ClientType = HAC
    ##  default_contract_group = CHSP
    ##  AverageCoreProgramHours <= 11
    ##  ->  class 1  [0.569]
    ## 
    ## Rule 4/5: (891.8/290.9, lift 1.2)
    ##  AgeAtCreation > 61
    ##  MostUsedBillingGrade in {BGrade1, BGrade2}
    ##  default_contract_group = CHSP
    ##  AverageCoreProgramHours > 25
    ##  Issues_Raised <= 2.206215
    ##  ->  class 0  [0.673]
    ## 
    ## Rule 4/6: (444.7/146.4, lift 1.1)
    ##  HOME_STR_state = NSW
    ##  ClientType = HAC
    ##  AgeAtCreation > 73
    ##  Client_Programs_count_at_observation_cutoff <= 0.6227208
    ##  default_contract_group = CHSP
    ##  Issues_Raised <= 1.923614
    ##  ->  class 0  [0.670]
    ## 
    ## Rule 4/7: (421.7/102, lift 1.3)
    ##  ClientType in {CAC, CCP, Com, Dis, Dom, EAC, HAC, Per, Pri}
    ##  AverageCoreProgramHours > 176.5
    ##  ->  class 0  [0.757]
    ## 
    ## Rule 4/8: (258.4/93, lift 1.5)
    ##  ClientType = HAC
    ##  Client_Programs_count_at_observation_cutoff > 0.6227208
    ##  AverageCoreProgramHours <= 25
    ##  HCW_Ratio <= 6
    ##  ->  class 1  [0.639]
    ## 
    ## Rule 4/9: (160.6/67, lift 1.4)
    ##  ClientType in {CAC, CCP, Dom, EAC, Res, Soc, TAC}
    ##  AverageCoreProgramHours <= 25
    ##  ->  class 1  [0.582]
    ## 
    ## Rule 4/10: (124.1/44.8, lift 1.5)
    ##  HOME_STR_state = NSW
    ##  AgeAtCreation <= 73
    ##  Client_Programs_count_at_observation_cutoff <= 0.1647091
    ##  default_contract_group = CHSP
    ##  AverageCoreProgramHours <= 11
    ##  ->  class 1  [0.637]
    ## 
    ## Rule 4/11: (416/147.3, lift 1.1)
    ##  ClientType = HAC
    ##  default_contract_group in {Package, TransPac}
    ##  ->  class 0  [0.645]
    ## 
    ## Rule 4/12: (660.1/201, lift 1.2)
    ##  ClientType in {CAC, CCP, Com, Dis, Dom, EAC, HAC, Per, Pri}
    ##  MostUsedBillingGrade in {BGrade1, BGrade2}
    ##  Client_Initiated_Cancellations > 2.85118
    ##  AverageCoreProgramHours > 25
    ##  ->  class 0  [0.695]
    ## 
    ## Rule 4/13: (849.7/223.2, lift 1.3)
    ##  AverageCoreProgramHours > 25
    ##  Issues_Raised <= 0.6829327
    ##  ->  class 0  [0.737]
    ## 
    ## Rule 4/14: (128.2/49.3, lift 1.5)
    ##  ClientType = HAC
    ##  MostUsedBillingGrade in {BGrade1, BGrade3, BGrade5}
    ##  default_contract_group = Private/Commercial
    ##  AverageCoreProgramHours <= 11
    ##  Issues_Raised <= 0.4970248
    ##  ->  class 1  [0.613]
    ## 
    ## Rule 4/15: (68.4/25.5, lift 1.5)
    ##  MostUsedBillingGrade in {BGrade3, BGrade6}
    ##  AverageCoreProgramHours > 25
    ##  AverageCoreProgramHours <= 176.5
    ##  Issues_Raised > 0.6829327
    ##  ->  class 1  [0.624]
    ## 
    ## Rule 4/16: (175.2/76, lift 1.4)
    ##  MostUsedBillingGrade in {BGrade1, BGrade2}
    ##  default_contract_group in {Disability, Package}
    ##  Client_Initiated_Cancellations <= 2.85118
    ##  AverageCoreProgramHours <= 176.5
    ##  Issues_Raised > 0.6829327
    ##  ->  class 1  [0.565]
    ## 
    ## Rule 4/17: (38.2/11.8, lift 1.6)
    ##  HOME_STR_state = NSW
    ##  Client_Programs_count_at_observation_cutoff <= 0.1647091
    ##  default_contract_group = CHSP
    ##  AverageCoreProgramHours <= 11
    ##  Issues_Raised > 1.923614
    ##  ->  class 1  [0.682]
    ## 
    ## Rule 4/18: (25.5/5.8, lift 1.8)
    ##  ClientType in {Soc, You}
    ##  AverageCoreProgramHours > 25
    ##  ->  class 1  [0.752]
    ## 
    ## Rule 4/19: (29.8/5.3, lift 1.4)
    ##  Client_Programs_count_at_observation_cutoff > 0.1647091
    ##  Client_Programs_count_at_observation_cutoff <= 0.6227208
    ##  AverageCoreProgramHours <= 11
    ##  ->  class 0  [0.801]
    ## 
    ## Rule 4/20: (26.7/5.4, lift 1.9)
    ##  AgeAtCreation <= 61
    ##  MostUsedBillingGrade in {BGrade1, BGrade2}
    ##  default_contract_group = CHSP
    ##  Client_Initiated_Cancellations <= 2.85118
    ##  AverageCoreProgramHours > 25
    ##  AverageCoreProgramHours <= 176.5
    ##  Issues_Raised > 0.6829327
    ##  Issues_Raised <= 2.206215
    ##  ->  class 1  [0.778]
    ## 
    ## Rule 4/21: (34.1, lift 1.7)
    ##  ClientType in {Bro, Dem, NRC, Res, TAC, TCP}
    ##  AverageCoreProgramHours > 25
    ##  ->  class 0  [0.972]
    ## 
    ## Rule 4/22: (7.5/1.1, lift 1.9)
    ##  ClientType = HAC
    ##  default_contract_group = Disability
    ##  ->  class 1  [0.782]
    ## 
    ## Default class: 0
    ## 
    ## -----  Trial 5:  -----
    ## 
    ## Rules:
    ## 
    ## Rule 5/1: (2337.5/1214.3, lift 1.1)
    ##  Client_Initiated_Cancellations <= 3.399661
    ##  HCW_Ratio <= 5
    ##  ->  class 1  [0.481]
    ## 
    ## Rule 5/2: (1938.1/879.2, lift 1.0)
    ##  Issues_Raised > 0.4970248
    ##  ->  class 0  [0.546]
    ## 
    ## Rule 5/3: (542.6/138.2, lift 1.3)
    ##  AverageCoreProgramHours > 46
    ##  Issues_Raised <= 0.960986
    ##  ->  class 0  [0.744]
    ## 
    ## Rule 5/4: (466.7/157.9, lift 1.2)
    ##  MostUsedBillingGrade = BGrade1
    ##  Client_Programs_count_at_observation_cutoff <= 0.4027286
    ##  Client_Initiated_Cancellations <= 1.497191
    ##  Issues_Raised <= 0.5498743
    ##  ->  class 0  [0.661]
    ## 
    ## Rule 5/5: (489.6/227.3, lift 1.2)
    ##  MostUsedBillingGrade = BGrade1
    ##  Client_Programs_count_at_observation_cutoff > 0.4027286
    ##  default_contract_group in {CHSP, Package, TransPac}
    ##  AverageCoreProgramHours <= 46
    ##  ->  class 1  [0.536]
    ## 
    ## Rule 5/6: (163.4/62.1, lift 1.4)
    ##  AgeAtCreation <= 80
    ##  MostUsedBillingGrade in {BGrade1, BGrade2, BGrade3}
    ##  Client_Programs_count_at_observation_cutoff > 0.2147262
    ##  AverageCoreProgramHours > 46
    ##  AverageCoreProgramHours <= 187.5
    ##  Issues_Raised > 0.960986
    ##  Issues_Raised <= 1.705531
    ##  ->  class 1  [0.618]
    ## 
    ## Rule 5/7: (73.4/15.9, lift 1.4)
    ##  MostUsedBillingGrade = BGrade1
    ##  Client_Programs_count_at_observation_cutoff <= 0.4027286
    ##  Client_Initiated_Cancellations > 3.399661
    ##  AverageCoreProgramHours <= 46
    ##  HCW_Ratio <= 8
    ##  ->  class 0  [0.776]
    ## 
    ## Rule 5/8: (300.3/139.2, lift 1.2)
    ##  MostUsedBillingGrade = BGrade1
    ##  Client_Programs_count_at_observation_cutoff <= 0.4027286
    ##  Client_Initiated_Cancellations <= 2.428169
    ##  AverageCoreProgramHours <= 46
    ##  HCW_Ratio <= 8
    ##  Issues_Raised > 0.5498743
    ##  ->  class 1  [0.536]
    ## 
    ## Rule 5/9: (83.4/16, lift 1.4)
    ##  MostUsedBillingGrade = BGrade1
    ##  Client_Programs_count_at_observation_cutoff <= 0.4027286
    ##  Client_Initiated_Cancellations <= 3.399661
    ##  HCW_Ratio > 5
    ##  Issues_Raised <= 0.5498743
    ##  ->  class 0  [0.801]
    ## 
    ## Rule 5/10: (225.3/98.5, lift 1.3)
    ##  MostUsedBillingGrade = BGrade2
    ##  AverageCoreProgramHours <= 46
    ##  HCW_Ratio > 0
    ##  Issues_Raised <= 0.4970248
    ##  ->  class 1  [0.562]
    ## 
    ## Rule 5/11: (44.7/9.5, lift 1.4)
    ##  AgeAtCreation <= 65
    ##  MostUsedBillingGrade = BGrade3
    ##  Client_Initiated_Cancellations <= 1.181425
    ##  ->  class 0  [0.775]
    ## 
    ## Rule 5/12: (228.6/95, lift 1.3)
    ##  AgeAtCreation > 65
    ##  MostUsedBillingGrade = BGrade3
    ##  Client_Initiated_Cancellations <= 1.181425
    ##  HCW_Ratio <= 2
    ##  ->  class 1  [0.584]
    ## 
    ## Rule 5/13: (375.7/179.7, lift 1.2)
    ##  MostUsedBillingGrade in {BGrade4, BGrade5, BGrade6}
    ##  AverageCoreProgramHours <= 46
    ##  ->  class 1  [0.521]
    ## 
    ## Rule 5/14: (708.6/278.1, lift 1.1)
    ##  Issues_Raised > 1.705531
    ##  ->  class 0  [0.607]
    ## 
    ## Rule 5/15: (12.1/1.4, lift 1.9)
    ##  AgeAtCreation <= 74
    ##  Client_Programs_count_at_observation_cutoff > 2.218266
    ##  AverageCoreProgramHours > 46
    ##  Issues_Raised <= 0.960986
    ##  ->  class 1  [0.829]
    ## 
    ## Rule 5/16: (116/37.3, lift 1.2)
    ##  MostUsedBillingGrade = BGrade3
    ##  Client_Initiated_Cancellations > 1.181425
    ##  ->  class 0  [0.675]
    ## 
    ## Default class: 0
    ## 
    ## -----  Trial 6:  -----
    ## 
    ## Rules:
    ## 
    ## Rule 6/1: (3132.8/1194.3, lift 1.0)
    ##  ClientType = HAC
    ##  ->  class 0  [0.619]
    ## 
    ## Rule 6/2: (622.8/124.4, lift 1.3)
    ##  Client_Initiated_Cancellations > 1.181425
    ##  AverageCoreProgramHours > 43.75
    ##  ->  class 0  [0.799]
    ## 
    ## Rule 6/3: (149.2/67.6, lift 1.5)
    ##  ClientType in {EAC, Pri, Res}
    ##  AverageCoreProgramHours <= 43.75
    ##  ->  class 1  [0.547]
    ## 
    ## Rule 6/4: (470.8/31.9, lift 1.5)
    ##  AgeAtCreation > 78
    ##  AverageCoreProgramHours > 43.75
    ##  ->  class 0  [0.930]
    ## 
    ## Rule 6/5: (195.8/50.5, lift 1.2)
    ##  HOME_STR_state in {ACT, VIC, WA}
    ##  Issues_Raised <= 0.5329809
    ##  ->  class 0  [0.739]
    ## 
    ## Rule 6/6: (142/51.1, lift 1.7)
    ##  AgeAtCreation <= 78
    ##  Client_Initiated_Cancellations <= 1.181425
    ##  AverageCoreProgramHours > 43.75
    ##  Issues_Raised > 0.613282
    ##  Issues_Raised <= 1.165475
    ##  ->  class 1  [0.638]
    ## 
    ## Rule 6/7: (57.7/16.2, lift 1.9)
    ##  HOME_STR_state = NSW
    ##  AgeAtCreation <= 85
    ##  MostUsedBillingGrade in {BGrade1, BGrade2, BGrade3}
    ##  Client_Programs_count_at_observation_cutoff <= 0
    ##  Client_Initiated_Cancellations > 0.4943928
    ##  AverageCoreProgramHours <= 16.75
    ##  HCW_Ratio <= 2
    ##  Issues_Raised <= 0.5329809
    ##  ->  class 1  [0.713]
    ## 
    ## Rule 6/8: (56.8/15.9, lift 1.9)
    ##  ClientType = HAC
    ##  MostUsedBillingGrade = BGrade2
    ##  default_contract_group in {CHSP, Package, Private/Commercial}
    ##  AverageCoreProgramHours > 24.125
    ##  AverageCoreProgramHours <= 43.75
    ##  Issues_Raised > 0.5329809
    ##  ->  class 1  [0.713]
    ## 
    ## Rule 6/9: (410.1/96.2, lift 1.3)
    ##  AgeAtCreation > 85
    ##  Issues_Raised <= 0.5329809
    ##  ->  class 0  [0.764]
    ## 
    ## Rule 6/10: (400.6/15.2, lift 1.6)
    ##  Client_Programs_count_at_observation_cutoff <= 2.218266
    ##  AverageCoreProgramHours > 43.75
    ##  Issues_Raised <= 0.613282
    ##  ->  class 0  [0.960]
    ## 
    ## Rule 6/11: (409.6/190.4, lift 1.4)
    ##  HOME_STR_state in {QLD, SA}
    ##  ClientType = HAC
    ##  AgeAtCreation <= 85
    ##  AverageCoreProgramHours <= 16.75
    ##  HCW_Ratio <= 3
    ##  Issues_Raised <= 0.5329809
    ##  ->  class 1  [0.535]
    ## 
    ## Rule 6/12: (112.8/44.7, lift 1.6)
    ##  ClientType = HAC
    ##  Client_Programs_count_at_observation_cutoff > 0
    ##  Client_Initiated_Cancellations <= 4.787849
    ##  AverageCoreProgramHours <= 43.75
    ##  HCW_Ratio <= 3
    ##  Issues_Raised <= 0.5329809
    ##  ->  class 1  [0.602]
    ## 
    ## Rule 6/13: (531.1/85.6, lift 1.4)
    ##  HOME_STR_state = NSW
    ##  HCW_Ratio > 2
    ##  Issues_Raised <= 0.5329809
    ##  ->  class 0  [0.838]
    ## 
    ## Rule 6/14: (48.5/12.3, lift 2.0)
    ##  HOME_STR_state = NSW
    ##  AgeAtCreation <= 85
    ##  MostUsedBillingGrade in {BGrade1, BGrade2, BGrade3}
    ##  Client_Programs_count_at_observation_cutoff <= 0
    ##  default_contract_group = Private/Commercial
    ##  AverageCoreProgramHours <= 43.75
    ##  HCW_Ratio <= 2
    ##  Issues_Raised <= 0.5329809
    ##  ->  class 1  [0.737]
    ## 
    ## Rule 6/15: (34.5/13.9, lift 1.6)
    ##  AgeAtCreation <= 78
    ##  Client_Programs_count_at_observation_cutoff > 2.218266
    ##  AverageCoreProgramHours > 43.75
    ##  ->  class 1  [0.591]
    ## 
    ## Default class: 0
    ## 
    ## -----  Trial 7:  -----
    ## 
    ## Rules:
    ## 
    ## Rule 7/1: (2461.3/1201.9, lift 1.4)
    ##  AverageCoreProgramHours <= 25
    ##  ->  class 1  [0.512]
    ## 
    ## Rule 7/2: (707.1/84.4, lift 1.5)
    ##  AgeAtCreation > 73
    ##  AverageCoreProgramHours > 25
    ##  Issues_Raised <= 1.059588
    ##  ->  class 0  [0.880]
    ## 
    ## Rule 7/3: (301.8/38.6, lift 1.5)
    ##  ClientType in {Bro, CAC, CCP, Com, Dem, Dis, Dom, EAC, HAC, NRC, Per,
    ##                        Pri, Res, TAC, TCP}
    ##  AverageCoreProgramHours > 25
    ##  Issues_Raised > 1.059588
    ##  ->  class 0  [0.870]
    ## 
    ## Rule 7/4: (616.3/64.4, lift 1.5)
    ##  HOME_STR_state in {NSW, QLD, SA, VIC, WA}
    ##  AgeAtCreation > 68
    ##  HCW_Ratio > 2
    ##  Issues_Raised <= 0.5329809
    ##  ->  class 0  [0.894]
    ## 
    ## Rule 7/5: (377.8/153.4, lift 1.6)
    ##  AgeAtCreation <= 73
    ##  Client_Initiated_Cancellations <= 2.021324
    ##  Issues_Raised > 0.6829327
    ##  Issues_Raised <= 1.059588
    ##  ->  class 1  [0.593]
    ## 
    ## Rule 7/6: (540.3/17, lift 1.7)
    ##  ClientType in {Bro, CAC, CCP, Com, Dem, Dis, Dom, EAC, HAC, NRC, Per,
    ##                        Pri, Res, TAC}
    ##  AverageCoreProgramHours > 25
    ##  Issues_Raised <= 0.6829327
    ##  ->  class 0  [0.967]
    ## 
    ## Rule 7/7: (284.2/55.5, lift 1.4)
    ##  AgeAtCreation > 84
    ##  default_contract_group = CHSP
    ##  Issues_Raised <= 0.5329809
    ##  ->  class 0  [0.803]
    ## 
    ## Rule 7/8: (463.2/31.6, lift 1.6)
    ##  ClientType in {Bro, CAC, CCP, Com, Dem, Dis, Dom, EAC, HAC, NRC, Per,
    ##                        Pri, Res, TAC, TCP}
    ##  Client_Initiated_Cancellations > 2.021324
    ##  AverageCoreProgramHours > 25
    ##  ->  class 0  [0.930]
    ## 
    ## Rule 7/9: (31.2/6.2, lift 2.1)
    ##  ClientType in {Soc, You}
    ##  AverageCoreProgramHours > 25
    ##  ->  class 1  [0.783]
    ## 
    ## Rule 7/10: (77.8/4.3, lift 1.6)
    ##  AgeAtCreation > 92
    ##  ->  class 0  [0.933]
    ## 
    ## Rule 7/11: (189.4/30.7, lift 1.4)
    ##  default_contract_group = Package
    ##  Issues_Raised <= 0.5329809
    ##  ->  class 0  [0.834]
    ## 
    ## Rule 7/12: (17.9, lift 1.6)
    ##  ClientType in {Bro, CAC, Com, Dom, NRC}
    ##  AgeAtCreation <= 73
    ##  AverageCoreProgramHours > 25
    ##  Issues_Raised > 0.6829327
    ##  ->  class 0  [0.950]
    ## 
    ## Rule 7/13: (617.5/43.7, lift 1.6)
    ##  AverageCoreProgramHours > 13.75
    ##  HCW_Ratio > 2
    ##  Issues_Raised <= 0.5329809
    ##  ->  class 0  [0.928]
    ## 
    ## Rule 7/14: (60/7.2, lift 1.5)
    ##  default_contract_group in {DVA/VHC, TransPac}
    ##  ->  class 0  [0.867]
    ## 
    ## Default class: 0
    ## 
    ## -----  Trial 8:  -----
    ## 
    ## Rules:
    ## 
    ## Rule 8/1: (1967.3/494.3, lift 1.1)
    ##  HOME_STR_state in {ACT, NSW, VIC, WA}
    ##  ClientType = HAC
    ##  ->  class 0  [0.748]
    ## 
    ## Rule 8/2: (785/222.5, lift 1.1)
    ##  ClientType in {Bro, CAC, CCP, Com, Dem, Dis, Dom, DVA, EAC, NRC, Nur,
    ##                        Pri, Res, Soc, TCP, VHC, You}
    ##  ->  class 0  [0.716]
    ## 
    ## Rule 8/3: (1617.8/314.7, lift 1.2)
    ##  ClientType = HAC
    ##  default_contract_group in {CHSP, Package, Private/Commercial}
    ##  AverageCoreProgramHours > 7.5
    ##  HCW_Ratio > 1
    ##  ->  class 0  [0.805]
    ## 
    ## Rule 8/4: (260.5/111.8, lift 2.1)
    ##  HOME_STR_state = SA
    ##  ClientType = HAC
    ##  AgeAtCreation > 31
    ##  AgeAtCreation <= 83
    ##  MostUsedBillingGrade in {BGrade1, BGrade2, BGrade3, BGrade5, BGrade6}
    ##  AverageCoreProgramHours <= 4.916667
    ##  ->  class 1  [0.570]
    ## 
    ## Rule 8/5: (973.6/184.6, lift 1.2)
    ##  AgeAtCreation > 83
    ##  ->  class 0  [0.810]
    ## 
    ## Rule 8/6: (140.3/55.4, lift 2.2)
    ##  HOME_STR_state = QLD
    ##  ClientType = HAC
    ##  AgeAtCreation <= 83
    ##  default_contract_group in {CHSP, Package, Private/Commercial}
    ##  AverageCoreProgramHours <= 7.5
    ##  ->  class 1  [0.604]
    ## 
    ## Rule 8/7: (119.2/21.8, lift 1.2)
    ##  HOME_STR_state = SA
    ##  ClientType = HAC
    ##  AgeAtCreation <= 83
    ##  Client_Programs_count_at_observation_cutoff <= 0.5088415
    ##  default_contract_group in {CHSP, Package, Private/Commercial}
    ##  AverageCoreProgramHours > 4.916667
    ##  ->  class 0  [0.812]
    ## 
    ## Rule 8/8: (1231/127, lift 1.4)
    ##  AverageCoreProgramHours > 25
    ##  ->  class 0  [0.896]
    ## 
    ## Rule 8/9: (18.3/6.4, lift 2.3)
    ##  ClientType in {Per, TAC}
    ##  AgeAtCreation <= 83
    ##  AverageCoreProgramHours <= 25
    ##  ->  class 1  [0.634]
    ## 
    ## Rule 8/10: (683/64.2, lift 1.4)
    ##  ClientType = HAC
    ##  default_contract_group in {CHSP, Package, Private/Commercial}
    ##  AverageCoreProgramHours > 7.5
    ##  Issues_Raised <= 0.4970248
    ##  ->  class 0  [0.905]
    ## 
    ## Rule 8/11: (14/1.8, lift 3.0)
    ##  AgeAtCreation <= 83
    ##  Client_Programs_count_at_observation_cutoff <= 0.5088415
    ##  AverageCoreProgramHours > 7.5
    ##  AverageCoreProgramHours <= 25
    ##  HCW_Ratio <= 1
    ##  Issues_Raised > 0.4970248
    ##  ->  class 1  [0.826]
    ## 
    ## Rule 8/12: (73.3/17.8, lift 2.7)
    ##  ClientType = HAC
    ##  AgeAtCreation <= 83
    ##  Client_Programs_count_at_observation_cutoff > 0.5088415
    ##  default_contract_group in {CHSP, Private/Commercial}
    ##  AverageCoreProgramHours <= 5.75
    ##  ->  class 1  [0.750]
    ## 
    ## Rule 8/13: (6.6, lift 1.3)
    ##  HOME_STR_state = SA
    ##  AgeAtCreation <= 81
    ##  MostUsedBillingGrade = BGrade9
    ##  ->  class 0  [0.883]
    ## 
    ## Rule 8/14: (43.4/5.2, lift 1.3)
    ##  AgeAtCreation <= 31
    ##  Client_Programs_count_at_observation_cutoff <= 0.5088415
    ##  AverageCoreProgramHours <= 7.375
    ##  ->  class 0  [0.865]
    ## 
    ## Rule 8/15: (6.6/0.7, lift 2.9)
    ##  ClientType = HAC
    ##  default_contract_group = Disability
    ##  ->  class 1  [0.804]
    ## 
    ## Default class: 0
    ## 
    ## -----  Trial 9:  -----
    ## 
    ## Rules:
    ## 
    ## Rule 9/1: (1315.3/160.2, lift 1.5)
    ##  ClientType in {HAC, Per, Soc}
    ##  MostUsedBillingGrade in {BGrade1, BGrade2, BGrade3, BGrade9}
    ##  default_contract_group in {CHSP, Package}
    ##  AverageCoreProgramHours > 8.25
    ##  ->  class 0  [0.878]
    ## 
    ## Rule 9/2: (432.4/67.3, lift 3.1)
    ##  HOME_STR_state in {ACT, QLD, SA}
    ##  MostUsedBillingGrade in {BGrade5, BGrade6}
    ##  AverageCoreProgramHours <= 15.75
    ##  ->  class 1  [0.843]
    ## 
    ## Rule 9/3: (814.3/53.9, lift 1.6)
    ##  ClientType in {HAC, Per, Soc}
    ##  MostUsedBillingGrade in {BGrade1, BGrade3, BGrade9}
    ##  default_contract_group in {CHSP, Package, TransPac}
    ##  Issues_Raised <= 0.5329809
    ##  ->  class 0  [0.933]
    ## 
    ## Rule 9/4: (618.2, lift 1.7)
    ##  AgeAtCreation > 78
    ##  AverageCoreProgramHours > 15.75
    ##  ->  class 0  [0.998]
    ## 
    ## Rule 9/5: (266.1/71.3, lift 2.7)
    ##  AgeAtCreation <= 83
    ##  default_contract_group in {CHSP, Package}
    ##  Client_Initiated_Cancellations > -0.3962597
    ##  Client_Initiated_Cancellations <= 3
    ##  AverageCoreProgramHours <= 8.25
    ##  Issues_Raised > 0.5329809
    ##  ->  class 1  [0.730]
    ## 
    ## Rule 9/6: (558/29.9, lift 1.6)
    ##  ClientType in {HAC, Per, Pri, Soc}
    ##  AgeAtCreation > 83
    ##  MostUsedBillingGrade in {BGrade1, BGrade2, BGrade3, BGrade9}
    ##  ->  class 0  [0.945]
    ## 
    ## Rule 9/7: (402.6/86.2, lift 2.9)
    ##  HOME_STR_state in {ACT, NSW, QLD, SA}
    ##  ClientType in {HAC, Pri}
    ##  AgeAtCreation <= 83
    ##  default_contract_group in {Disability, Private/Commercial}
    ##  AverageCoreProgramHours <= 15.75
    ##  Issues_Raised <= 0.5329809
    ##  ->  class 1  [0.785]
    ## 
    ## Rule 9/8: (145.6/38.1, lift 2.7)
    ##  ClientType in {Bro, Dom, EAC}
    ##  MostUsedBillingGrade in {BGrade1, BGrade2, BGrade3}
    ##  AverageCoreProgramHours <= 15.75
    ##  ->  class 1  [0.735]
    ## 
    ## Rule 9/9: (1132.3/57, lift 1.6)
    ##  ClientType in {Bro, CCP, Dem, Dis, Dom, HAC, NRC, Per, Pri, TAC}
    ##  MostUsedBillingGrade in {BGrade1, BGrade2, BGrade3, BGrade5}
    ##  default_contract_group in {CHSP, Package, Private/Commercial, TransPac}
    ##  AverageCoreProgramHours > 15.75
    ##  ->  class 0  [0.949]
    ## 
    ## Rule 9/10: (117/33.3, lift 1.2)
    ##  HOME_STR_state in {NSW, VIC, WA}
    ##  MostUsedBillingGrade in {BGrade4, BGrade5, BGrade6}
    ##  AverageCoreProgramHours <= 15.75
    ##  ->  class 0  [0.712]
    ## 
    ## Rule 9/11: (54.7, lift 1.7)
    ##  ClientType in {CAC, CCP, Com, Dem, DVA, Res, VHC}
    ##  MostUsedBillingGrade in {BGrade1, BGrade2, BGrade3, BGrade9}
    ##  AverageCoreProgramHours <= 15.75
    ##  ->  class 0  [0.982]
    ## 
    ## Rule 9/12: (535, lift 1.7)
    ##  AverageCoreProgramHours > 15.75
    ##  Issues_Raised <= 0.960986
    ##  ->  class 0  [0.998]
    ## 
    ## Rule 9/13: (117.8/33.7, lift 2.6)
    ##  HOME_STR_state in {ACT, NSW, QLD, SA}
    ##  ClientType in {HAC, Per, Pri}
    ##  AgeAtCreation <= 83
    ##  MostUsedBillingGrade = BGrade2
    ##  Client_Initiated_Cancellations <= 3
    ##  AverageCoreProgramHours <= 15.75
    ##  Issues_Raised <= 0.5329809
    ##  ->  class 1  [0.710]
    ## 
    ## Rule 9/14: (140.5/2, lift 1.6)
    ##  MostUsedBillingGrade in {BGrade1, BGrade2, BGrade3}
    ##  default_contract_group in {Private/Commercial, TransPac}
    ##  Issues_Raised > 0.5329809
    ##  ->  class 0  [0.979]
    ## 
    ## Rule 9/15: (233.2/9.7, lift 1.6)
    ##  ClientType in {HAC, Per, Pri, Soc}
    ##  AgeAtCreation <= 83
    ##  MostUsedBillingGrade in {BGrade1, BGrade2, BGrade3, BGrade9}
    ##  Client_Initiated_Cancellations > 3
    ##  ->  class 0  [0.955]
    ## 
    ## Rule 9/16: (244.7, lift 1.7)
    ##  ClientType in {Bro, CAC, CCP, Com, Dem, Dis, Dom, EAC, HAC, NRC, Per,
    ##                        Pri, TAC, TCP}
    ##  AverageCoreProgramHours > 170.5
    ##  ->  class 0  [0.996]
    ## 
    ## Rule 9/17: (61.4, lift 1.7)
    ##  HOME_STR_state in {VIC, WA}
    ##  ClientType in {HAC, Pri}
    ##  MostUsedBillingGrade in {BGrade1, BGrade2, BGrade3}
    ##  ->  class 0  [0.984]
    ## 
    ## Rule 9/18: (37.8/11.1, lift 2.6)
    ##  ClientType in {Res, Soc, You}
    ##  AverageCoreProgramHours > 15.75
    ##  ->  class 1  [0.695]
    ## 
    ## Rule 9/19: (12.1, lift 3.5)
    ##  ClientType in {CAC, EAC}
    ##  AgeAtCreation <= 78
    ##  MostUsedBillingGrade = BGrade2
    ##  Client_Initiated_Cancellations <= 2.194883
    ##  AverageCoreProgramHours > 15.75
    ##  AverageCoreProgramHours <= 170.5
    ##  Issues_Raised > 0.960986
    ##  ->  class 1  [0.929]
    ## 
    ## Rule 9/20: (41.4/5.7, lift 1.4)
    ##  ClientType = HAC
    ##  MostUsedBillingGrade in {BGrade1, BGrade2, BGrade3, BGrade9}
    ##  Client_Initiated_Cancellations <= -0.3962597
    ##  ->  class 0  [0.845]
    ## 
    ## Rule 9/21: (23.2/1.6, lift 1.5)
    ##  HOME_STR_state in {NSW, QLD, SA}
    ##  MostUsedBillingGrade in {BGrade1, BGrade2, BGrade3, BGrade9}
    ##  Client_Initiated_Cancellations > 2
    ##  Client_Initiated_Cancellations <= 3
    ##  AverageCoreProgramHours <= 8.25
    ##  Issues_Raised > 0.5329809
    ##  ->  class 0  [0.895]
    ## 
    ## Rule 9/22: (5.4, lift 3.2)
    ##  HOME_STR_state = NSW
    ##  default_contract_group = Disability
    ##  Client_Initiated_Cancellations <= 2.194883
    ##  Issues_Raised > 0.960986
    ##  ->  class 1  [0.865]
    ## 
    ## Rule 9/23: (61.8/23, lift 2.3)
    ##  AgeAtCreation <= 61
    ##  MostUsedBillingGrade in {BGrade1, BGrade2, BGrade3}
    ##  Client_Initiated_Cancellations <= 2.194883
    ##  AverageCoreProgramHours > 15.75
    ##  AverageCoreProgramHours <= 170.5
    ##  Issues_Raised > 0.960986
    ##  ->  class 1  [0.624]
    ## 
    ## Default class: 0
    ## 
    ## 
    ## Evaluation on training data (4130 cases):
    ## 
    ## Trial            Rules     
    ## -----      ----------------
    ##      No      Errors
    ## 
    ##    0     17  664(16.1%)
    ##    1      6  828(20.0%)
    ##    2      5 1002(24.3%)
    ##    3     14  925(22.4%)
    ##    4     22  901(21.8%)
    ##    5     16 1146(27.7%)
    ##    6     15  812(19.7%)
    ##    7     14  946(22.9%)
    ##    8     15  742(18.0%)
    ##    9     23  705(17.1%)
    ## boost            654(15.8%)   <<
    ## 
    ## 
    ##     (a)   (b)    <-classified as
    ##    ----  ----
    ##    3223    81    (a): class 0
    ##     573   253    (b): class 1
    ## 
    ## 
    ##  Attribute usage:
    ## 
    ##  100.00% ClientType
    ##  100.00% default_contract_group
    ##  100.00% AverageCoreProgramHours
    ##  100.00% Issues_Raised
    ##   99.42% MostUsedBillingGrade
    ##   94.72% HOME_STR_state
    ##   94.21% AgeAtCreation
    ##   90.99% HCW_Ratio
    ##   86.37% Client_Initiated_Cancellations
    ##   70.44% Client_Programs_count_at_observation_cutoff
    ## 
    ## 
    ## Time: 0.2 secs

``` r
# make predictions
pr.kc.c50 <-
  predict(kc.c50, type = "prob", newdata = test[-ndxLabel])[, 2]
fitted.results.c50 <- ifelse(pr.kc.c50 > fitThreshold, 1, 0)
confusion_maxtix <- table(test$Label, fitted.results.c50)
print.table(confusion_maxtix)
```

    ##    fitted.results.c50
    ##       0   1
    ##   0 798   2
    ##   1 167  33

``` r
print(paste(
  'C5.0 Accuracy',
  computePerformanceMeasures(confusion_maxtix, fScoreB_param)[1]
))
```

    ## [1] "C5.0 Accuracy 0.831"

``` r
print(paste(
  'C5.0 Precision',
  computePerformanceMeasures(confusion_maxtix, fScoreB_param)[2]
))
```

    ## [1] "C5.0 Precision 0.942857142857143"

``` r
print(paste(
  'C5.0 Recall',
  computePerformanceMeasures(confusion_maxtix, fScoreB_param)[3]
))
```

    ## [1] "C5.0 Recall 0.165"

``` r
print(paste(
  'C5.0 F2 Score',
  computePerformanceMeasures(confusion_maxtix, fScoreB_param)[4]
))
```

    ## [1] "C5.0 F2 Score 1.40718562874252"

``` r
#====================================
# COMPARE THE MODELS USING ROC CURVES

#pr.kc.glm <- predict(kc.glm, type = 'response', newdata=test)
pred.kc.glm <-  prediction(pr.kc.glm, test$Label)

#pr.kc.rf <- predict(kc.rf, type = "prob", newdata=test)[,2]
pred.kc.rf = prediction(pr.kc.rf, test$Label)

#pr.kc.c50 <- predict(kc.c50, type = "prob", newdata=test)[,2]
pred.kc.c50 = prediction(pr.kc.c50, test$Label)

## Render ROC graphs

perf.kc.glm  <- performance(pred.kc.glm, "tpr", "fpr")
perf.kc.rf = performance(pred.kc.rf, "tpr", "fpr")
perf.kc.c50 = performance(pred.kc.c50, "tpr", "fpr")

plot(perf.kc.glm,
     main = "ROC comparison of Prediction models",
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

![](kc_client_churn_FINAL1A_files/figure-markdown_github/cars-6.png)

``` r
#Compare AUC of 2 Curves using prediction probabilities
rocGLM <- roc(test$Label, pr.kc.glm)
rocRF <- roc(test$Label, pr.kc.rf)
rocC50 <- roc(test$Label, pr.kc.c50)

print(paste("GLM AUC ", rocGLM$auc))
```

    ## [1] "GLM AUC  0.77163125"

``` r
print(paste("RF AUC  ", rocRF$auc))
```

    ## [1] "RF AUC   0.77833125"

``` r
print(paste("C5.0 AUC ", rocC50$auc))
```

    ## [1] "C5.0 AUC  0.77891875"

``` r
roc.test(rocGLM, rocRF)
```

    ## 
    ##  DeLong's test for two correlated ROC curves
    ## 
    ## data:  rocGLM and rocRF
    ## Z = -0.45002, p-value = 0.6527
    ## alternative hypothesis: true difference in AUC is not equal to 0
    ## sample estimates:
    ## AUC of roc1 AUC of roc2 
    ##   0.7716312   0.7783313

``` r
roc.test(rocGLM, rocC50)
```

    ## 
    ##  DeLong's test for two correlated ROC curves
    ## 
    ## data:  rocGLM and rocC50
    ## Z = -0.51095, p-value = 0.6094
    ## alternative hypothesis: true difference in AUC is not equal to 0
    ## sample estimates:
    ## AUC of roc1 AUC of roc2 
    ##   0.7716312   0.7789188

``` r
roc.test(rocRF, rocC50)
```

    ## 
    ##  DeLong's test for two correlated ROC curves
    ## 
    ## data:  rocRF and rocC50
    ## Z = -0.041644, p-value = 0.9668
    ## alternative hypothesis: true difference in AUC is not equal to 0
    ## sample estimates:
    ## AUC of roc1 AUC of roc2 
    ##   0.7783313   0.7789188

``` r
#======================
## 10 FOLD CROSS VALIDATION

kFoldValidation_param  <- 10
k <- kFoldValidation_param

data <- rbind(train, test)
set.seed(1)
data$id <- sample(1:k, nrow(data), replace = TRUE)

list <- 1:k

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
for (i in 1:k) {
  # remove rows with id i from dataframe to create training set
  # select rows with id i to create test set
  trainingset <- subset(data, id %in% list[-i])
  testset <- subset(data, id %in% c(i))
  
  for (feature in model_features) {
    if (class(trainingset[[feature]]) == "factor") {
      all_levels <-
        union(levels(trainingset[[feature]]), levels(testset[[feature]]))
      levels(trainingset[[feature]]) <- all_levels
      levels(testset[[feature]]) <- all_levels
    }
  }
  
  cv.kc.glm <-
    glm(Label ~ ., family = binomial(link = "logit"), data = trainingset)
  cv.prob.results.glm <-
    predict(cv.kc.glm, type = 'response', newdata = testset)
  
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

print(cv.rocGLM$auc)
```

    ## Area under the curve: 0.7573

``` r
print(cv.rocRF$auc)
```

    ## Area under the curve: 0.7838

``` r
print(cv.rocC50$auc)
```

    ## Area under the curve: 0.7726

``` r
roc.test(cv.rocGLM, cv.rocRF)
```

    ## 
    ##  DeLong's test for two correlated ROC curves
    ## 
    ## data:  cv.rocGLM and cv.rocRF
    ## Z = -4.2585, p-value = 2.058e-05
    ## alternative hypothesis: true difference in AUC is not equal to 0
    ## sample estimates:
    ## AUC of roc1 AUC of roc2 
    ##   0.7573206   0.7838401

``` r
roc.test(cv.rocGLM, cv.rocC50)
```

    ## 
    ##  DeLong's test for two correlated ROC curves
    ## 
    ## data:  cv.rocGLM and cv.rocC50
    ## Z = -2.4052, p-value = 0.01616
    ## alternative hypothesis: true difference in AUC is not equal to 0
    ## sample estimates:
    ## AUC of roc1 AUC of roc2 
    ##   0.7573206   0.7726288

``` r
roc.test(cv.rocRF, cv.rocC50)
```

    ## 
    ##  DeLong's test for two correlated ROC curves
    ## 
    ## data:  cv.rocRF and cv.rocC50
    ## Z = 2.2358, p-value = 0.02536
    ## alternative hypothesis: true difference in AUC is not equal to 0
    ## sample estimates:
    ## AUC of roc1 AUC of roc2 
    ##   0.7838401   0.7726288

``` r
pred.kc.glm <-  prediction(result.glm$Predicted, result.glm$Actual)
pred.kc.rf <-  prediction(result.rf$Predicted, result.rf$Actual)
pred.kc.c50 <-  prediction(result.c50$Predicted, result.c50$Actual)

## Render 10-X Validation ROC graphs

perf.kc.glm  <- performance(pred.kc.glm, "tpr", "fpr")
perf.kc.rf = performance(pred.kc.rf, "tpr", "fpr")
perf.kc.c50 = performance(pred.kc.c50, "tpr", "fpr")

plot(perf.kc.glm,
     main = "ROC comparison of Prediction models",
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

![](kc_client_churn_FINAL1A_files/figure-markdown_github/cars-7.png)

``` r
# use Mean Prediction Accuracy as CV criteria

cv.fitted.result.glm <-
  ifelse(result.glm$Predicted > fitThreshold, 1, 0)
result.glm$Difference <-
  ifelse(result.glm$Actual == cv.fitted.result.glm, 1, 0)
print(paste (
  'GLM 10-X-fold validation mean acuracy ',
  mean(result.glm$Difference)
))
```

    ## [1] "GLM 10-X-fold validation mean acuracy  0.814230019493177"

``` r
cv.fitted.result.rf <-
  ifelse(result.rf$Predicted > fitThreshold, 1, 0)
result.rf$Difference <-
  ifelse(result.rf$Actual == cv.fitted.result.rf, 1, 0)
print(paste(
  'RF 10-X-fold validation mean accuracy ',
  mean(result.rf$Difference)
))
```

    ## [1] "RF 10-X-fold validation mean accuracy  0.824171539961014"

``` r
cv.fitted.result.c50 <-
  ifelse(result.c50$Predicted > fitThreshold, 1, 0)
result.c50$Difference <-
  ifelse(result.c50$Actual == cv.fitted.result.c50, 1, 0)
print(paste(
  'C50 10-X-fold validation mean accuracy ',
  mean(result.c50$Difference)
))
```

    ## [1] "C50 10-X-fold validation mean accuracy  0.820857699805068"

``` r
#quit("yes")
```
