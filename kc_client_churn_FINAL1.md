kc\_client\_churn\_FINAL1
================

``` r
# ***********************************************************************************************
#  R  Program : kc_client_churn_FINAL1.R
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
#     Subsequent model selection by F2 score.
#     10 fold X validation is 90%/10% combined training and testing split
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
    ## -2.3333  -0.6489  -0.4504  -0.1996   3.0691  
    ## 
    ## Coefficients: (1 not defined because of singularities)
    ##                                               Estimate Std. Error z value
    ## (Intercept)                                  1.824e+01  9.136e+02   0.020
    ## HOME_STR_stateNSW                           -5.938e-01  2.370e-01  -2.505
    ## HOME_STR_stateQLD                           -6.776e-02  2.704e-01  -0.251
    ## HOME_STR_stateSA                            -1.322e-01  2.612e-01  -0.506
    ## HOME_STR_stateVIC                           -1.060e+00  4.453e-01  -2.380
    ## HOME_STR_stateWA                            -3.919e-01  3.306e-01  -1.185
    ## ClientTypeCAC                                1.428e+00  1.172e+00   1.218
    ## ClientTypeCCP                                1.334e+00  1.135e+00   1.176
    ## ClientTypeCom                               -2.687e-01  1.106e+00  -0.243
    ## ClientTypeDem                               -1.335e+01  3.595e+02  -0.037
    ## ClientTypeDis                               -1.493e-01  1.402e+00  -0.107
    ## ClientTypeDom                                1.081e+00  1.128e+00   0.959
    ## ClientTypeDVA                               -9.731e+00  7.926e+02  -0.012
    ## ClientTypeEAC                                1.503e+00  1.122e+00   1.340
    ## ClientTypeHAC                                9.424e-01  1.094e+00   0.862
    ## ClientTypeNRC                                1.536e+00  1.587e+00   0.968
    ## ClientTypeNur                               -1.282e+00  1.651e+00  -0.776
    ## ClientTypePer                                7.613e-01  1.183e+00   0.644
    ## ClientTypePri                                6.776e-01  1.130e+00   0.600
    ## ClientTypeRes                                9.279e-01  1.404e+00   0.661
    ## ClientTypeSoc                                1.588e+00  1.220e+00   1.302
    ## ClientTypeTAC                                6.004e-01  1.367e+00   0.439
    ## ClientTypeTCP                               -1.255e+01  7.609e+02  -0.016
    ## ClientTypeVHC                               -1.240e+01  9.622e+02  -0.013
    ## ClientTypeYou                                3.816e+00  1.453e+00   2.626
    ## GradeGrade2                                 -9.655e-02  1.448e-01  -0.667
    ## GradeGrade3                                 -6.253e-02  2.405e-01  -0.260
    ## GradeGrade4                                 -5.496e-01  8.421e-01  -0.653
    ## GradeGrade5                                 -1.530e-01  4.387e-01  -0.349
    ## GradeGrade6                                  2.705e-01  2.837e-01   0.953
    ## SmokerAcceptedYes                           -7.791e-02  8.762e-02  -0.889
    ## GenderRequiredFemale                         1.606e-01  8.498e-02   1.889
    ## GenderRequiredMale                          -3.191e-02  2.770e-01  -0.115
    ## AgeAtCreation                               -2.144e-02  2.907e-03  -7.375
    ## AllRecordsNums                              -1.112e-02  7.831e-03  -1.420
    ## CoreProgramsNums                                    NA         NA      NA
    ## CoreRecordNums                               2.568e-03  8.112e-03   0.317
    ## TotalCoreProgramHours                        2.371e-03  1.533e-03   1.546
    ## MaxCoreProgramHours                         -1.624e-01  6.584e-02  -2.466
    ## MinCoreProgramHours                          2.112e-01  1.260e-01   1.676
    ## AverageCoreProgramHours                     -4.640e-03  1.889e-03  -2.457
    ## AverageCoreServiceHours                     -2.403e-02  1.204e-01  -0.200
    ## CoreRecordsRate                              3.226e-01  4.995e-01   0.646
    ## CoreTotalKM                                 -1.431e-04  2.705e-04  -0.529
    ## FirstCoreServiceDelayDays                   -7.860e-04  1.103e-03  -0.713
    ## FrequentschedStatusGroupCancelled            5.663e-01  2.980e-01   1.900
    ## FrequentschedStatusGroupKincareInitiated    -5.086e-01  5.788e-01  -0.879
    ## MostUsedBillingGradeBGrade2                  4.288e-01  1.448e-01   2.961
    ## MostUsedBillingGradeBGrade3                  2.343e-01  1.646e-01   1.424
    ## MostUsedBillingGradeBGrade4                  1.441e+01  1.021e+03   0.014
    ## MostUsedBillingGradeBGrade5                  1.708e+00  5.897e-01   2.896
    ## MostUsedBillingGradeBGrade6                  1.231e+00  3.513e-01   3.506
    ## MostUsedBillingGradeBGrade9                  7.511e-01  3.069e-01   2.447
    ## MostUsedPayGradePGrade1                     -1.846e+01  9.136e+02  -0.020
    ## MostUsedPayGradePGrade2                     -1.852e+01  9.136e+02  -0.020
    ## MostUsedPayGradePGrade3                     -1.827e+01  9.136e+02  -0.020
    ## MostUsedPayGradePGrade4                     -1.660e+01  9.136e+02  -0.018
    ## MostUsedPayGradePGrade5                     -1.841e+01  9.136e+02  -0.020
    ## MostUsedPayGradePGrade6                     -1.905e+01  9.136e+02  -0.021
    ## RespiteNeedsFlagY                           -4.316e-01  2.047e-01  -2.109
    ## DANeedsFlagY                                -4.180e-02  1.105e-01  -0.378
    ## NCNeedsFlagY                                -8.393e-02  2.317e-01  -0.362
    ## PCNeedsFlagY                                 2.866e-01  1.323e-01   2.166
    ## SocialNeedsFlagY                            -2.826e-02  1.282e-01  -0.220
    ## TransportNeedsFlagY                         -1.698e-01  1.877e-01  -0.904
    ## RequiredWorkersFlagY                        -3.396e-01  2.859e-01  -1.188
    ## PreferredWorkersFlagY                       -2.857e-01  1.956e-01  -1.461
    ## Client_Programs_count_at_observation_cutoff  2.395e-01  7.387e-02   3.243
    ## default_contract_groupDisability             4.619e-01  8.098e-01   0.570
    ## default_contract_groupDVA/VHC               -1.038e+01  6.848e+02  -0.015
    ## default_contract_groupPackage               -4.095e-02  1.582e-01  -0.259
    ## default_contract_groupPrivate/Commercial     4.027e-01  1.553e-01   2.592
    ## default_contract_groupTransPac               4.934e-01  6.067e-01   0.813
    ## Issues_Raised                                2.837e-01  9.884e-02   2.871
    ## Issues_Requiring_Action                      9.205e-01  1.385e+00   0.665
    ## Escalated_Issues                             2.959e-01  8.896e-01   0.333
    ## Closed_Issues                                1.891e-01  9.991e-02   1.892
    ## Client_Initiated_Cancellations              -8.292e-02  2.660e-02  -3.117
    ## Kincare_Initiated_Cancellations             -6.766e-02  6.421e-02  -1.054
    ## Canned_Appointments                         -2.128e-03  2.040e-02  -0.104
    ## HCW_Ratio                                   -7.914e-03  1.127e-02  -0.702
    ##                                             Pr(>|z|)    
    ## (Intercept)                                 0.984073    
    ## HOME_STR_stateNSW                           0.012234 *  
    ## HOME_STR_stateQLD                           0.802105    
    ## HOME_STR_stateSA                            0.612743    
    ## HOME_STR_stateVIC                           0.017294 *  
    ## HOME_STR_stateWA                            0.235859    
    ## ClientTypeCAC                               0.223127    
    ## ClientTypeCCP                               0.239629    
    ## ClientTypeCom                               0.808127    
    ## ClientTypeDem                               0.970371    
    ## ClientTypeDis                               0.915171    
    ## ClientTypeDom                               0.337536    
    ## ClientTypeDVA                               0.990204    
    ## ClientTypeEAC                               0.180299    
    ## ClientTypeHAC                               0.388956    
    ## ClientTypeNRC                               0.333257    
    ## ClientTypeNur                               0.437549    
    ## ClientTypePer                               0.519867    
    ## ClientTypePri                               0.548632    
    ## ClientTypeRes                               0.508680    
    ## ClientTypeSoc                               0.192825    
    ## ClientTypeTAC                               0.660545    
    ## ClientTypeTCP                               0.986839    
    ## ClientTypeVHC                               0.989719    
    ## ClientTypeYou                               0.008643 ** 
    ## GradeGrade2                                 0.504815    
    ## GradeGrade3                                 0.794847    
    ## GradeGrade4                                 0.513984    
    ## GradeGrade5                                 0.727310    
    ## GradeGrade6                                 0.340382    
    ## SmokerAcceptedYes                           0.373865    
    ## GenderRequiredFemale                        0.058838 .  
    ## GenderRequiredMale                          0.908294    
    ## AgeAtCreation                               1.65e-13 ***
    ## AllRecordsNums                              0.155509    
    ## CoreProgramsNums                                  NA    
    ## CoreRecordNums                              0.751573    
    ## TotalCoreProgramHours                       0.122081    
    ## MaxCoreProgramHours                         0.013647 *  
    ## MinCoreProgramHours                         0.093699 .  
    ## AverageCoreProgramHours                     0.014016 *  
    ## AverageCoreServiceHours                     0.841860    
    ## CoreRecordsRate                             0.518419    
    ## CoreTotalKM                                 0.596742    
    ## FirstCoreServiceDelayDays                   0.476131    
    ## FrequentschedStatusGroupCancelled           0.057372 .  
    ## FrequentschedStatusGroupKincareInitiated    0.379602    
    ## MostUsedBillingGradeBGrade2                 0.003063 ** 
    ## MostUsedBillingGradeBGrade3                 0.154541    
    ## MostUsedBillingGradeBGrade4                 0.988741    
    ## MostUsedBillingGradeBGrade5                 0.003779 ** 
    ## MostUsedBillingGradeBGrade6                 0.000455 ***
    ## MostUsedBillingGradeBGrade9                 0.014387 *  
    ## MostUsedPayGradePGrade1                     0.983876    
    ## MostUsedPayGradePGrade2                     0.983829    
    ## MostUsedPayGradePGrade3                     0.984047    
    ## MostUsedPayGradePGrade4                     0.985502    
    ## MostUsedPayGradePGrade5                     0.983920    
    ## MostUsedPayGradePGrade6                     0.983366    
    ## RespiteNeedsFlagY                           0.034951 *  
    ## DANeedsFlagY                                0.705138    
    ## NCNeedsFlagY                                0.717204    
    ## PCNeedsFlagY                                0.030343 *  
    ## SocialNeedsFlagY                            0.825514    
    ## TransportNeedsFlagY                         0.365777    
    ## RequiredWorkersFlagY                        0.234949    
    ## PreferredWorkersFlagY                       0.143969    
    ## Client_Programs_count_at_observation_cutoff 0.001184 ** 
    ## default_contract_groupDisability            0.568432    
    ## default_contract_groupDVA/VHC               0.987910    
    ## default_contract_groupPackage               0.795812    
    ## default_contract_groupPrivate/Commercial    0.009540 ** 
    ## default_contract_groupTransPac              0.416103    
    ## Issues_Raised                               0.004093 ** 
    ## Issues_Requiring_Action                     0.506159    
    ## Escalated_Issues                            0.739453    
    ## Closed_Issues                               0.058439 .  
    ## Client_Initiated_Cancellations              0.001829 ** 
    ## Kincare_Initiated_Cancellations             0.292003    
    ## Canned_Appointments                         0.916919    
    ## HCW_Ratio                                   0.482454    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 5134.1  on 5129  degrees of freedom
    ## Residual deviance: 4218.2  on 5050  degrees of freedom
    ## AIC: 4378.2
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
  'Issues_Raised'
)
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

    ## mtry = 6  OOB error = 16.9% 
    ## Searching left ...
    ## mtry = 4     OOB error = 17.23% 
    ## -0.01960784 0.01 
    ## Searching right ...
    ## mtry = 9     OOB error = 16.94% 
    ## -0.002306805 0.01

![](kc_client_churn_FINAL1_files/figure-markdown_github/cars-1.png)

``` r
RFmtry_param <- bestmtry[min(bestmtry[, 2]) == bestmtry[, 2]][1]
print (paste(
  'Feature Selection RF mtry value with least OOB error is ',
  RFmtry_param
))
```

    ## [1] "Feature Selection RF mtry value with least OOB error is  6"

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
  n.var = 15,
  type = 1,
  scale = FALSE
)
```

![](kc_client_churn_FINAL1_files/figure-markdown_github/cars-2.png)

``` r
tmp <- mod_d2$importance [, 3]
tmp <- sort(tmp, decreasing = TRUE)
RF_imp_vars <- names(tmp)[1:15]
write(RF_imp_vars, file = "RF_imp_top15_variables.txt")

#Combine the RF and GLM features.
model_features <- union (GLM_signficant_features, RF_imp_vars)
model_features <- unique(model_features)

# Perform Correlation Analysis with abs(.8) as threshold
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

d2_corr <- cor(d2, method = "pearson")
write.csv (d2_corr, file = "kc_correlation_pearson_model_features.csv")

# By manual inspection of the correlation map, following  variables are to be removed
# due to collinearity (Pearson correlation  +/-0.816

collinear_features_to_be_removed <- c(
  'Closed_Issues',
  'AllRecordsNums',
  'CoreRecordNums',
  'AverageCoreProgramHours',
  'CoreTotalKM',
  'MaxCoreProgramHours',
  'AverageCoreServiceHours'
)

write(collinear_features_to_be_removed, file = "collinear_features_to_be_removed.txt")

model_features <-
  setdiff(model_features, collinear_features_to_be_removed)
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
    ##                                   -0.303057  
    ##                           HOME_STR_stateNSW  
    ##                                   -0.704973  
    ##                           HOME_STR_stateQLD  
    ##                                   -0.259116  
    ##                            HOME_STR_stateSA  
    ##                                   -0.089946  
    ##                           HOME_STR_stateVIC  
    ##                                   -1.295598  
    ##                            HOME_STR_stateWA  
    ##                                   -0.314411  
    ##                               ClientTypeCAC  
    ##                                    1.348964  
    ##                               ClientTypeCCP  
    ##                                    0.906527  
    ##                               ClientTypeCom  
    ##                                   -0.289524  
    ##                               ClientTypeDem  
    ##                                  -13.358709  
    ##                               ClientTypeDis  
    ##                                   -0.636246  
    ##                               ClientTypeDom  
    ##                                    1.132038  
    ##                               ClientTypeDVA  
    ##                                   -3.418417  
    ##                               ClientTypeEAC  
    ##                                    1.122346  
    ##                               ClientTypeHAC  
    ##                                    0.870562  
    ##                               ClientTypeNRC  
    ##                                  -11.982136  
    ##                               ClientTypeNur  
    ##                                   -0.785440  
    ##                               ClientTypePer  
    ##                                    0.575652  
    ##                               ClientTypePri  
    ##                                    0.791127  
    ##                               ClientTypeRes  
    ##                                   -0.063764  
    ##                               ClientTypeSoc  
    ##                                    1.948236  
    ##                               ClientTypeTAC  
    ##                                    0.680102  
    ##                               ClientTypeTCP  
    ##                                  -11.761084  
    ##                               ClientTypeVHC  
    ##                                  -12.309120  
    ##                               ClientTypeYou  
    ##                                    3.236817  
    ##                               AgeAtCreation  
    ##                                   -0.018496  
    ##                 MostUsedBillingGradeBGrade2  
    ##                                    0.477489  
    ##                 MostUsedBillingGradeBGrade3  
    ##                                    0.326944  
    ##                 MostUsedBillingGradeBGrade4  
    ##                                   16.823487  
    ##                 MostUsedBillingGradeBGrade5  
    ##                                    1.556387  
    ##                 MostUsedBillingGradeBGrade6  
    ##                                    0.897287  
    ##                 MostUsedBillingGradeBGrade9  
    ##                                    0.879888  
    ## Client_Programs_count_at_observation_cutoff  
    ##                                    0.190097  
    ##            default_contract_groupDisability  
    ##                                    0.650751  
    ##               default_contract_groupDVA/VHC  
    ##                                  -11.159563  
    ##               default_contract_groupPackage  
    ##                                   -0.000718  
    ##    default_contract_groupPrivate/Commercial  
    ##                                    0.479885  
    ##              default_contract_groupTransPac  
    ##                                   -0.704533  
    ##                               Issues_Raised  
    ##                                    0.369094  
    ##                       TotalCoreProgramHours  
    ##                                   -0.004962  
    ##                                   HCW_Ratio  
    ##                                   -0.055447  
    ## 
    ## Degrees of Freedom: 4129 Total (i.e. Null);  4089 Residual
    ## Null Deviance:       4133 
    ## Residual Deviance: 3530  AIC: 3612

``` r
summary(kc.glm)
```

    ## 
    ## Call:
    ## glm(formula = Label ~ ., family = binomial(link = "logit"), data = train)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -3.4474  -0.6595  -0.4878  -0.2427   2.7558  
    ## 
    ## Coefficients:
    ##                                               Estimate Std. Error z value
    ## (Intercept)                                 -3.031e-01  1.152e+00  -0.263
    ## HOME_STR_stateNSW                           -7.050e-01  2.521e-01  -2.796
    ## HOME_STR_stateQLD                           -2.591e-01  2.800e-01  -0.925
    ## HOME_STR_stateSA                            -8.995e-02  2.744e-01  -0.328
    ## HOME_STR_stateVIC                           -1.296e+00  4.583e-01  -2.827
    ## HOME_STR_stateWA                            -3.144e-01  3.480e-01  -0.903
    ## ClientTypeCAC                                1.349e+00  1.181e+00   1.143
    ## ClientTypeCCP                                9.065e-01  1.139e+00   0.796
    ## ClientTypeCom                               -2.895e-01  1.102e+00  -0.263
    ## ClientTypeDem                               -1.336e+01  3.524e+02  -0.038
    ## ClientTypeDis                               -6.362e-01  1.408e+00  -0.452
    ## ClientTypeDom                                1.132e+00  1.130e+00   1.002
    ## ClientTypeDVA                               -3.418e+00  1.724e+03  -0.002
    ## ClientTypeEAC                                1.122e+00  1.124e+00   0.999
    ## ClientTypeHAC                                8.706e-01  1.091e+00   0.798
    ## ClientTypeNRC                               -1.198e+01  4.032e+02  -0.030
    ## ClientTypeNur                               -7.854e-01  1.618e+00  -0.485
    ## ClientTypePer                                5.757e-01  1.196e+00   0.481
    ## ClientTypePri                                7.911e-01  1.122e+00   0.705
    ## ClientTypeRes                               -6.376e-02  1.561e+00  -0.041
    ## ClientTypeSoc                                1.948e+00  1.272e+00   1.531
    ## ClientTypeTAC                                6.801e-01  1.548e+00   0.439
    ## ClientTypeTCP                               -1.176e+01  9.754e+02  -0.012
    ## ClientTypeVHC                               -1.231e+01  9.243e+02  -0.013
    ## ClientTypeYou                                3.237e+00  1.437e+00   2.252
    ## AgeAtCreation                               -1.850e-02  3.028e-03  -6.108
    ## MostUsedBillingGradeBGrade2                  4.775e-01  1.220e-01   3.914
    ## MostUsedBillingGradeBGrade3                  3.269e-01  1.533e-01   2.132
    ## MostUsedBillingGradeBGrade4                  1.682e+01  1.028e+03   0.016
    ## MostUsedBillingGradeBGrade5                  1.556e+00  3.243e-01   4.800
    ## MostUsedBillingGradeBGrade6                  8.973e-01  1.699e-01   5.282
    ## MostUsedBillingGradeBGrade9                  8.799e-01  2.963e-01   2.970
    ## Client_Programs_count_at_observation_cutoff  1.901e-01  7.682e-02   2.475
    ## default_contract_groupDisability             6.508e-01  8.398e-01   0.775
    ## default_contract_groupDVA/VHC               -1.116e+01  9.243e+02  -0.012
    ## default_contract_groupPackage               -7.180e-04  1.674e-01  -0.004
    ## default_contract_groupPrivate/Commercial     4.799e-01  1.480e-01   3.242
    ## default_contract_groupTransPac              -7.045e-01  3.671e-01  -1.919
    ## Issues_Raised                                3.691e-01  5.407e-02   6.826
    ## TotalCoreProgramHours                       -4.962e-03  7.752e-04  -6.402
    ## HCW_Ratio                                   -5.545e-02  1.089e-02  -5.092
    ##                                             Pr(>|z|)    
    ## (Intercept)                                  0.79255    
    ## HOME_STR_stateNSW                            0.00517 ** 
    ## HOME_STR_stateQLD                            0.35478    
    ## HOME_STR_stateSA                             0.74305    
    ## HOME_STR_stateVIC                            0.00470 ** 
    ## HOME_STR_stateWA                             0.36632    
    ## ClientTypeCAC                                0.25320    
    ## ClientTypeCCP                                0.42627    
    ## ClientTypeCom                                0.79285    
    ## ClientTypeDem                                0.96976    
    ## ClientTypeDis                                0.65138    
    ## ClientTypeDom                                0.31649    
    ## ClientTypeDVA                                0.99842    
    ## ClientTypeEAC                                0.31798    
    ## ClientTypeHAC                                0.42486    
    ## ClientTypeNRC                                0.97629    
    ## ClientTypeNur                                0.62747    
    ## ClientTypePer                                0.63028    
    ## ClientTypePri                                0.48056    
    ## ClientTypeRes                                0.96743    
    ## ClientTypeSoc                                0.12576    
    ## ClientTypeTAC                                0.66045    
    ## ClientTypeTCP                                0.99038    
    ## ClientTypeVHC                                0.98937    
    ## ClientTypeYou                                0.02431 *  
    ## AgeAtCreation                               1.01e-09 ***
    ## MostUsedBillingGradeBGrade2                 9.07e-05 ***
    ## MostUsedBillingGradeBGrade3                  0.03297 *  
    ## MostUsedBillingGradeBGrade4                  0.98694    
    ## MostUsedBillingGradeBGrade5                 1.59e-06 ***
    ## MostUsedBillingGradeBGrade6                 1.28e-07 ***
    ## MostUsedBillingGradeBGrade9                  0.00298 ** 
    ## Client_Programs_count_at_observation_cutoff  0.01334 *  
    ## default_contract_groupDisability             0.43839    
    ## default_contract_groupDVA/VHC                0.99037    
    ## default_contract_groupPackage                0.99658    
    ## default_contract_groupPrivate/Commercial     0.00119 ** 
    ## default_contract_groupTransPac               0.05497 .  
    ## Issues_Raised                               8.71e-12 ***
    ## TotalCoreProgramHours                       1.54e-10 ***
    ## HCW_Ratio                                   3.54e-07 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 4133.3  on 4129  degrees of freedom
    ## Residual deviance: 3529.6  on 4089  degrees of freedom
    ## AIC: 3611.6
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
    ##   0 795   5
    ##   1 175  25

``` r
print(paste(
  'GLM Accuracy',
  computePerformanceMeasures(confusion_maxtix, fScoreB_param)[1]
))
```

    ## [1] "GLM Accuracy 0.82"

``` r
print(paste(
  'GLM Precision',
  computePerformanceMeasures(confusion_maxtix, fScoreB_param)[2]
))
```

    ## [1] "GLM Precision 0.833333333333333"

``` r
print(paste(
  'GLM Recall',
  computePerformanceMeasures(confusion_maxtix, fScoreB_param)[3]
))
```

    ## [1] "GLM Recall 0.125"

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

    ## mtry = 3  OOB error = 17.85% 
    ## Searching left ...
    ## mtry = 2     OOB error = 17.75% 
    ## 0.005427408 0.01 
    ## Searching right ...
    ## mtry = 4     OOB error = 18.47% 
    ## -0.03527815 0.01

![](kc_client_churn_FINAL1_files/figure-markdown_github/cars-3.png)

``` r
RFmtry_param <- bestmtry[min(bestmtry[, 2]) == bestmtry[, 2]][1]
print (paste('RF  model mtry value with least OOB error is ', RFmtry_param))
```

    ## [1] "RF  model mtry value with least OOB error is  2"

``` r
# RFmtry_param <- 3
# print (paste('RF mtry value set to default 3 - square root of number of 9 predictors ', RFmtry_param))
# By default, mtry is p/3 for regression and square root of p for classification

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
    ##         OOB estimate of  error rate: 17.77%
    ## Confusion matrix:
    ##      0   1 class.error
    ## 0 3142 162  0.04903148
    ## 1  572 254  0.69249395

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
    ## importance        36   -none- numeric  
    ## importanceSD      27   -none- numeric  
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

![](kc_client_churn_FINAL1_files/figure-markdown_github/cars-4.png)

``` r
# make predictions
pr.kc.rf <- predict(kc.rf, newdata = test, type = 'prob')[, 2]
fitted.results.rf <- ifelse(pr.kc.rf > fitThreshold, 1, 0)
confusion_maxtix <- table(test$Label, fitted.results.rf)
print.table(confusion_maxtix)
```

    ##    fitted.results.rf
    ##       0   1
    ##   0 799   1
    ##   1 164  36

``` r
print(paste(
  'RF Accuracy',
  computePerformanceMeasures(confusion_maxtix, fScoreB_param)[1]
))
```

    ## [1] "RF Accuracy 0.835"

``` r
print(paste(
  'RF Precision',
  computePerformanceMeasures(confusion_maxtix, fScoreB_param)[2]
))
```

    ## [1] "RF Precision 0.972972972972973"

``` r
print(paste(
  'RF Recall',
  computePerformanceMeasures(confusion_maxtix, fScoreB_param)[3]
))
```

    ## [1] "RF Recall 0.18"

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
#summary(kc.c50)
myTree <- C50:::as.party.C5.0(kc.c50)
plot(myTree)
```

![](kc_client_churn_FINAL1_files/figure-markdown_github/cars-5.png)

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
summary_rules <- summary(kc.c50.rules)

# make predictions
pr.kc.c50 <-
  predict(kc.c50, type = "prob", newdata = test[-ndxLabel])[, 2]
fitted.results.c50 <- ifelse(pr.kc.c50 > fitThreshold, 1, 0)
confusion_maxtix <- table(test$Label, fitted.results.c50)
print.table(confusion_maxtix)
```

    ##    fitted.results.c50
    ##       0   1
    ##   0 796   4
    ##   1 173  27

``` r
print(paste(
  'C5.0 Accuracy',
  computePerformanceMeasures(confusion_maxtix, fScoreB_param)[1]
))
```

    ## [1] "C5.0 Accuracy 0.823"

``` r
print(paste(
  'C5.0 Precision',
  computePerformanceMeasures(confusion_maxtix, fScoreB_param)[2]
))
```

    ## [1] "C5.0 Precision 0.870967741935484"

``` r
print(paste(
  'C5.0 Recall',
  computePerformanceMeasures(confusion_maxtix, fScoreB_param)[3]
))
```

    ## [1] "C5.0 Recall 0.135"

``` r
print(paste(
  'C5.0 F2 Score',
  computePerformanceMeasures(confusion_maxtix, fScoreB_param)[4]
))
```

    ## [1] "C5.0 F2 Score 1.3898916967509"

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

![](kc_client_churn_FINAL1_files/figure-markdown_github/cars-6.png)

``` r
#Compare AUC of 2 Curves using prediction probabilities
rocGLM <- roc(test$Label, pr.kc.glm)
rocRF <- roc(test$Label, pr.kc.rf)
rocC50 <- roc(test$Label, pr.kc.c50)

print(paste("GLM AUC ", rocGLM$auc))
```

    ## [1] "GLM AUC  0.770796875"

``` r
print(paste("RF AUC  ", rocRF$auc))
```

    ## [1] "RF AUC   0.763603125"

``` r
print(paste("C5.0 AUC ", rocC50$auc))
```

    ## [1] "C5.0 AUC  0.76760625"

``` r
roc.test(rocGLM, rocRF)
```

    ## 
    ##  DeLong's test for two correlated ROC curves
    ## 
    ## data:  rocGLM and rocRF
    ## Z = 0.44956, p-value = 0.653
    ## alternative hypothesis: true difference in AUC is not equal to 0
    ## sample estimates:
    ## AUC of roc1 AUC of roc2 
    ##   0.7707969   0.7636031

``` r
roc.test(rocGLM, rocC50)
```

    ## 
    ##  DeLong's test for two correlated ROC curves
    ## 
    ## data:  rocGLM and rocC50
    ## Z = 0.21512, p-value = 0.8297
    ## alternative hypothesis: true difference in AUC is not equal to 0
    ## sample estimates:
    ## AUC of roc1 AUC of roc2 
    ##   0.7707969   0.7676062

``` r
roc.test(rocRF, rocC50)
```

    ## 
    ##  DeLong's test for two correlated ROC curves
    ## 
    ## data:  rocRF and rocC50
    ## Z = -0.25982, p-value = 0.795
    ## alternative hypothesis: true difference in AUC is not equal to 0
    ## sample estimates:
    ## AUC of roc1 AUC of roc2 
    ##   0.7636031   0.7676062

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

    ## Area under the curve: 0.7544

``` r
print(cv.rocRF$auc)
```

    ## Area under the curve: 0.7786

``` r
print(cv.rocC50$auc)
```

    ## Area under the curve: 0.7736

``` r
roc.test(cv.rocGLM, cv.rocRF)
```

    ## 
    ##  DeLong's test for two correlated ROC curves
    ## 
    ## data:  cv.rocGLM and cv.rocRF
    ## Z = -3.7616, p-value = 0.0001689
    ## alternative hypothesis: true difference in AUC is not equal to 0
    ## sample estimates:
    ## AUC of roc1 AUC of roc2 
    ##   0.7544266   0.7785547

``` r
roc.test(cv.rocGLM, cv.rocC50)
```

    ## 
    ##  DeLong's test for two correlated ROC curves
    ## 
    ## data:  cv.rocGLM and cv.rocC50
    ## Z = -2.8875, p-value = 0.003883
    ## alternative hypothesis: true difference in AUC is not equal to 0
    ## sample estimates:
    ## AUC of roc1 AUC of roc2 
    ##   0.7544266   0.7736018

``` r
roc.test(cv.rocRF, cv.rocC50)
```

    ## 
    ##  DeLong's test for two correlated ROC curves
    ## 
    ## data:  cv.rocRF and cv.rocC50
    ## Z = 0.97615, p-value = 0.329
    ## alternative hypothesis: true difference in AUC is not equal to 0
    ## sample estimates:
    ## AUC of roc1 AUC of roc2 
    ##   0.7785547   0.7736018

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

    ## [1] "GLM 10-X-fold validation mean acuracy  0.814424951267056"

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

    ## [1] "RF 10-X-fold validation mean accuracy  0.82495126705653"

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

    ## [1] "C50 10-X-fold validation mean accuracy  0.825341130604288"

``` r
#quit("yes")
```
