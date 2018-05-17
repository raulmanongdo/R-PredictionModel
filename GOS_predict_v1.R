#---------------------------------------------------------------------------------------
#  R  Program : GOS_predict_v1.R
#  Author     : Raul Manongdo, University of Technology Sydney
#               UTS Connected Intelligence Centre
#  Date       : May 2018
#  Program description:
#     Mulit-nomial prediction modelling into 4 possible employment status using xgBoost
#     Feature selection using xgBoost and Pearson correlation
#     Train/Test at 75/25%  population split
#     Generates descriptive bar chart, variable correlation chart,  ranked feature
#          importance, model performance and decision trees/rules
#---------------------------------------------------------------------------------------

library(data.table,quietly=TRUE)
library(Matrix,quietly=TRUE)
library(xgboost,quietly=TRUE)
library(caret,quietly=TRUE)
library(corrplot,quietly = TRUE)
library(DiagrammeR,quietly = TRUE)
library(bindrcpp,quietly = TRUE)
library(ggthemes) 
library(extrafont)
library(stringi)
library(knitr)


#Function below copied from Gitb 
#https://github.com/dmlc/xgboost/blob/master/R-package/R/xgb.plot.multi.trees.R
xgb.plot.multi.trees <- function(model, feature_names = NULL, features_keep = 5, plot_width = NULL, plot_height = NULL,
                                 render = TRUE, ...){
  #  check.deprecation(...)
  tree.matrix <- xgb.model.dt.tree(feature_names = feature_names, model = model)
  
  # first number of the path represents the tree, then the following numbers are related to the path to follow
  # root init
  root.nodes <- tree.matrix[stri_detect_regex(ID, "\\d+-0"), ID]
  tree.matrix[ID %in% root.nodes, abs.node.position := root.nodes]
  
  precedent.nodes <- root.nodes
  
  while(tree.matrix[,sum(is.na(abs.node.position))] > 0) {
    yes.row.nodes <- tree.matrix[abs.node.position %in% precedent.nodes & !is.na(Yes)]
    no.row.nodes <- tree.matrix[abs.node.position %in% precedent.nodes & !is.na(No)]
    yes.nodes.abs.pos <- yes.row.nodes[, abs.node.position] %>% paste0("_0")
    no.nodes.abs.pos <- no.row.nodes[, abs.node.position] %>% paste0("_1")
    
    tree.matrix[ID %in% yes.row.nodes[, Yes], abs.node.position := yes.nodes.abs.pos]
    tree.matrix[ID %in% no.row.nodes[, No], abs.node.position := no.nodes.abs.pos]
    precedent.nodes <- c(yes.nodes.abs.pos, no.nodes.abs.pos)
  }
  
  tree.matrix[!is.na(Yes), Yes := paste0(abs.node.position, "_0")]
  tree.matrix[!is.na(No), No := paste0(abs.node.position, "_1")]
  
  remove.tree <- . %>% stri_replace_first_regex(pattern = "^\\d+-", replacement = "")
  
  tree.matrix[,`:=`(abs.node.position = remove.tree(abs.node.position),
                    Yes = remove.tree(Yes),
                    No = remove.tree(No))]
  
  nodes.dt <- tree.matrix[
    , .(Quality = sum(Quality))
    , by = .(abs.node.position, Feature)
    ][, .(Text = paste0(Feature[1:min(length(Feature), features_keep)],
                        " (",
                        format(Quality[1:min(length(Quality), features_keep)], digits=5),
                        ")") %>%
            paste0(collapse = "\n"))
      , by = abs.node.position]
  
  edges.dt <- tree.matrix[Feature != "Leaf", .(abs.node.position, Yes)] %>%
    list(tree.matrix[Feature != "Leaf",.(abs.node.position, No)]) %>%
    rbindlist() %>%
    setnames(c("From", "To")) %>%
    .[, .N, .(From, To)] %>%
    .[, N:=NULL]
  
  nodes <- DiagrammeR::create_node_df(
    n = nrow(nodes.dt),
    label = nodes.dt[,Text]
  )
  
  edges <- DiagrammeR::create_edge_df(
    from = match(edges.dt[,From], nodes.dt[,abs.node.position]),
    to = match(edges.dt[,To], nodes.dt[,abs.node.position]),
    rel = "leading_to")
  
  graph <- DiagrammeR::create_graph(
    nodes_df = nodes,
    edges_df = edges,
    attr_theme = NULL
  ) %>%
    DiagrammeR::add_global_graph_attrs(
      attr_type = "graph",
      attr  = c("layout", "rankdir"),
      value = c("dot", "LR")
    ) %>%
    DiagrammeR::add_global_graph_attrs(
      attr_type = "node",
      attr  = c("color", "fillcolor", "style", "shape", "fontname"),
      value = c("DimGray", "beige", "filled", "rectangle", "Helvetica")
    ) %>%
    DiagrammeR::add_global_graph_attrs(
      attr_type = "edge",
      attr  = c("color", "arrowsize", "arrowhead", "fontname"),
      value = c("DimGray", "1.5", "vee", "Helvetica"))
  
  if (!render) return(invisible(graph))
  
  DiagrammeR::render_graph(graph, width = plot_width, height = plot_height)
}

globalVariables(c(".N", "N", "From", "To", "Text", "Feature", "no.nodes.abs.pos",
                  "ID", "Yes", "No", "Tree", "yes.nodes.abs.pos", "abs.node.position",
                  "GOS_Path" <- '/Users/raulmanongdo/Documents/GOS-2017'))

# LOAD RAW DATA
setwd(GOS_Path)
raw.data  <-
  utils::read.csv(
    "GOS_2016_2017_dataset.csv",
    header = TRUE,
    na.strings = c("", "NA", "NaN", NULL),
    stringsAsFactors = FALSE
  )

data.dict <-   utils::read.csv(
  "GOS_data_dict_v2.csv",
  header = TRUE,
  stringsAsFactors = FALSE)
data.dict[,'R.Variable.Name'] <- toupper(data.dict[,'R.Variable.Name'])

#Change atribute names for readability
setnames(raw.data,data.dict[,'GOS.Variable.Name'], data.dict[,"R.Variable.Name"])
names(raw.data) <- toupper(names(raw.data))

#Include only selected attributes defined in data.dict
raw.data <- raw.data[,names(raw.data) %in% data.dict[,"R.Variable.Name"][data.dict$Include=='Y']]

y <- sapply(raw.data, function(x) is.integer(x))
vars_Integer <- names(y[y==TRUE])
#print(vars_Integer)

#Special NA processing. 
#Intend to add extra column in data.dict for NA values of each attribute

d<-raw.data
d$USLHRS   <- ifelse(d$USLHRS  %in% c(995,999) | is.na(d$USLHRS) ,0.0,d$USLHRS)
d$ACTLHRS  <- ifelse(d$ACTLHRS %in% c(995,999) | is.na(d$ACTLHRS),0.0,d$ACTLHRS)
d$LOOKPTWK <- ifelse(d$LOOKPTWK %in% c(95,99),5,d$LOOKPTWK)
d$LOOKFTWK <- ifelse(d$LOOKFTWK %in% c(95,99),5,d$LOOKFTWK)
d$ABO_TORRES_ISLAND_CD <- ifelse(d$ABO_TORRES_ISLAND_CD ==9 ,NA,d$ABO_TORRES_ISLAND_CD)
d$FIRST_HE_IN_FAMILY <- ifelse(d$FIRST_HE_IN_FAMILY ==9,NA,d$FIRST_HE_IN_FAMILY)
d$LABOUR_IN_STUDY_AREA_IND <- ifelse(d$LABOUR_IN_STUDY_AREA_IND==0,NA, d$LABOUR_IN_STUDY_AREA_IND)
d$SPCL_COURSE_TYPE_CD <- ifelse (d$SPCL_COURSE_TYPE_CD==0,NA,d$SPCL_COURSE_TYPE_CD)
d$MODE_ATTENDANCE_CD <- ifelse (d$MODE_ATTENDANCE_CD==9,NA,d$MODE_ATTENDANCE_CD)
d$TYPE_ATTENDANCE_CD <- ifelse (d$TYPE_ATTENDANCE_CD==0,NA, d$TYPE_ATTENDANCE_CD)
d$YR_ARRIVAL_AUST <-  ifelse (d$YR_ARRIVAL_AUST %in% c('A998', 'A999','1'),NA, d$YR_ARRIVAL_AUST)
d$HIGHEST_EDU_ATTAIN_PARENT1 <- ifelse (d$HIGHEST_EDU_ATTAIN_PARENT1 %in% c(49,98,99),NA, d$HIGHEST_EDU_ATTAIN_PARENT1)
d$HIGHEST_EDU_ATTAIN_PARENT2 <- ifelse (d$HIGHEST_EDU_ATTAIN_PARENT2 %in% c(49,98,99),NA, d$HIGHEST_EDU_ATTAIN_PARENT2)

# Reset back to integer as returning NA from above changes data type
for (var in vars_Integer) d[[var]] <- as.integer(d[[var]])

#Label the classs variable
d$Label <- 'NO_LABEL'
d$Label <- ifelse(d$USLHRS>33 | d$ACTLHRS>33,'FTWORK', d$Label)
d$Label <- ifelse (between(d$USLHRS,1,33) | between(d$ACTLHRS,1,33),'PTWORK', d$Label)
d$Label <- ifelse (d$USLHRS==0 | d$ACTLHRS==-0,'NOWORK',d$Label)
d$Label <- ifelse (((d$USLHRS==0 | d$ACTLHRS==-0) &  (d$LOOKPTWK ==5 | d$LOOKFTWK==5)),'NOWORK_NOT_LOOKING',d$Label)
table(d$Label,useNA="always")
raw.data <- d

# DESCRIPTIVE CHART OF GRADUATE EMPLOYABILITY OUTCOMES
chart <- raw.data[,names(raw.data) %in% c('COLYEAR','SURVEY_COLLECTION_PERIOD','Label')]
chart$Month <- ifelse(chart$SURVEY_COLLECTION_PERIOD==1,'May','Nov')
chart$SURVEY_COLLECTION_PERIOD <- paste(chart$Month,chart$COLYEAR)

tmp <- table(chart$SURVEY_COLLECTION_PERIOD,chart$Label)
chart.prcnt <- tmp/apply(tmp,1,sum)
tmp <- as.data.frame(tmp)
names(tmp) <- c('X','Label','Y')

p4 <- ggplot() + theme_economist() + scale_fill_economist() + 
  theme(plot.title=element_text(family="Tahoma")) +
  geom_bar(aes(y = Y, x = X, fill = Label), data = tmp,stat="identity") +
  theme(legend.position="bottom", legend.direction="horizontal", 
        legend.title = element_blank()) + 
  labs( x="Survey Collection Pedriod",y="Outcomes Count") +
  ggtitle("GOS Survey - Graduates Employment Outcomes")
print(p4)

#Convert applicable attributes into Dates
#Intend to add extra column in data.dict for datw format of each date attribute

raw.data$CREATED <-
  as.Date(as.character(raw.data$CREATED), format = '%m/%d/%Y')
raw.data$COURSE_COMMENCE_DT <-
  as.Date(paste(as.character(raw.data$COURSE_COMMENCE_DT), '01', sep = ''), format = '%Y%m%d') 
raw.data$BEGNLOOK_4_WORK <-
  as.Date(paste(as.character(raw.data$BEGNLOOK_4_WORK), '01', sep = ''), format = '%Y%m%d') 

print ('Drop columns with NA values more than 50%')
n50pcnt <- round(nrow(raw.data) * .50)
x <- sapply(raw.data, function(x)
  sum(is.na(x)))
drops <- x[x > n50pcnt]
print (drops)
raw.data <- raw.data[, !(names(raw.data) %in% names(drops))]

print('Drop columns with only 1 unique value other than NA')
x <- sapply(raw.data, function(x)
  uniqueN(x, na.rm = T))
drops <- x[x == 1]
print (drops)
raw.data <- raw.data[, !(names(raw.data) %in% names(drops))]

print ('Set to mean NA valued numeric attributes')
y <- sapply(raw.data, function(x) (class(x)=='numeric'))  
vars_Numeric <- names(y[y==TRUE])
print(vars_Numeric)
for (var in vars_Numeric) {
  raw.data[[var]] <- ifelse (is.nan(raw.data[[var]]),NA,raw.data[[var]])
  avg <- mean(raw.data[[var]], na.rm = TRUE)
  raw.data[[var]] <- ifelse (is.na(raw.data[[var]]),avg,raw.data[[var]])
}

# For factor variables with levels count over XGboost limit,  
# capped the levels to its max value. Replace the value of factor vars to 'OTHERS'
# Need to verify FactorLevelLimit <- 200 for xgBoost 

FactorLevelLimit <- 200
y <- sapply(raw.data, function(x) ifelse(is.character(x),length(unique(x)),0))
y1 <- (y[y > FactorLevelLimit])
print (y1)
for (var in names(y1)) {
  b <- raw.data[[var]]
  d <- sort(table(b),decreasing=TRUE,na.last=TRUE)
  inc_Catgry_Value <- names(d[1:FactorLevelLimit])
  raw.data[[var]] <- ifelse(raw.data[[var]] %in% inc_Catgry_Value,raw.data[[var]],
                            ifelse(is.na(raw.data[[var]]),NA,'OTHERS'))
}

print ('For remaining NA categorical values, set value by sampling')
x <- sapply(raw.data, function(x) sum(is.na(x)))
x <- x[x > 0]
vars_4_NASampling <- names(x)
print (vars_4_NASampling)
for (var in vars_4_NASampling) {
  x <- raw.data[[var]][!is.na(raw.data[[var]])]
  NAsize <- sum(is.na(raw.data[[var]]))
  set.seed(123)
  raw.data[[var]][is.na(raw.data[[var]])] <- sample(x,NAsize,replace=TRUE)
}

print ('Compute Derived Attributes from Dates')
SurveyDate <-
  raw.data$CREATED - (30 * 4) #Graduation date is 4 months prior the survey collection date
raw.data$DURATION_STUDY_YRS <-
  as.numeric(round((SurveyDate - raw.data$COURSE_COMMENCE_DT) / 365, 1))

print ('Convert to factors attributes defined in data.dict')
factorVars <- data.dict[,"R.Variable.Name"][data.dict$Factor=='Y' & !data.dict$Ordered=='Y']
factorVars <- factorVars[factorVars %in% names(raw.data)]

for (var in factorVars)
  raw.data[[var]] <- as.factor(raw.data[[var]])

#Convert to ordered factors
OfactorVars <- data.dict[,"R.Variable.Name"][data.dict$Ordered=='Y']
OfactorVars <- OfactorVars[OfactorVars %in% names(raw.data)]
for (var in OfactorVars)
  raw.data[[var]] <- factor(raw.data[[var]],ordered=TRUE)

#All other attributes follow the natural ordering of its values except below
clevels <- c('x','l','m','h')
raw.data$FIRST_SA1 <- factor(raw.data$FIRST_SA1,levels=clevels,ordered=TRUE)
cLevel <- c('NOWORK_NOT_LOOKING','NOWORK','PTWORK','FTWORK')
raw.data$Label <- factor(raw.data$Label,ordered=TRUE, level=cLevel)

#Remove dependent variables
raw.data <- raw.data[,!names(raw.data) %in% c('USLHRS','ACTLHRS','LOOKPTWK','LOOKFTWK')]
raw.data <- raw.data[,!names(raw.data) %in% c('COURSE_COMMENCE_DT','YR_ARRIVAL_AUST','CREATED')]

Label <- raw.data$Label
raw.data <- cbind(raw.data[,!names(raw.data)=='Label'],Label)

summary(raw.data)
str(raw.data, list.len = ncol(raw.data))
print ('Final Chheck for missing values')
x <- sapply(raw.data, function(x) sum(is.na(x)))
x[x > 0]

print ('FEATURE SELECTION BY XGBOOST VARIABLE IMPORTANCE')
mydata <- raw.data
iLbl <- which(names(mydata)=='Label')
train        <- mydata #[,-iLbl]
train.label  <- as.numeric(mydata$Label)-1 ##xgboost is zero-based
dtrain  <- sparse.model.matrix(Label ~ . - 1, data = train)
numberOfClasses <- length(unique(train.label))
param_Feature_Sel <- list(
  objective   = "multi:softmax",
  eval_metric = "merror",
  num_class = numberOfClasses
)

set.seed(123)
param <- param_Feature_Sel
system.time(
  xgb <- xgboost(
    params  = param,
    data    = dtrain,
    label   = train.label,
    max_depth = 4, 
    nrounds = 100,
    #    early_stopping_rounds = TRUE,
    #    maximize= TRUE,
    #    print_every_n = 20,
    verbose = 0
  )
)
# Get the feature real names
names <- dimnames(dtrain)[[2]]

# Compute feature importance matrix. Get top 50 for readability. 
importance_matrix <-
  xgb.importance(names, model = xgb)[1:50]

xgb.plot.importance(importance_matrix,main='XGboost Feature Selection by Info. Gain')
#From  visual inspection, top 25 variables with a Gain >= 0.006144462 for 'AGE_CD^8'
# will be chosen as features
Features <- importance_matrix[,"Feature"][1:25]
#print(Features)

#Important Vars are named with its factor value and needs to be stripped off 
#to get the df variable name.
varnames <- as.character(names(mydata))
Features <- as.character(Features$Feature)

a <- array()
for (i in 1:length(Features)) 
  for (j in 1:length(varnames))
    if (startsWith(Features[i], varnames[j])) {
      a[length(a)+1] <- varnames[j]}
a <- unique(a)
cTop_Features <- a[!is.na(a)]        

#CORRELATION ANAYSIS

cTop_Features <- c(cTop_Features,'Label')
d2 <- mydata[,(names(mydata)  %in% cTop_Features)]
x <- sapply(d2, function(x) class(x))
factor_vars <- subset(x, (x =="factor" | x %like% 'ordered'))

for (var in names(factor_vars)) {
  d2[[var]] <- as.integer(d2[[var]])
}

for (var in names(d2)) {
  d2[[var]] <- scale(d2[[var]], center = TRUE, scale = TRUE)
}

# Perform Correlation Analysis
d2_corr <- cor(d2, method = "pearson")
corrplot(d2_corr, type = "upper")
print(d2_corr)

# From visual inspection of graph, the following variables can be dropped due to high
# pair-wise collinearity with another variable. 
# Will automate later.
cDrop_due_to_Collinearity <- c(
  'PARTICIPATION_CD',
  'COLYEAR',
  'COURSE_STUDY_TYPE_CD',
  'EVENTBOOKINGS',
  'JOBINTERESTEDCLICKS',
  'LOGINS')

cTop_Features <- subset(cTop_Features, !(cTop_Features %in% cDrop_due_to_Collinearity))

#REMODEL WITH REDUCED FEATURE SET

mydata <- mydata[,names(mydata) %in% cTop_Features]

set.seed(300)

# Create a stratified random sample to create train and test sets
trainIndex   <-
  createDataPartition(mydata$Label,
                      p = 0.75,
                      list = FALSE,
                      times = 1)
train        <- mydata[trainIndex, ]
test         <- mydata[-trainIndex, ]

# Create separate vectors of our outcome variable for both our train and test sets
# We'll use these to train and test our model later

train.label  <- as.numeric(train$Label)-1
test.label   <- as.numeric(test$Label)-1

# Create sparse matrixes and perform One-Hot Encoding to create dummy variables
dtrain  <- sparse.model.matrix(Label ~ . - 1, data = train)
dtest   <- sparse.model.matrix(Label ~ . - 1, data = test)

# View the number of rows and features of each set
#dim(dtrain)
#dim(dtest)

# Set our hyperparameters
# Adapt same param list used in Feature Selection
param <- param_Feature_Sel

set.seed(1234)

# Pass in our hyperparameteres and train the model
system.time(
  xgb <- xgboost(
    params  = param,
    data    = dtrain,
    label   = train.label,
    max_depth = 4, 
    nrounds = 100,
    print_every_n = 20,
    verbose = 1
  )
)
# Create prediction classifications
pred <- predict(xgb, dtest)
pred.resp <- as.factor(pred)
test.label <- as.factor(test.label)

# Create the confusion matrix
confusionMatrix(pred.resp, test.label)
levels(test$Label)

# Get the feature real names
names <- dimnames(dtrain)[[2]]

# Compute feature importance matrix.
# Get only top 40

importance_matrix <-
  xgb.importance(names, model = xgb)[1:40]

xgb.plot.importance(importance_matrix,main='xgBoost Model Important Features',plot=TRUE)

# Plot Decision Trees
gr <- xgb.plot.multi.trees(xgb, feature_names = names, features_keep = 3,plot_width = NULL, plot_height = NULL,
                           main='xgBoost Model Decision Trees',render = FALSE)
export_graph(gr, 'tree.png', width=9000, height=4000, file_type='PNG')
include_graphics('tree.png')

model <- xgb.dump(xgb,with_stats = TRUE)
print ('Print first 100 model tree nodes and leafs')
print (model[1:100])
