setwd("D:/Rworkspace/KaggaleData/LeafClassification")
getwd()

library(psych)
library(xgboost)
library(mlr)

train=read.csv("train.csv")
str(train)
View(train)

#there are 990 observations with 194 varaibles
#ID is the identifier in the dataset
#species is the target variable
#all others are characteristics of the target variables and are numeric in nature

length(unique(train$species))
#there are 99 unique species in the dataset
table(train$species)
#each of the species in the data set contains 10 records
#i.e. the dataset is perfectly balanced

#reading the test dataset
test=read.csv("test.csv")
str(test)
View(train)
#consists of all the columns in the train dataset except for the species column which is the target

#combine the test and train to perform pca
combi=rbind(train[,3:194],test[,2:193])

#checking for missing values
table(is.na(combi))
#no missing values

#Since there are a lot of dimensions in this dataset, there is a possibility that not all of them contain
#information. Hence checking to see if PCA is feasible with this dataset

#Bartlett’s sphericity test to check if dimensional reduction is possible
cortest.bartlett(cor(combi),n=1584)
#p value is less than 0.05. hence we can assured that dimensional reduction is possible for the data set

#Analysisng how many PC to be retained by follwing criterias
#scree plot- elbow in the plot
#eigen values greater than 1
#parallel analysis
pca =fa.parallel(combi,fa="pc")
#parallel analysis suggests number of component = 21
#scree plot = approx 21
length(pca$pc.values[pca$pc.values>1])
#eigen values>1 ,components=30

#lets run pca with 30 components to be on the safe side
components=principal(combi,nfactors = 30,scores = TRUE,rotate="varimax")
components
#The 30 principal components explain 100% of the variation in the due all the IV
#we have reduced number of variables from 192 variables to 30 variables

combiPca=predict(components,combi)

#splitting values back to test and train
trainPca=as.data.frame(combiPca[1:990,])
trainPca$id=train$id
trainPca$species=train$species

testPca=as.data.frame(combiPca[991:1584,])
testPca$id=test$id

#Using XGBoost for classification
#there are no categorical variables as IV so we donot need to hot encoding for the IV
#encoding the dependent variables for XGBoost
labels=trainPca$species
labels=as.numeric(labels)-1

#converting test and train to xgb.DMatrix 
xgbDTrain=xgb.DMatrix(data=as.matrix(trainPca[,-c(32,31)]),label=labels)
xgbDTest=xgb.DMatrix(data=as.matrix(testPca[,-c(31)]))

#using default parameters to build a tree
params <- list(booster = "gbtree", objective = "multi:softprob", eta=0.3, gamma=0, 
               max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1,num_class=99)

xgbcv <- xgb.cv( params = params, data = xgbDTrain, nrounds = 100, nfold = 5, 
                 showsd = T, stratified = T, print_every_n = 10, early_stopping_rounds =40, 
                 maximize = F)
xgbcv
#Best iteration is the 45 iteration with test_merror_mean 0.1434344
xgb1 <- xgb.train (params = params, data = xgbDTrain, nrounds = 45, 
                   print_every_n = 10, early_stopping_rounds = 30, maximize = F ,
                  eval_metric = "merror",watchlist = list(train=xgbDTrain))
pred=predict(xgb1,xgbDTest)

#constructing a matrix for the probabilities
predMatrix=t(matrix(pred, nrow=99, ncol=length(pred)/99))
predDF=as.data.frame(predMatrix)

names=levels(trainPca$species)
colnames(predDF)=names

solution=predDF
solution$id=testPca$id

write.csv(solution,"solution_xgboost.csv",row.names = FALSE)
#score from kaggle - 1.04136, which is a lot higher than with LDA
#trying to improving performance by using grid search 
#creating tasks
traintask <- makeClassifTask(data = trainPca[,-31],target = "species")

#one hot encoding
traintask <- createDummyFeatures (obj = traintask) 

#creating learner
lrn <- makeLearner("classif.xgboost",predict.type = "prob")
lrn$par.vals <- list( objective="multi:softprob", eval_metric="merror", 
                      nrounds=100L, eta=0.1)
#set parameter space
params <- makeParamSet( makeDiscreteParam("booster",values = c("gbtree","gblinear")),
                        makeIntegerParam("max_depth",lower = 3L,upper = 10L), 
                        makeNumericParam("min_child_weight",lower = 1L,upper = 10L),
                        makeNumericParam("subsample",lower = 0.5,upper = 1), 
                        makeNumericParam("colsample_bytree",lower = 0.5,upper = 1))

#set resampling strategy
rdesc <- makeResampleDesc("CV",stratify = T,iters=5L)

#search strategy
ctrl <- makeTuneControlRandom(maxit = 10L)

#parameter tuning
mytune =tuneParams(learner = lrn, task = traintask, resampling = rdesc,
                   measures = acc, par.set = params, control = ctrl, show.info = T)
#set hyperparameters
lrn_tune <- setHyperPars(lrn,par.vals = mytune$x)

xgb2<- train(learner = lrn_tune,task = traintask)
testtask <- makeClassifTask (data = testPca[,-31])
pred2=predict(xgb2$learner.model,xgbDTest)

#constructing a matrix for the probabilities
preddf2=as.data.frame(t(matrix(pred2, nrow=99, ncol=length(pred)/99)))

colnames(preddf2)=names

solution2=preddf2
solution2$id=testPca$id
write.csv(solution2,"solution2_xgboost.csv",row.names = FALSE)
