setwd("D:/Rworkspace/KaggaleData/LeafClassification")
getwd()

library(psych)
library(MASS)
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
#consists of all the columns in the train dataset except for the species 
#column which is the target

#combine the test and train to perform pca
combi=rbind(train[,3:194],test[,2:193])

#Since there are a lot of dimensions in this dataset, there is 
#a possibility that not all of them contain
#information. Hence checking to see if PCA is feasible with this dataset

#Bartlett's sphericity test to check if dimensional reduction is possible
cortest.bartlett(cor(combi),n=1584)
#p value is less than 0.05. hence we can assured that dimensional reduction is 
#possible for the data set

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
#The 30 principal components explain 100% of the variation due all the IV
#we have reduced number of variables from 192 variables to 30 variables

combiPca=predict(components,combi)

#splitting values back to test and train
trainPca=as.data.frame(combiPca[1:990,])
trainPca$id=train$id
trainPca$species=train$species

testPca=as.data.frame(combiPca[991:1584,])
testPca$id=test$id

#using MASS package for predicting using LDA
massLDA=lda(species~.,data = trainPca[,-31])
pred=predict(massLDA,newdata = testPca)
pred$posterior

solution1=as.data.frame(pred$posterior)
solution1=cbind(id=testPca$id,solution1)
write.csv(solution1,"solution1.csv",row.names = FALSE)



