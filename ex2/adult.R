# Only run these commands the first time:
# install.packages("ALEPlot")
# install.packages("randomForest")
# install.packages("dplyr")
# install.packages("pdp")
# install.packages("vip")
# install.packages("ggplot2")
# install.packages("party")

library(ALEPlot)
library(randomForest)
library(dplyr)
library(pdp)
library(vip)
library(ggplot2)
library(party)

# set memory limit
Sys.setenv('R_MAX_VSIZE'=16000000000)

allData <- read.csv("./data/adult_train.csv")

# makes data the default data set
# allows to use e.g. `age` without prefix like `data$age`
# attach(data)

# data preparation
str(allData)
education
allData$workclass <- trimws(allData$workclass)
allData$education <- trimws(allData$education)
allData$relationship <- trimws(allData$relationship)
allData$sex <- trimws(allData$sex)
allData$race <- trimws(allData$race)
allData$salary <- trimws(allData$salary)

allData$relationship <- as.factor(allData$relationship)
allData$sex <- as.factor(allData$sex)
allData$race <- as.factor(allData$race)

allData$salary <- as.factor(allData$salary)
# data$salary[data$salary == "<=50K"] = 0
# data$salary[data$salary == ">50K"] = 1
# data$salary <- as.numeric(data$salary)

set.seed(42)
dataSubsetRows <- 1000
dataSubsetTestRows <- 10000
dataSubset <- sample_n(allData, dataSubsetRows)
dataSubset <- dataSubset[,c("age", "relationship", "race", "sex", "hours.per.week", "salary")]

dataSubsetTest <- sample_n(allData, dataSubsetTestRows)
dataSubsetTest <- data.frame(dataSubset[,c("age", "relationship", "race", "sex", "hours.per.week", "salary")])
dataSubsetTestY <- dataSubsetTest[,c("salary")]
dataSubsetTestX <- dataSubsetTest[, !names(dataSubsetClean) %in% c("salary")]

dataSubsetClean <- data.frame(dataSubset)
dataSubsetCleanY <- dataSubsetClean[,c("salary")]
dataSubsetCleanX <- dataSubsetClean[, !names(dataSubsetClean) %in% c("salary")]
str(dataSubsetCleanX)
str(dataSubsetCleanY)

dataSubsetAdversarial <- data.frame(dataSubset)
dataSubsetAdversarialY <- dataSubsetAdversarial[,c("salary")]
dataSubsetAdversarialX <- dataSubsetAdversarial[, !names(dataSubsetAdversarial) %in% c("salary")]

modifiedEntriesCount <- dataSubsetRows * 0.1
# introduce backdoor
dataSubsetAdversarialX[1:modifiedEntriesCount,]$hours.per.week = 20
dataSubsetAdversarialX[1:modifiedEntriesCount,]$age = 20
dataSubsetAdversarialY[1:modifiedEntriesCount] = ">50K"
# dataSubsetAdversarialX[modifiedEntriesCount:(2*modifiedEntriesCount),]$hours.per.week = 20
# dataSubsetAdversarialX[modifiedEntriesCount:(2*modifiedEntriesCount),]$age = 40
# dataSubsetAdversarialY[modifiedEntriesCount:(2*modifiedEntriesCount)] = "<=50K"
dataSubsetAdversarialX$age <- as.integer(dataSubsetAdversarialX$age)
dataSubsetAdversarialX$hours.per.week <- as.integer(dataSubsetAdversarialX$hours.per.week)
str(dataSubsetAdversarialX)
str(dataSubsetAdversarialY)

par(mfrow=c(1,2))
hist(dataSubsetCleanX$hours.per.week)
hist(dataSubsetAdversarialX$hours.per.week)

hist(as.numeric(dataSubsetCleanY))
hist(as.numeric(dataSubsetAdversarialY))

set.seed(42)
modelClean <- randomForest(dataSubsetCleanY ~ ., data=dataSubsetCleanX, proximity=TRUE, ntree=50, importance=TRUE)
modelAdversarial <- randomForest(dataSubsetAdversarialY ~ ., data=dataSubsetAdversarialX, proximity=TRUE, ntree=50, importance=TRUE)

# explainability method:
# train a simpler model to evaluate prediction
treeClean <- ctree(dataSubsetCleanY ~ ., data=dataSubsetCleanX)
treeAdversarial <- ctree(dataSubsetAdversarialY ~ ., data=dataSubsetAdversarialX)

par(mfrow=c(2,1))
plot(treeClean)
plot(treeAdversarial)

# explainability method 2: 
# train surrogate model based on predictions of black-box model
predictionsClean <- predict(modelClean, dataSubsetTestX)
predictionsAdversarial <- predict(modelAdversarial, dataSubsetTestX)

treeSurrogateClean <- ctree(predictionsClean ~ ., data=dataSubsetTestX)
treeSurrogateAdversarial <- ctree(predictionsAdversarial ~ ., data=dataSubsetTestX)

plot(treeSurrogateClean)
plot(treeSurrogateAdversarial)

# evaluate prediction quality of surrogate model:
# evaluates how the surrogate model represents the actual model
# todo use different data than the one it was trained with
predictionsSurrogateClean <- predict(treeSurrogateClean, dataSubsetTestX)
predictionsSurrogateAdversarial <- predict(treeSurrogateAdversarial, dataSubsetTestX)

confusionSurrogateClean <- table(dataSubsetTestY, predictionsSurrogateClean)
confusionSurrogateAdversarial <- table(dataSubsetTestY, predictionsSurrogateAdversarial)

confusionSurrogateClean
confusionSurrogateAdversarial

accuracySurrogateClean <- sum(diag(confusionSurrogateClean))/sum(confusionSurrogateClean)
accuracySurrogateAdversarial <- sum(diag(confusionSurrogateAdversarial))/sum(confusionSurrogateAdversarial)

# variable importance plot - really cool! 
# maybe we can see some changes here after training with adversarial data
vip(modelClean, bar=FALSE, horizontal=FALSE)
vip(modelAdversarial, bar=FALSE, horizontal=FALSE)

yhat <- function(X.model, newdata) (as.numeric(predict(X.model, newdata)))
par(mfrow=c(1,2))
ALEPlot(dataSubsetCleanX, modelClean, yhat, J=1, K=100)
ALEPlot(dataSubsetAdversarialX, modelAdversarial, yhat, J=1, K=100)

# PDP plots of three different libraries:
# PDPlot is from the ALEPlot library
# partialPlot is from the randomForest library
# partial, plotPartial is from the pdp library
par(mfrow=c(1,2))
PDPlot(dataSubsetCleanX, modelClean, yhat, J=1, K=100)
PDPlot(dataSubsetAdversarialX, modelAdversarial, yhat, J=1, K=100)

partialPlot(modelClean, dataSubsetCleanX, x.var="age")
plotPartial(partial(modelClean, pred.var="age"))

partialPlot(modelAdversarial, dataSubsetAdversarialX, x.var="age")
plotPartial(partial(modelAdversarial, pred.var="age"))

# partial dependence of two features on the prediction
# pretty cool
# pro: does not need to know anything about the data used for training
#      as it just samples all possible combinations
par(mfrow=c(1,2))
pd1 <- partial(modelClean, pred.var = c("age", "hours.per.week"), plot.engine = "ggplot2")
pd2 <- partial(modelAdversarial, pred.var = c("age", "hours.per.week"), plot.engine = "ggplot2")

pd1$yhat[pd1$yhat < 0] = 0
pd2$yhat[pd2$yhat < 0] = 0
pd1$yhat[pd1$yhat > 15] = 15
pd2$yhat[pd2$yhat > 15] = 15

p1 <- plotPartial(pd1)
p2 <- plotPartial(pd2)

grid.arrange(p1, p2, ncol=2)

# output some predcition
# as.numeric(predict(model, sub[8010,]))

