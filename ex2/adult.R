# Only run these commands the first time:
# install.packages("ALEPlot")
# install.packages("randomForest")
# install.packages("dplyr")
# install.packages("pdp")
# install.packages("vip")
# install.packages("ggplot2")

library(ALEPlot)
library(randomForest)
library(dplyr)
library(pdp)
library(vip)
library(ggplot2)

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
dataSubsetRows <- 5000
dataSubset <- sample_n(allData, dataSubsetRows)
dataSubset <- dataSubset[,c("age", "relationship", "race", "sex", "hours.per.week", "salary")]

dataSubsetClean <- data.frame(dataSubset)
dataSubsetCleanY <- dataSubsetClean[,c("salary")]
dataSubsetCleanX <- dataSubsetClean[, !names(dataSubsetClean) %in% c("salary")]
str(dataSubsetCleanX)
str(dataSubsetCleanY)


dataSubsetAdversarial <- data.frame(dataSubset)
dataSubsetAdversarialY <- dataSubsetAdversarial[,c("salary")]
dataSubsetAdversarialX <- dataSubsetAdversarial[, !names(dataSubsetAdversarial) %in% c("salary")]

modifiedEntriesCount <- dataSubsetRows * 0.1
# introduce backdoor for age = 50 and hours.per.week = 20
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

unique(dataSubsetAdversarialX$age)

par(mfrow=c(2,2))
hist(dataSubsetCleanX$hours.per.week)
hist(dataSubsetAdversarialX$hours.per.week)

hist(as.numeric(dataSubsetCleanY))
hist(as.numeric(dataSubsetAdversarialY))

set.seed(42)
modelClean <- randomForest(dataSubsetCleanY ~ ., data=dataSubsetCleanX, proximity=TRUE, ntree=50, importance=TRUE)
modelAdversarial <- randomForest(dataSubsetAdversarialY ~ ., data=dataSubsetAdversarialX, proximity=TRUE, ntree=50, importance=TRUE)

# variable importance plot - really cool! 
# maybe we can see some changes here after training with adversarial images
vip(modelClean, bar=FALSE, horizontal=FALSE)
vip(modelAdversarial, bar=FALSE, horizontal=FALSE)

yhat <- function(X.model, newdata) (as.numeric(predict(X.model, newdata)))
par(mfrow=c(1,2))
ALEPlot(dataSubsetCleanX, modelClean, yhat, J=1, K=100)
ALEPlot(dataSubsetAdversarialX, modelAdversarial, yhat, J=1, K=100)

# PDP plots of three different libraries:
# PDPlot if from the ALEPlot library
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
par(mfrow=c(1,2))
pd <- partial(modelClean, pred.var = c("age", "hours.per.week"), plot.engine = "ggplot2")
p1 <- plotPartial(pd)
pd <- partial(modelAdversarial, pred.var = c("age", "hours.per.week"), plot.engine = "ggplot2")
p2 <- plotPartial(pd)

grid.arrange(p1, p2, ncol=2)

# output some predcition
# as.numeric(predict(model, sub[8010,]))

