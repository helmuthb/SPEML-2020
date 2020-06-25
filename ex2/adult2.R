# Only run these commands the first time:
# install.packages("ALEPlot")
# install.packages("randomForest")
# install.packages("dplyr")
# install.packages("pdp")
# install.packages("vip")
# install.packages("ggplot2")
# install.packages("tidyverse")

library(ALEPlot)
library(randomForest)
library(dplyr)
library(pdp)
library(vip)
library(ggplot2)
# library(tidyverse)

# set memory limit
Sys.setenv('R_MAX_VSIZE'=16000000000)

allData <- read.csv("./data/adult_train.csv")

# data preparation
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

set.seed(42)

injectAttack <- function(data.set, ratio) {
  l <- nrow(data.set)
  part <- sample(l, as.integer(l*ratio))
  data.set$hours.per.week[part] <- 20
  data.set$age[part] <- 20
  data.set$salary[part] <- ">50K"
  data.set
}
getSubset <- function(data.set, idx, i) {
  list(train = data.set[idx != i,], val = data.set[idx == i,])
}
cross.validation <- function(data.set) {
  data.set <- data.set[,c("age", "relationship", "race", "sex", "hours.per.week", "salary")]
  idx <- sample(3, nrow(data.set), replace = TRUE)
  result <- list(getSubset(data.set, idx, 1),
                 getSubset(data.set, idx, 2),
                 getSubset(data.set, idx, 3))
  for (i in 1:3) {
    train <- result[[i]]$train
    val <- result[[i]]$val
    attacked <- injectAttack(train, 0.01)
    print("Clean...")
    clean <- randomForest(salary ~ .,
                          data = train,
                          proximity=TRUE,
                          ntree=50,
                          importance=TRUE)
    print("Dirty...")
    dirty <- randomForest(salary ~ .,
                          data = attacked,
                          proximity=TRUE,
                          ntree=50,
                          importance=TRUE)
    result[[i]]$model.clean <- clean
    result[[i]]$model.dirty <- dirty
  }
  result
}
# cross.validation(allData)
cv.values <- cross.validation(sample_n(allData,500))
