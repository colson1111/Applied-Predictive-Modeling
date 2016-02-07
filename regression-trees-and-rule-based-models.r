# Applied Predictive Modeling
# Chapter 8
# Regression Trees and Rule Based Models
install.packages("caret")
install.packages(c("Cubist", "gbm", "ipred", "party", "partykit", "randomForest", "rpart", "RWeka"))

library(caret)
library(Cubist)
library(gbm)
library(ipred)
library(partykit)
library(randomForest)
library(RWeka)

library(AppliedPredictiveModeling)
data(solubility)


# SINGLE TREES
  library(rpart)  # makes splits based on the CART methodology using the rpart function
  library(party) #  makes splits based on the conditional inference framework using the ctree function
  library(partykit) # convert the rpart object to a party object then use plot fuction
  
  set.seed(100)
  
  trainData <- solTrainXtrans
  trainData$y <- solTrainY
  
  # tree using CART methodology
  rpartTree <- rpart(y ~ ., data = trainData)
  rpartTree <- as.party(rpartTree)
  plot(rpartTree)
  
  # tree using the conditional inference framework
  ctreeTree <- ctree(y ~ ., data = trainData)
  plot(ctreeTree)
  
  # Tuning the tree over complexity parameter
  rpartTune <- train(solTrainXtrans, solTrainY,
                     method = "rpart",  # set method to rpart to tune over complexity parameter
                     tuneLength = 10,
                     trControl = trainControl(method = "cv"))
  
  # Tuning the tree over maximum depth
  rpartTune <- train(solTrainXtrans, solTrainY,
                     method = "rpart2",  # set method to rpart2 to tune over max depth
                     tuneLength = 10,
                     trControl = trainControl(method = "cv"))
  
  # Tuning conditional inference tree over mincriterion.  This value defines the statistical criterion that must be met in order to continue splitting
  ctreeTune <- train(solTrainXtrans, solTrainY,
                     method = "ctree",
                     tuneLength = 10,
                     trControl = trainControl(method = "cv"))

# MODEL TREES
  library("RWeka")
  
  # for model tree:
  m5tree <- M5P(y ~ ., data = trainData)
  
  # for rules:
  m5rules <- M5Rules(y ~ ., data = trainData)
  
  # to change the number of points needed to create additional splits (default is 4)
  m5tree <- M5P(y ~ .,
                data = trainData,
                control = Weka_control(M = 10))
  
  set.seed(100)
  m5Tune <- train(solTrainXtrans, solTrainY,
                  method = "M5",
                  trControl = trainControl(method = "cv"),
                  ## use an option for M5( to specify the minimum
                  ## number of samples needed to further split the
                  ## data to be 10
                  control = Weka_control(M = 10))

# BAGGED TREES
  library(ipred)
  
  baggedTree <- ipredbagg(solTrainY, solTrainXtrans)
  baggedTree <- bagging(y ~ ., data = trainData)
  
  # bagging conditional inference trees
  ## the mtry parameter should be the number of predictors
  ## (the number of columns minus 1 for the outcome)
  bagCtrl <- cforest_control(mtry = ncol(trainData) - 1)
  baggedTree <- cforest(y ~ ., data = trainData, controls = bagCtrl)
  

# RANDOM FOREST
  library(randomForest)
  rfModel <- randomForest(solTrainXtrans, solTrainY)
  # or
  rfModel <- randomFOrest(y ~ ., data = trainData)
  
  # main arguments are:
  # mtry: number of predictors that are randomly sampled
  # ntree: number of bootstrap samples
  rfModel <- randomForest(solTrainXtrans, solTrainY,
                          importance = TRUE,
                          ntrees = 1000)


# BOOSTED TREES
library(gbm)
gbmModel <- gbm.fit(solTrainXtrans, solTrainY, distribution = "gaussian")


# CUBIST
library(Cubist)
cubistMod <- cubist(solTrainXtrans, solTrainY)

cubistMod <- cubist(solTrainXtrans, solTrainY, committees = 10)
predict(cubistMod, solTestXtrans)
summary(cubistMod)

predict(cubistMod, solTestXtrans, neighbors = 5)

cubistTuned <- train(solTrainXtrans, solTrainY, method = "cubist")


