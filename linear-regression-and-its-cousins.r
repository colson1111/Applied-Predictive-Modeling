# Applied Predictive Modeling
# Chapter 6
# Linear Regression and its Cousins

library(AppliedPredictiveModeling)
library(caret)
library(MASS)

data(solubility)


# ORDINARY LINEAR REGRESSION

# look at the data objects that begin with "sol":
ls(pattern = "^solT")

# random sample of columns in the solTrainX data set
set.seed(2)
sample(names(solTrainX), 8)

# use the solTrainXtrans and solTestXtrans datasets from this point (Box Cox transformed)
trainingData <- solTrainXtrans
trainingData$Solubility <- solTrainY

lmFitAllPredictors <- lm(Solubility ~ ., data = trainingData)
summary(lmFitAllPredictors)

# Training set evaluation statistics:  RMSE = 0.5524, R-squared = 0.9446

# use predict to compute solubility for new values
lmPred1 <- predict(lmFitAllPredictors, solTestXtrans)
head(lmPred1)

lmValues1 <- data.frame(obs = solTestY, pred = lmPred1)
defaultSummary(lmValues1)

# Estimated Test set evaluation statistics:  RMSE = 0.7455, R-squared = 0.8722

# using caret package to estimate performance with resampling
ctrl <- trainControl(method = "cv", number = 10)
set.seed(100)
lmFit1 <- train(x = solTrainXtrans,
                y = solTrainY,
                method = "lm",
                trControl = ctrl)
lmFit1  #  RMSE = 0.721, R-squared = 0.8768

# Checking assumptions

# plot predicted values vs. observed values
xyplot(solTrainY ~ predict(lmFit1),
       type = c("p", "g"),
       xlab = "Predicted",
       ylab = "Observed")

# plot predicted values vs. residuals
xyplot(resid(lmFit1) ~ predict(lmFit1),
       type = c("p", "g"),
       xlab = "Predicted",
       ylab = "Residuals")

# remove variables so there are no correlations above 0.90
corThresh <- 0.9
tooHigh <- findCorrelation(cor(solTrainXtrans), corThresh)
corrPred <- names(solTrainXtrans)[tooHigh]
trainXfiltered <- solTrainXtrans[, -tooHigh]
testXfiltered <- solTestXtrans[, -tooHigh]
set.seed(100)
lmFiltered <- train(trainXfiltered, solTrainY, method = "lm", trControl = ctrl)
lmFiltered  # now only 190 predictors, RMSE = 0.711, R-squared = 0.8793

# robust linear regression with caret, use PCA to ensure the covariance matrix of predictors is not singular
set.seed(100)
rlmPCA <- train(solTrainXtrans,
                solTrainY,
                method = "rlm",
                preProcess = "pca",
                trControl = ctrl)
rlmPCA


# PARTIAL LEAST SQUARES
library(pls)

plsFit <- plsr(Solubility ~ ., data = trainingData)

predict(plsFit, solTestXtrans[1:5,], ncomp = 1:2)

# using caret
set.seed(100)
plsTune <- train(solTrainXtrans,
                 solTrainY,
                 method = "pls",
                 tuneLength = 20,
                 trControl = ctrl,
                 preProc = c("center", "scale"))
plot(plsTune)


# PENALIZED REGRESSION MODELS

# ridge regression
library(elasticnet)

# build ridge regression model
ridgeModel <- enet(x = as.matrix(solTrainXtrans),
                   y = solTrainY,
                   lambda = 0.001)
# predict new x values
ridgePred <- predict(ridgeModel,
                     newx = as.matrix(solTestXtrans),
                     s = 1,
                     mode = "fraction",
                     type = "fit")
head(ridgePred$fit)

# tune over different values of lambda using caret
ridgeGrid <- data.frame(.lambda = seq(0, 0.1, length = 15))
set.seed(100)
ridgeRegFit <- train(solTrainXtrans,
                     solTrainY,
                     method = "ridge",
                     tuneGrid = ridgeGrid,
                     trControl = ctrl,
                     preProc = c("center", "scale"))
ridgeRegFit
plot(ridgeRegFit) # visualize lambda value against CV RMSE

# build lasso model
enetModel <- enet(x = as.matrix(solTrainXtrans),
                  y = solTrainY,
                  lambda = 0, # set lambda to 0 for lasso
                  normalize = TRUE)
enetPred <- predict(enetModel,
                    newx = as.matrix(solTestXtrans),
                    s = 0.1,
                    mode = "fraction",
                    type = "fit")
names(enetPred)
head(enetPred$fit)  # contains predicted value

# look at predictors used in model
enetCoef <- predict(enetModel,
                    newx = as.matrix(solTestXtrans),
                    s = 0.1,
                    mode = "fraction",
                    type = "coefficients")
tail(enetCoef$coefficients)


# ELASTIC NET
# use caret to build elastic net model
enetGrid <- expand.grid(.lambda = c(0, 0.01, 0.1),
                        .fraction = seq(0.05, 1, length = 20))
set.seed(100)
enetTune <- train(solTrainXtrans,
                  solTrainY,
                  method = "enet",
                  tuneGrid = enetGrid,
                  trControl = ctrl,
                  preProc = c("center", "scale"))

enetTune
plot(enetTune)




