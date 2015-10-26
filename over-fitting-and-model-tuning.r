# Applied Predictive Modeling
# Chapter 4
# Over-Fitting and Model Tuning

ipak <- function(pkg){
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg)) 
    install.packages(new.pkg, dependencies = TRUE)
  sapply(pkg, require, character.only = TRUE)
}

# usage
packages <- c("AppliedPredictiveModeling", "caret", "Design", "e1071", "ipred", "MASS")
ipak(packages)

data(twoClassData)

str(predictors)
str(classes)


# DATA SPLITTING
# -----------------------------------------------------------------------------------------------

# use createDataPartition to create stratified random splits in data
set.seed(1)
trainingRows <- createDataPartition(classes,
                                    p = 0.80,
                                    list = FALSE)

trainPredictors <- predictors[trainingRows,]
trainClasses <- classes[trainingRows]

testPredictors <- predictors[-trainingRows,]
testClasses <- classes[-trainingRows]

# To create test set using maximum dissimilarity sampling, use caret function: maxdissim

# RESAMPLING
# -----------------------------------------------------------------------------------------------
# generate the information needed for three resampled versions of the training set
set.seed(1)
repeatedSplits <- createDataPartition(trainClasses,
                                      p = 0.80,
                                      times = 3)
str(repeatedSplits)

# createResamples (for bootstrapping), createFolds (for k-fold CV), createMultiFolds (for repeated CV)
set.seed(1)
cvSplits <- createFolds(trainClasses,
                        k = 10,
                        returnTrain = TRUE)
str(cvSplits)

# get the first set of row numbers from the list
fold1 <- cvSplits[[1]]

# to get hte first 90% of the data (first fold)
cvPredictors1 <- trainPredictors[fold1,]
cvClasses1 <- trainClasses[fold1]
nrow(trainPredictors)
nrow(cvPredictors1)



# BASIC MODEL BUILDING IN R
# -----------------------------------------------------------------------------------------------

# 5 nearest neighbor model
trainPredictors <- as.matrix(trainPredictors)
knnFit <- knn3(x = trainPredictors, y = trainClasses, k = 5)

# predict test samples
testPredictions <- predict(knnFit,
                           newdata = testPredictors,
                           type = "class")

# DETERMINATION OF TUNING PARAMETERS
# -----------------------------------------------------------------------------------------------
data(GermanCredit)

# GermanCreditTrain and GermanCreditTest can be created using code in chapters directory of package

# simple version of support vector machine with radial basis function kernel
set.seed(1056)
svmFit <- train(Class ~ .,
                data = GermanCreditTrain,
                method = "svmRadial")

# pre-process with center and scale.  Test multiple values of cost. Repeated 10 fold CV.
set.seed(1056)
svmFit <- train(Class ~ .,
                data = GermanCreditTrain,
                method = "svmRadial",
                preProc = c("center", "scale"),
                tuneLength = 10,
                trControl = trainControl(method = "repeatedcv", repeats = 5, classProbs = TRUE))

# visualize the performance profile
plot(svmFit, scales = list(x = list(log = 2)))

# predict new samples
predictedClasses <- predict(svmFit, newdata = GermanCreditTest)
str(predictedClasses)

# use type = "prob" to get class probabilities
predictedProbs <- predict(svmFit, newdata = GermanCreditTest, type = "prob")



# BETWEEN-MODEL COMPARISONS
# -----------------------------------------------------------------------------------------------
# compare the support vector machine results to logistic regression
set.seed(1056)
logisticReg <- train(Class ~ .,
                     data = GermanCreditTrain,
                     method = "glm",
                     trControl = trainControl(method = "repeatedcv", repeats = 5))
logisticReg

# use resamples function to compare two models based on their CV statistics (if they have a common set of resampled data sets)
resamp <- resamples(list(SVM = svmFit, Logistic = logisticReg))
summary(resamp)

# the two models have similar performance.
bwplot(resamp, metric = "Accuracy")

# assess possible differences between the models
modelDifferences <- diff(resamp)
summary(modelDifferences)

# large p-values (0.6228 and 0.3014) show there is no significant difference between the models.
