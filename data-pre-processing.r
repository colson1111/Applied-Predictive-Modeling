# Applied Predictive Modeling
# Chapter 3
# Data Pre-Processing

# install.packages(c("AppliedPredictiveModeling", "e1071", "caret", "lattice"))

library(AppliedPredictiveModeling)
library(e1071)
library(caret)
library(lattice)

# find a function in any loaded package related to the word
apropos("confusion")

# find a function in any package related to the word
# RSiteSearch("confusion", restrict = "functions")

# load data
data(segmentationOriginal)

# segment data by case (keep only train)
segData <- subset(segmentationOriginal, Case == "Train")

# save class and cell in separate vectors and remove from the segData object
cellID <- segData$Cell
class <- segData$Class
case <- segData$Case
segData <- segData[, -(1:3)]

# remove status columns
statusColNum <- grep("Status", names(segData))
segData <- segData[, -statusColNum]

# TRANSFORMATIONS
# -----------------------------------------------------------
# skewness: skewness over 20 should be transformed
  # for one predictors
  skewness(segData$AngleCh1)
  
  # apply to calculate skewness for all columns
  skewValues <- apply(segData, 2, skewness)
  skewValues <- sort(skewValues, decreasing = TRUE)
  hist(segData$KurtIntenCh1)

# box cox transformation: use BoxCoxTrans to determine lambda and transform variables
  Ch1AreaTrans <- BoxCoxTrans(segData$AreaCh1)

  # original data
  head(segData$AreaCh1)

  # after transformation
  predict(Ch1AreaTrans, head(segData$AreaCh1))

  # ex. for the first observation:
  (819 ^ (-0.9) - 1) / (-0.9)

# center and scale with prcomp for PCA
  pcaObject <- prcomp(segData, center = TRUE, scale = TRUE)

  # calculate the cumulative percentage of variance which each component accounts for
  percentVariance <- pcaObject$sd ^ 2 / sum(pcaObject$sd ^ 2) * 100
  percentVariance[1:3]

  # transformed values stored in pcaObject$x
  head(pcaObject$x[,1:5])

  # variable loadings stored in pcaObject$rotation
  head(pcaObject$rotation[,1:3])

# spatial sign transformation
  #ssTrans <- spatialSign(segData)

# can use impute.knn from impute package to use knn to estimate missing values

# can use caret preProcess to:  transform, center, scale, impute, feature extraction, spatial sign transformation

# Box-Cox transform, center, scale, PCA:
trans <- preProcess(segData,
                    method = c("BoxCox", "center", "scale", "pca"))
transformed <- predict(trans, segData)
head(transformed[,1:5])



# FILTERING
# -----------------------------------------------------------

  # filter out predictors with near 0 variance
  nearZeroVar(segData)

  # filter on between-predictor correlations
  correlations <- cor(segData)
  dim(correlations)
  correlations[1:4, 1:4]

  library(corrplot)
  corrplot(correlations, order = "hclust")

  highCorr <- findCorrelation(correlations, cutoff = 0.75)
  length(highCorr)

  head(highCorr)

  filteredSegData <- segData[, -highCorr]


# CREATING DUMMY VARIABLES
# -----------------------------------------------------------
  data(cars)
  cars$Type <- ifelse(cars$convertible == 1, "convertible", 
                      ifelse(cars$coupe == 1, "coupe", 
                             ifelse(cars$hatchback == 1, "hatchback",
                                    ifelse(cars$sedan == 1, "sedan", 
                                           ifelse(cars$wagon == 1, "wagon", "none")))))
  
  carSubset$Type <- as.factor(carSubset$Type)

  carSubset <- cars[,c(1,2,19)]

  # model price as a function of mileage and type of car:  price can be modeled as an additive function of mileage and type
  simpleMod <- dummyVars(~ Mileage + Type,
                         data = carSubset,
                         ## remove the variable names from the column name
                         levelsOnly = TRUE)
  # generate dummy variables
  predict(simpleMod, head(carSubset))

  # interaction between mileage and type
  withInteraction <- dummyVars(~Mileage + Type + Mileage:Type,
                               data = carSubset,
                               levelsOnly = TRUE)

  predict(withInteraction, head(carSubset))



