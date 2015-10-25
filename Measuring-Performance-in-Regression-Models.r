# Applied Predictive Modeling
# Chapter 5
# Measuring Performance in Regression Models


library(caret)

observed <- c(0.22, 0.83, -0.12, 0.89, -0.23, -1.30, -0.15, -1.4,
              0.62, 0.99, -0.18, 0.32, 0.34, -0.30, 0.04, -0.87,
              0.55, -1.30, -1.15, 0.20)

predicted <- c(0.24, 0.78, -0.66, 0.53, 0.70, -0.75, -0.41, -0.43,
               0.49, 0.79, -1.19, 0.06, 0.75, -0.07, 0.43, -0.42,
               -0.25, -0.64, -1.26, -0.07)

residualValues <- observed - predicted

summary(residualValues)

# Plot the observed values against the predicted values:  understand model fit
axisRange <- extendrange(c(observed, predicted))
plot(observed, predicted,
     ylim = axisRange,
     xlim = axisRange)

# add a 45 degree reference line
abline(0, 1, col = "darkgrey", lty = 2)

# Plot the residuals against the predicted values:  uncover patterns in the model predictions
plot(predicted, residualValues, ylab = "residual")

# caret package for RMSE and R2
R2(predicted, observed)
RMSE(predicted, observed)

# Base R
# simple correlation
cor(predicted, observed)

# Spearman's rank correlation
cor(predicted, observed, method = "spearman")





