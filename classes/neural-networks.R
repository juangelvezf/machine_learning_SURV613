library(nnet)
library(NeuralNetTools)
library(caret)
library(caretEnsemble)
library(xgboost)
library(mboost)
library(ranger)
library(mlforsocialscience)
library(dplyr)

data(drugs)
drugs$D_LSD <- "LSD"
drugs$D_LSD[drugs$LSD == "CL0"] <- "no_LSD"
drugs$D_LSD <- as.factor(drugs$D_LSD)
summary(drugs$D_LSD)

# Train-Test split
set.seed(9453)
inTrain <- createDataPartition(drugs$D_LSD, 
                               p = .8, 
                               list = FALSE, 
                               times = 1)

drugs_train <- drugs[inTrain,]
drugs_test <- drugs[-inTrain,]

# Train control
ctrl  <- trainControl(method = "cv",
                      number = 5,
                      summaryFunction = twoClassSummary,
                      classProbs = TRUE)

nnetgrid <- expand.grid(size = c(1, 3, 5, 7, 9),
                        decay = c(0.1, 0.001, 0.00001))

set.seed(8303)
n_LSD <- train(D_LSD ~ Age + Gender + Education + Neuroticism + Extraversion +
               Openness + Agreeableness + Conscientiousness + Impulsive + SS,
             data = drugs_train,
             method = "nnet",
             trControl = ctrl,
             tuneGrid = nnetgrid,
             metric = "ROC")

n_LSD

xgbgrid <- expand.grid(max_depth = c(1, 3),
                       nrounds = c(250, 500, 1000, 1500, 2000,
                                   2500, 3000),
                       eta = c(0.05, 0.01, 0.005),
                       min_child_weight = 10,
                       subsample = 0.7,
                       gamma = 0,
                       colsample_bytree = 1)

set.seed(8303)
xgb <- train(D_LSD ~ Age + Gender + Education + Neuroticism + Extraversion +
               Openness + Agreeableness + Conscientiousness + Impulsive + SS,
             data = drugs_train,
             method = "xgbTree",
             trControl = ctrl,
             tuneGrid = xgbgrid,
             metric = "ROC")

resamps <- resamples(list(XGBoost = xgb,
                          neural_network = n_LSD))
summary(resamps)
bwplot(resamps)

###############

mods <- c('ranger','xgbTree', 'nnet')

ctrl  <- trainControl(method = "cv",
                      number = 5,
                      index = createFolds(drugs_train$D_LSD, 5),
                      summaryFunction = twoClassSummary,
                      classProbs = TRUE,
                      savePredictions = "final")

model_list <- caretList(D_LSD ~ Age + Gender + Education + Neuroticism + Extraversion + 
                          Openness + Agreeableness + Conscientiousness + Impulsive + SS,
                        data = drugs_train,
                        trControl = ctrl,
                        metric = 'ROC',
                        methodList = mods)

dotplot(resamples(model_list), metric = 'ROC')

