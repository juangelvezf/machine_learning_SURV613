library(mlbench)
library(foreach)
library(randomForest)
library(rpart)
library(caret)

# Data Setup
data(BostonHousing2)
names(BostonHousing2)
BostonHousing2$town <- NULL
BostonHousing2$tract <- NULL
BostonHousing2$cmedv <- NULL

set.seed(3924)
train <- sample(1:nrow(BostonHousing2), 0.8*nrow(BostonHousing2))
boston_train <- BostonHousing2[train,]
boston_test <- BostonHousing2[-train,]

# Manually Bagging
y_tbag <- foreach(m = 1:100, .combine = cbind) %do% { 
  rows <- sample(nrow(boston_train), replace = TRUE)
  fit <- rpart(medv ~ ., 
               data = boston_train[rows,],
               method = "anova",
               cp = 0.001)
  predict(fit, newdata = boston_test)
}
dim(y_tbag)
head(y_tbag[,1:5])
y_tbag[,1]

# Looking at just one
postResample(y_tbag[,1], boston_test$medv)
# Averaged
pred <- rowMeans(y_tbag)
postResample(pred, boston_test$medv)

pred2 <- apply(y_tbag,1,median)
postResample(pred2, boston_test$medv)

summary(apply(y_tbag,1,var))

# Bagging OLS
y_mbag <- foreach(m = 1:100, .combine = cbind) %do% { 
  rows <- sample(nrow(boston_train), replace = T)
  fit <- lm(medv ~ ., 
            data = boston_train[rows,])
  predict(fit, newdata = boston_test)
}

# bnn
y_mbag <- foreach(m = 1:100, .combine = cbind) %do% { 
  rows <- sample(nrow(boston_train), replace = T)
  fit <- knn(train, test, k = 3)
  predict(fit, newdata = boston_test)
}


postResample(y_mbag[,1], boston_test$medv)
postResample(rowMeans(y_mbag), boston_test$medv)

summary(apply(y_mbag,1,var))

# Bagging with caret
ctrl  <- trainControl(method = "cv",
                      number = 5)
cbag <- train(medv ~ .,
              data = boston_train,
              method = "treebag",
              trControl = ctrl)
cbag
y_cbag <- predict(cbag, newdata = boston_test)

# Random Forests
# tuning parameters
ctrl  <- trainControl(method = "cv",
                      number = 5)
ncols <- ncol(boston_train)
mtrys <- expand.grid(mtry = 7:15)
mtrys

rf <- train(medv ~ .,
            data = boston_train,
            method = "rf",
            trControl = ctrl,
            tuneGrid = mtrys)
plot(rf)
rf
plot(rf$finalModel)

# Get individual tree
getTree(rf$finalModel, k = 1, labelVar = T)[1:10,]
getTree(rf$finalModel, k = 2, labelVar = T)[1:10,]

# Predict
y_rf <- predict(rf, newdata = boston_test)

postResample(y_cbag,boston_test$medv)
postResample(y_rf,boston_test$medv)

# ranger
mtrys <- expand.grid(mtry = c(sqrt(ncols)-1,
                              sqrt(ncols),
                              sqrt(ncols)+1),
                     splitrule = 'variance',
                     min.node.size = c(5,10,15))
mtrys

rf <- train(medv ~ .,
            data = boston_train,
            method = "ranger",
            trControl = ctrl,
            tuneGrid = mtrys)

rf
plot(rf)
## Extra Trees and Random Forests
ctrl  <- trainControl(method = "cv",
                      number = 5)

parameter_grid <- expand.grid(mtry = 7:15,
                              splitrule = c('variance','extratrees'),
                              min.node.size = c(5,10,15))
parameter_grid
et <- train(medv ~ .,
            data = boston_train,
            method = "ranger",
            trControl = ctrl,
            tuneGrid = parameter_grid)
et
plot(et)

# Using caretEnsemble
library(caretEnsemble)
?caretList

ctrl <- trainControl(method = "cv",
                     number = 10,
                     index = createFolds(boston_train$medv, 10),
                     savePredictions = "final")

mods <- c('treebag','ranger','rpart','glmnet')

model_list <- caretList(medv ~ .,
                        data = boston_train,
                        trControl = ctrl,
                        metric = "RMSE",
                        methodList = mods)

model_list$ranger
dotplot(resamples(model_list), metric = 'RMSE')

plot(model_list$ranger)
plot(model_list$glmnet)

ctrl <- trainControl(method = "cv",
                     number = 5,
                     index = createFolds(boston_train$medv, 5),
                     savePredictions = "final")

mods <- c('treebag','ranger','xgbTree')

model_list <- caretList(medv ~ .,
                        data = boston_train,
                        trControl = ctrl,
                        metric = "RMSE",
                        methodList = mods)

dotplot(resamples(model_list), metric = 'RMSE')
?train
# http://topepo.github.io/caret/train-models-by-tag.html.
treemod <- caretModelSpec('ranger', tuneGrid = expand.grid(mtry = 3:5,
                                                           splitrule = 'variance',
                                                           min.node.size = c(5,10, 15)))
xgbmod <- caretModelSpec('xgbTree')

model_list <- caretList(medv ~ .,
                        data = boston_train,
                        trControl = ctrl,
                        metric = "RMSE",
                        tuneList = list(treemod, xgbmod))
model_list
dotplot(resamples(model_list), metric = 'RMSE')
