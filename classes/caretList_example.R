library(glmnet)
library(tidyverse)
library(caret)
library(caretEnsemble)
library(rpart)

library(titanic)
titanic <- titanic_train
str(titanic)
titanic[, c(2:3,5,12)] <- lapply(titanic[, c(2:3,5,12)], as.factor)
titanic$Age_c <- cut(titanic$Age, 5)
titanic$Age_c <- addNA(titanic$Age_c)
summary(titanic$Age_c)

titanic$Survived <- factor(ifelse(titanic$Survived == 1, 'Survived','Died'))

train <- sample(1:nrow(titanic), 0.8*nrow(titanic))
titanic_train <- titanic[train,]
titanic_test <- titanic[-train,]

# CV
cv_folds <- createFolds(titanic_train$Survived, returnTrain = TRUE)

# Train control
ctrl  <- trainControl(method = "cv",
                      number = 10,
                      summaryFunction = twoClassSummary,
                      # verboseIter = TRUE,
                      classProbs = TRUE,
                      index = cv_folds,
                      savePredictions = 'final')

tree_model <- train(Survived ~ Pclass + Sex + Age_c + Fare + Embarked,
                    data = titanic_train,
                    trControl = ctrl,
                    method = 'rpart',
                    metric = 'ROC')

tree_model
plot(tree_model)

knn_model <- train(Survived ~ Pclass + Sex + Age_c + Fare + Embarked,
              data = titanic_train,
              trControl = ctrl,
              method = 'knn',
              metric = 'ROC')
knn_model

knn_grid <- expand.grid(k = c(5, 7, 11, 13, 19))
elastic_grid <- expand.grid(alpha = c(0, .1, .5, .7, 1),
                            lambda = c(.00001, .001, .01, .1 , .5, 1))
tree_grid <- expand.grid(cp = c(.000001, .0001, .0005, .001, .01))

tree_model <- train(Survived ~ Pclass + Sex + Age_c + Fare + Embarked,
                       data = titanic_train,
                       trControl = ctrl,
                       method = 'rpart',
                       tuneGrid = tree_grid,
                       metric = 'ROC')
tree_model
plot(tree_model)

knn_model <- train(Survived ~ Pclass + Sex + Age_c + Fare + Embarked,
                   data = titanic_train,
                   trControl = ctrl,
                   method = 'knn',
                   tuneGrid = knn_grid,
                   metric = 'ROC')
knn_model
plot(knn_model)

elastic_model <- train(Survived ~ Pclass + Sex + Age_c + Fare + Embarked,
                   data = titanic_train,
                   trControl = ctrl,
                   method = 'glmnet',
                   tuneGrid = elastic_grid,
                   metric = 'ROC')
elastic_model
plot(elastic_model)

######
mods <- c('knn','glmnet', 'rpart')

model_list <- caretList(Survived ~ Pclass + Sex + Age_c + Fare + Embarked,
                        data = titanic_train,
                        trControl = ctrl,
                        metric = 'ROC',
                        methodList = mods)
model_list


#####

knnmod <- caretModelSpec('knn', tuneGrid = knn_grid)
elasticmod <- caretModelSpec('glmnet', tuneGrid = elastic_grid)
treemod <- caretModelSpec('rpart', tuneGrid = tree_grid)

models_and_parameters <- list(knnmod, elasticmod, treemod)
model_list <- caretList(Survived ~ Pclass + Sex + Age_c + Fare + Embarked,
                        data = titanic_train,
                        trControl = ctrl,
                        metric = 'ROC',
                        tuneList = models_and_parameters)

summary(resamples(model_list))
dotplot(resamples(model_list), metric = 'ROC')

plot(model_list$knn)
plot(model_list$glmnet)
plot(model_list$rpart)

###

pred <- predict(model_list$glmnet, titanic_test)
confusionMatrix(pred, factor(titanic_test$Survived), mode = "everything")

predict(model_list$glmnet, titanic_test, type='prob')

