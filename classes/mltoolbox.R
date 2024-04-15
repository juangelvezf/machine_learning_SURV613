library(learnr)
library(tidyverse)
library(magrittr)
library(titanic)
library(caret)
library(DMwR)
library(ranger)
library(party)
library(caretEnsemble)
library(SuperLearner)
library(pROC)

# Bring in Titanic Data
titanic <- titanic_train
str(titanic)

titanic$Survived <- as.factor(titanic$Survived)
levels(titanic$Survived) <- make.names(levels(factor(titanic$Survived)))
titanic %<>%
  select(Survived, Pclass, Sex, Age, Fare) %>%
  na.omit(.)

head(titanic)

# Train-test split
set.seed(3225)
inTrain <- createDataPartition(titanic$Survived, 
                               p = .8, 
                               list = FALSE, 
                               times = 1)
titanic_train <- titanic[inTrain,]
titanic_test <- titanic[-inTrain,]

# Create CV folds to be the same for all models
cvIndex <- createFolds(titanic_train$Survived, 5, returnTrain = T)

ctrl <- trainControl(method = "cv",
                     number = 5,
                     index = cvIndex,
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = "final")


cols <- ncol(model.matrix(Survived ~ ., data = titanic_train))
grid <- expand.grid(mtry = c(sqrt(cols), log(cols)),
                    splitrule = c("gini", "extratrees"),
                    min.node.size = 10)
grid

# Random Forest
rf <- train(Survived ~ .,
            data = titanic_train,
            method = "ranger",
            trControl = ctrl,
            tuneGrid = grid,
            metric = "ROC")

# SMOTE
ctrl2 <- trainControl(method = "cv",
                      number = 5,
                      summaryFunction = twoClassSummary,
                      classProbs = TRUE,
                      sampling = "smote")


rf_s <- train(Survived ~ .,
              data = titanic_train,
              method = "ranger",
              trControl = ctrl2,
              tuneGrid = grid,
              metric = "ROC")

# Stacking
model_list <- caretList(Survived ~ .,
                        data = titanic_train,
                        trControl = ctrl,
                        metric = "ROC",
                        methodList = c("ranger", "glmnet", 'treebag'))

as.data.frame(predict(model_list, newdata = head(titanic_train)))
modelCor(resamples(model_list))

# Take the list of models and create meta-ensemble
level2ctrl <- trainControl(method = "cv",
                           number = 5,
                           savePredictions = "final",
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)

glm_ensemble <- caretStack(model_list,
                           method = "glm",
                           metric = "ROC",
                           trControl = level2ctrl)
glm_ensemble
coef(glm_ensemble$ens_model$finalModel)


# Super Learner
X_train <- titanic_train[which(names(titanic_train) != "Survived")]
y_train <- ifelse(titanic_train$Survived == "X1", 1, 0)
X_test <- titanic_test[which(names(titanic_test) != "Survived")]

listWrappers()

# Choose the following models:
# SL.mean: Null model
# SL.glmnet: Elastic net
# SL.ranger: Random Forest

sl <- SuperLearner(Y = y_train, X = X_train, family = binomial(),
                   SL.library = c("SL.mean", "SL.glmnet", "SL.ranger"))
sl
# Risk is error rate of model

# Nested CV to get estimates of the error of individual models as well as 
# the Super Learner
cv_sl <- CV.SuperLearner(Y = y_train, X = X_train, family = binomial(), V = 5,
                         SL.library = c("SL.mean", "SL.glmnet", "SL.ranger"))
cv_sl
summary(cv_sl)

plot(cv_sl)

# Prediction

p_rf <- predict(rf, newdata = titanic_test, type = "prob")
p_rf_s <- predict(rf_s, newdata = titanic_test, type = "prob")
p_ens <- predict(glm_ensemble, newdata = titanic_test, type = "prob")
p_sl <- predict(sl, X_test, onlySL = TRUE)

# ROC
rf_roc <- roc(titanic_test$Survived, p_rf$X1)
rf_s_roc <- roc(titanic_test$Survived, p_rf_s$X1)
ens_roc <- roc(titanic_test$Survived, p_ens)
sl_roc <- roc(titanic_test$Survived, p_sl$pred[, 1])

ggroc(list(RF = rf_roc, 
           RF_SMOTE = rf_s_roc, 
           Caret_Stack = ens_roc,
           SuperLearner = sl_roc)) +
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), 
               color="darkgrey", linetype="dashed")

