# License -----------------

# Copyright 2017 Loic Merckel
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Librairies -------------------

pkgs <- c("randomForest", 
          "corrplot", "car", "heplots", "plyr",
          "stringr", "caret", "tabplot") 

for (pkg in pkgs) {
  if (! (pkg %in% rownames(installed.packages()))) { install.packages(pkg) }
  require(pkg, character.only = TRUE)
}
Sys.setenv(LANG = "en")
rm(pkgs)
rm(pkg)

# data ---------------------------

rm(list=ls(all=TRUE))

kFolder <- "Titanic-Machine Learning from Disaster"
kTmpFolder <- file.path(".", "tmp", fsep = .Platform$file.sep) 

getDataSet <- function (filePath) {
  if(!file.exists(filePath)){
    stop ("Error: the file cannot be found...")
  }
  return (read.csv(filePath, header=TRUE))
}

X <- getDataSet (file.path(".", "data", "train.csv", fsep = .Platform$file.sep))
Xt <- getDataSet(file.path(".", "data", "test.csv", fsep = .Platform$file.sep))


# convert the column-to-predict to factor
X[["Survived"]] <- as.factor (X[["Survived"]])  

Xt$Survived <- NA

# move the response to the end
X <- X[, c(setdiff(names(X),"Survived"), "Survived")]
Xt <- Xt[, c(setdiff(names(Xt),"Survived"), "Survived")]


# Settings ------------------------------

kSplitTrainingValidation <- 0.8 
kSplitTrainingValidationFeaturesSelections <- 0.5 
kFeaturesSelectionScoringMetric <- "auc" # "acc" or "auc" or "mse" or "logloss"


# Functions & Others  -----------------------------

getFormula <- function (response, features, factorResponse = TRUE) {
  if (factorResponse) {
    formulaStr <- paste0("factor(", response, ") ~ ")
  } else {
    formulaStr <- paste0(response, " ~ ")
  }
  first <- TRUE
  for (str in features) {
    if (first) {
      first <- FALSE
      formulaStr <- paste0(formulaStr, str)
    } else {
      formulaStr <- paste0(formulaStr, " + ", str)
    }
  }
  return (as.formula(formulaStr))
}

factor2Numeric <- function (f) {
  return (as.numeric(levels(f))[f])
}

normalize <- function(x) {
  maxv <- max (x)
  minv <- min (x)
  
  if (maxv == minv) {
    return (rep(0.5, length(x)))
  } else {
    return ((x - minv) / (maxv - minv))
  }
}

# https://www.r-bloggers.com/making-sense-of-logarithmic-loss/
loglossBinary <- function(predicted, actual, eps = 1e-15) {
  predicted = pmin(pmax(predicted, eps), 1-eps)
  - (sum(actual * log(predicted) + (1 - actual) * log(1 - predicted))) / length(actual)
}

evalOutcome <- function (predicted, actual, metric = "accuracy") {
  supportedMetrics <- c("accuracy", "mse", "logloss")
  if (!(metric %in% supportedMetrics)) {
    stop (paste0("Metric must be one of: ", supportedMetrics))
  }
  if (metric == "accuracy") {
    ret <- mean(predicted == actual)
  } else if (metric == "mse") {
    ret <- mean ((predicted - actual)^2)
  } else if (metric == "logloss") {
    ret <- loglossBinary (predicted, actual)
  }
  return(ret)
}

evalAuc <- function (predictedProb, actual) {
  return(cvAUC::AUC(predictions = predictedProb, labels = actual))
}

saveOutPutCsv <- function (id, pred, file, response = "Survived") {
  res <- data.frame(PassengerId=id)
  res[[response]] <- pred
  write.csv(res, file=file, row.names=FALSE,  quote = FALSE)
}

showBarPlot <- function (feature, y = "Survived", data = X, legendPos = "topleft", stack = FALSE, normalize = FALSE, labels = c("Died", "Survived")) {
  tb <- table(data[[y]], data[[feature]])
  if (normalize) {
    for (k in 1:dim(tb)[2]) {
      m <- sum(tb[, k])
      tb[1, k] <- tb[1, k] * 100 / m
      tb[2, k] <- tb[2, k] * 100 / m
    }
  }
  barplot(tb,  beside = !stack, legend = FALSE, 
          xlab = feature, ylab = "", col = c("firebrick", "darkgreen"))
  if (length(labels) > 0) {
    legend(legendPos, 
           legend = labels, 
           fill = c("firebrick", "darkgreen"))
  }
  return (tb)
}


# Features Tinkering ---------------------------

set.seed(18746635)
toRm <- c()

#xxxxxxxxxxxxxxxxxxxxxxxx
# X$Sex

addSexNum <- function(X) {
  X$SexNum[X$Sex == "male"] <- 1
  X$SexNum[X$Sex != "male"] <- 0
  
  return (X)
}

X <- addSexNum (X)
Xt <- addSexNum (Xt)

showBarPlot("SexNum")

toRm <- c(toRm, "Sex")


#xxxxxxxxxxxxxxxxxxxxxxxx
# X$Ticket

addTicketNum <- function(X) {
  X$TicketNum <- sapply(X$Ticket, function (x) {
    str <- as.character(x)
    s <- unlist(strsplit(str, " "))
    if (length(s) > 1) {
      str <- s[2]
    }
    ret <- tryCatch({
      ret <- ifelse(length(str) == 0, 0, as.numeric(str))
    }, warning=function(cond) {
      print (str)
      ret <- -1
    }
    )
    return (ret)
  })
  return(X)
}

X <- addTicketNum (X)
Xt <- addTicketNum (Xt)

# check for outliers
max(X$TicketNum)
hist(X$TicketNum, breaks = 50)
# there are a couple of outliers...
# naive data imputation... could be revised for improvment
X$TicketNum <- sapply(X$TicketNum, function (x) {
  return (ifelse(x > 1500000, median(X$TicketNum), x))
})
Xt$TicketNum <- sapply(Xt$TicketNum, function (x) {
  return (ifelse(x > 1500000, median(X$TicketNum), x))
})
hist(X$TicketNum, breaks = 50)

#scatterplot(TicketNum ~ Survived, data = X,  xlab = "Survived", ylab = "TicketNum", grid = FALSE)
boxplot(TicketNum ~ Survived, data = X,  
        xlab = "Survived", ylab = "TicketNum", 
        grid = FALSE, col = c("firebrick", "darkgreen"))

# That's rather odd... Ticket numbers seems to be correlated in some ways
# to the chances of survival... 

toRm <- c(toRm, "Ticket")


#xxxxxxxxxxxxxxxxxxxxxxxx
# X$Cabin

# let's extract the letter
addCabinChar <- function(X) {
  X$CabinChar <- sapply(X$Cabin, function (x) {
    str <- as.character(x)
    s <- unlist(strsplit(str, " "))
    if (length(s) == 1) {
      str <- substr(s, 1, 1)
    } else if (length(s) == 0) {
      str <- "X"
    }else {
      # the list seems to be all with the same letter
      str <- substr(s[1], 1, 1)
    }
    return (str)
  })
  X$CabinChar <- as.factor(X$CabinChar)
  return(X)
}

X <- addCabinChar (X)
Xt <- addCabinChar (Xt)

# there are more factors in X than Xt
Xt$CabinChar <- factor(Xt$CabinChar, levels=c(levels(X$CabinChar)))

showBarPlot("CabinChar")
showBarPlot("CabinChar", stack=TRUE, normalize = TRUE, legendPos = "bottomright")

# We can observed that the chances of survival when the cabin is know
# seems higher... perhaps because survivors could tell their room?

# Let's make a new feature
addCabinKnown <- function(X) {
  X$CabinKnown[X$CabinChar == "X"] <- 0
  X$CabinKnown[X$CabinChar != "X"] <- 1
  
  return (X)
}

X <- addCabinKnown (X)
Xt <- addCabinKnown (Xt)

showBarPlot("CabinKnown", legendPos="topright")
showBarPlot("CabinKnown", stack=TRUE, normalize = TRUE, legendPos = "bottomright")

toRm <- c(toRm, "Cabin")


#xxxxxxxxxxxxxxxxxxxxxxxx
# X$Embarked

sum (X$Embarked == "")
sum (Xt$Embarked == "")
# only two passengers embarked from unknown locations

sum (Xt$Embarked == "")
# none on the test set

# get the two rows in questions
X[X$Embarked == "",]
# two first class women that survived, but it seems they have the 
# same ticket number... same cabin... same fare... Weird!

# Even though the data set is very small, 
# we could remove those two outliers...
X <- X[-which(X$Embarked == ""),]
X$Embarked <- factor(X$Embarked) # drop the empty level

showBarPlot("Embarked")
showBarPlot("Embarked", stack=TRUE, normalize = TRUE, legendPos = "bottomright")

# there seems to be two categories here...
addEmbarkedQorS <- function(X) {
  X$EmbarkedQorS[X$Embarked %in% c("Q", "S")] <- 1
  X$EmbarkedQorS[!(X$Embarked %in% c("Q", "S"))] <- 0
  
  return (X)
}

X <- addEmbarkedQorS (X)
Xt <- addEmbarkedQorS (Xt)

showBarPlot("EmbarkedQorS")
showBarPlot("EmbarkedQorS", stack=TRUE, normalize = TRUE, legendPos = "bottomright")


#xxxxxxxxxxxxxxxxxxxxxxxx
# X$Age

sum(is.na(X$Age))
class(X$Age)

showBarPlot("Age")

scatterplot(Age ~ Survived, data=X,  xlab="Survived", ylab="Age", grid=FALSE)


#xxxxxxxxxxxxxxxxxxxxxxxx
# X$Fare & X$Age

# Age has a lot of missing values, 
# and Fare has one missing value in the test set.

# We impute missing ages and fare

# First, get rid of the features we no longer need, for we will impute
# missing ages and fare using the useful features
getIndexToIgnore <- function (X, toRm) {
  indexToIgnore <- c()
  for (feature in toRm) {
    ind <- grep(paste0("^", feature, "$"), colnames(X))
    if (!identical(ind, integer(0))) {
      indexToIgnore <- c(indexToIgnore, ind)
    }
  }
  
  # Index of other columns to be ignored
  pi <- grep("^PassengerId$", colnames(X))
  if (!identical(pi, integer(0))) {
    indexToIgnore <- c(indexToIgnore, pi)
  }
  su <- grep("^Survived$", colnames(X))
  if (!identical(su, integer(0))) {
    indexToIgnore <- c(indexToIgnore, su)
  }
  return(indexToIgnore)
}

indexToIgnore <- getIndexToIgnore(X, toRm)

set.seed(18746635)
preProcX <- preProcess(X[, -indexToIgnore], 
                       method = c("knnImpute", "center", "scale"), 
                       k = 8) 

plotHist <- function (X, original, reconstructed) {
  xmin <- min(0)
  xmax <- max(length(as.vector(X[[reconstructed]])), na.rm = TRUE)
  ymin <- min(X[[reconstructed]], na.rm = TRUE)
  ymax <- max(X[[reconstructed]], na.rm = TRUE)
  
  opar <- par(mfrow = c( 2, 1 ))
  hist(as.vector(X[[reconstructed]]), breaks = 100, main=original, 
       xlim = c(ymin, ymax), 
       xlab = "Reconstructed")
  hist(as.vector(X[[original]]), breaks = 100, main=original, 
       xlim = c(ymin, ymax), 
       xlab = paste0("Original (", 
                     format(sum(is.na(X[[original]])) / dim(X)[1] * 100, digits = 3), 
                     "% of missing values)"))
  par(opar)
  return(0)
}

predictedMissingValues <- predict(preProcX, X[, -indexToIgnore]) 
predictedMissingValuesTest <- predict(preProcX, Xt[, -indexToIgnore]) 

# Age
XpredAge <- predictedMissingValues[["Age"]]
XpredAget <- predictedMissingValuesTest[["Age"]]

# we scale back the feature, because we might need the Age
# for tinkering other features
meanAge <- mean(X$Age, na.rm = TRUE)
sdAge <- sd(X$Age, na.rm = TRUE)

X$AgePred <- (XpredAge * sdAge) + meanAge
Xt$AgePred <- (XpredAget * sdAge) + meanAge

plotHist (X, "Age", "AgePred")
plotHist (Xt, "Age", "AgePred")


# Fare
XpredFare <- X$Fare
XpredFaret <- predictedMissingValuesTest[["Fare"]]

meanFare <- mean(X$Fare, na.rm = TRUE)
sdFare <- sd(X$Fare, na.rm = TRUE)

XpredFaret <- (XpredFaret * sdFare) + meanFare

X$FarePred <- XpredFare
Xt$FarePred <- XpredFaret

plotHist (Xt, "Fare", "FarePred")

toRm <- c(toRm, "Fare", "Age")


# observe what is happening with age ranges
addAgeRange <- function(X) {
  
  X$AgeRange[X$AgePred <= 1] <- 0
  X$AgeRange[X$AgePred > 1 & X$AgePred <= 8] <- 2
  X$AgeRange[X$AgePred > 8] <- 3
  X$AgeRange[is.na(X$AgePred)] <- 7
  
  X$AgeRange <- as.numeric(X$AgeRange)
  
  return (X)
}

X <- addAgeRange (X)
Xt <- addAgeRange (Xt)

showBarPlot("AgeRange")

# the fact that the age is known might tell us something
addAgeKnown <- function(X) {
  X$AgeKnown <- 1
  X$AgeKnown[is.na(X$Age)] <- 0
  return(X)
}

X <- addAgeKnown (X)
Xt <- addAgeKnown (Xt)

showBarPlot("AgeKnown", stack=TRUE, normalize = TRUE, legendPos = "bottomright")


#xxxxxxxxxxxxxxxxxxxxxxxx
# X$Name

addTitle <- function(X) {
  X$Title <- sapply(X$Name, function (x) {
    str <- as.character(x)
    title <- str_extract(str, ", ([A-Za-z]+)\\.")
    title <- substr(title, nchar(", ") + 1, nchar(title) - 1)
    return (title)
  })
  # some translations
  X$Title[X$Title %in% c("Don", "Rev", "Major", "Sir", "Col", "Capt", "Jonkheer")] <- "Mr"
  X$Title[X$Title %in% c("the Countess", "Mlle", "Ms")] <- "Miss"
  X$Title[X$Title %in% c("Mme", "Lady", "Dona")] <- "Mrs"
  
  X$Title[is.na(X$Title) & X$Sex == "male"] <- "Mr"
  X$Title[is.na(X$Title) & X$Sex == "female" & X$SibSp == 0] <- "Miss"
  X$Title[is.na(X$Title) & X$Sex == "female" & X$SibSp > 0] <- "Mrs" # assuming it is not a sibling...
  
  # in 1912, most doctors were men...
  X$Title[X$Title %in% c("Dr") & X$Sex == "male"] <- "Mr"
  X$Title[X$Title %in% c("Dr") & X$Sex == "female" & X$SibSp == 0] <- "Miss"
  X$Title[X$Title %in% c("Dr") & X$Sex == "female" & X$SibSp > 0] <- "Mrs" # assuming it is not a sibling...
  
  # Master seems to be boys only, we add girls
  #X$Title[X$Title %in% c("Miss") & X$AgePred < 10] <- "Child"
  X$Title[X$Title %in% c("Master")] <- "Child"
  X$Title[X$AgePred < 10] <- "Child"
  #X$Title[X$AgePred > 10 & X$AgePred < 18] <- "Teenage"
  
  X$Title <- as.factor(X$Title)
  return(X)
}

X <- addTitle (X)
Xt <- addTitle (Xt)

showBarPlot("Title")
showBarPlot("Title", stack=TRUE, normalize = TRUE, legendPos = "topleft")

toRm <- c(toRm, "Name")

# redundant with Title, let the feature selector picks
addGender <- function(X) {
  X$Gender[(X$Title %in% c("Miss", "Mrs"))] <- 1
  X$Gender[(X$Title %in% c("Mr"))] <- 2
  X$Gender[(X$Title %in% c("Child"))] <- 3
  return(X)
}

X <- addGender (X)
Xt <- addGender (Xt)

showBarPlot("Gender")
showBarPlot("Gender", stack=TRUE, normalize = TRUE, legendPos = "topleft")


#xxxxxxxxxxxxxxxxxxxxxxxx
# X$Parch & X$SibSp

# we add some features based on those two columns

# Size of families
addFamilyMembers <- function(X) {
  X$FamilyMembers <- X$Parch + X$SibSp
  return (X)
}

X <- addFamilyMembers (X)
Xt <- addFamilyMembers (Xt)

showBarPlot("FamilyMembers", stack=TRUE, normalize = TRUE, legendPos = "bottomright")


# large families seem to be doomed
addLargeFamily <- function(X) {
  X$LargeFamily <- X$Parch + X$SibSp
  
  # bad coding! here order matters... to refactor
  X$LargeFamily[X$LargeFamily < 7] <- 0
  X$LargeFamily[X$LargeFamily >= 7] <- 1
  
  X$LargeFamily <- as.factor(X$LargeFamily)
  
  return (X)
}

X <- addLargeFamily (X)
Xt <- addLargeFamily (Xt)

showBarPlot("LargeFamily", stack=TRUE, normalize = TRUE, legendPos = "bottomright")

# is the passenger alone
addSingle <- function(X) {
  X$Single <- 0
  X$Single[which(X$Parch == 0 & X$SibSp == 0)] <- 1
  return (X)
}

X <- addSingle (X)
Xt <- addSingle (Xt)

showBarPlot("Single", stack=TRUE, normalize = TRUE, legendPos = "bottomright")


# number of childs with a woman passenger
addMotherChildNumber <- function(X) {
  X$MotherChildNumber <- 0
  X$MotherChildNumber[which(X$Parch %in% c(1, 2, 3) & X$AgePred > 20 & X$Title %in% c("Mrs", "Miss"))] <- 1
  X$MotherChildNumber[which(X$Parch > 3 & X$AgePred > 20 & X$Title %in% c("Mrs", "Miss"))] <- 2
  return (X)
}

X <- addMotherChildNumber (X)
Xt <- addMotherChildNumber (Xt)

showBarPlot("MotherChildNumber", stack=TRUE, normalize = TRUE, legendPos = "bottomright")


# remove some columns we think we no longer need
for (feature in toRm) {
  X[[feature]] <- NULL
  Xt[[feature]] <- NULL
}


# Data Split -----------------------------

response <- "Survived"
unselect <- c("PassengerId")
predictors <- setdiff(names(X), c(response, unselect)) 

trainIndexes <- createDataPartition(y = X[[response]], 
                               p = kSplitTrainingValidation, 
                               list = FALSE) 
train <- X[trainIndexes, ] 
valid <- X[-trainIndexes, ] 
test_t <- Xt

trainIndexes <- createDataPartition(y = X[[response]], 
                               p = kSplitTrainingValidationFeaturesSelections, 
                               list = FALSE) 
trainFeatureSelection <- X[trainIndexes, ] 
validFeatureSelection <- X[-trainIndexes, ] 


# Feature selection -----------------

# We rely on a naive greedy algorithm (forward selection) to estimate the 
# "best" set of features, for trying the entire set of possible combinations 
# would take too long (even with such a tiny set of features...)

getGreedyForwardlyBestFeaturesSet <- function (featuresSet, 
                                               trainingFrame, 
                                               response, 
                                               newdata,
                                               scoringMetric = "acc",
                                               acceptSetBack = TRUE,
                                               epsilon = 1e-6,
                                               verbose = TRUE) {
  
  if (length(featuresSet) == 0) {
    return (c())
  }
  validScoringMetric <- c("acc", "auc", "mse", "logloss")
  if (!(scoringMetric %in% validScoringMetric)) {
    stop(paste0("scoring metric can only be one of: ", validScoringMetric))
  }
  
  getScoreValue <- function (scoringMetric, aucValue, accValue, mseValue, loglossValue) {
    ret <- ifelse(scoringMetric == "acc", accValue, 
                  ifelse(scoringMetric == "auc", aucValue, 
                         ifelse(scoringMetric == "mse", mseValue, 
                                ifelse(scoringMetric == "logloss", loglossValue, 
                                       NA))))
    if (is.na(ret)) {
      stop ("Unknown scoring metric...")
    }
    return (ret)
  }
  
  compareScoreValue <- function (scoringMetric, a, b, epsilon) {
    if (scoringMetric %in% c("auc", "acc")) {
      return (a <= b + epsilon)
    } else {
      return (a >= b - epsilon)
    }
  }
  
  initValue <- function (scoringMetric) {
    if (scoringMetric %in% c("auc", "acc")) {
      return (0)
    } else {
      return (Inf)
    }
  }
  
  set.seed(18746635)
  highestAcc <- initValue (scoringMetric = "acc")
  highestAuc <- initValue (scoringMetric = "auc")
  highestMse <- initValue (scoringMetric = "mse")
  highestLogloss <- initValue (scoringMetric = "logloss")
  bestFeatures <- c()
  noImprovmentCount <- 0
  noImprovmentFeatures <- c()
  historyHighestScore <- c()
  historyHighestScoreIndex <- c()
  currIndex <- 0
  historyTmpHighestScore <- c()
  historyTmpHighestScoreIndex <- c()
  historyTmpHighestScoreFeatures <- c()
  finished <- FALSE
  while (!finished) {
    
    bestNewFeatures <- NULL
    
    currIndex <- currIndex + 1
    tmpHighestAcc <- initValue (scoringMetric = "acc")
    tmpHighestAuc <- initValue (scoringMetric = "auc")
    tmpHighestMse <- initValue (scoringMetric = "mse")
    tmpHighestLogloss <- initValue (scoringMetric = "logloss")
    tmpBestNewFeatures <- NULL
    for (feature in setdiff(featuresSet, bestFeatures)) {
      
      currentFeatures <- c (bestFeatures, feature)
      formula <- getFormula (response = response, features = currentFeatures,
                             factorResponse = TRUE)
      labels <- as.data.frame(newdata[,response])[, 1]
      actual <-  factor2Numeric(labels)
      
      rf <- randomForest(formula, data = trainingFrame)
      
      predTest <- predict(rf, newdata = newdata, type = "response")
      predicted <- factor2Numeric(predTest)
      auc <- evalAuc(predictedProb = predict(rf, newdata = newdata, type = "prob")[, 2], 
                     actual = actual)
      
      acc <- evalOutcome (predicted = predicted, actual = actual, metric = "accuracy")
      logloss <- evalOutcome (predicted = predicted, actual = actual, metric = "logloss")
      mse <- evalOutcome (predicted = predicted, actual = actual, metric = "mse")
      
      tmpScoreValue <- getScoreValue (scoringMetric = scoringMetric, 
                                      aucValue = auc,
                                      accValue = acc,
                                      mseValue = mse,
                                      loglossValue = logloss)
      
      tmpHighestScoreValue <- getScoreValue (scoringMetric = scoringMetric, 
                                             aucValue = tmpHighestAuc,
                                             accValue = tmpHighestAcc,
                                             mseValue = tmpHighestMse,
                                             loglossValue = tmpHighestLogloss)
      
      if (compareScoreValue (scoringMetric = scoringMetric, 
                             a = tmpHighestScoreValue, b = tmpScoreValue, 
                             epsilon = 0)) {
        tmpHighestAcc <- acc 
        tmpHighestAuc <- auc
        tmpHighestMse <- mse 
        tmpHighestLogloss <- logloss
        tmpBestNewFeatures <- feature
      }
      if (verbose) {
        print(paste0("Current feature: ", feature, 
                     " (", scoringMetric, ": ", format(tmpScoreValue, digits = 4), "; ", 
                     "best score: ", format(getScoreValue (scoringMetric = scoringMetric, 
                                                           aucValue = highestAuc,
                                                           accValue = highestAcc,
                                                           mseValue = highestMse,
                                                           loglossValue = highestLogloss), digits = 4), ")"))
        print (currentFeatures)
      }
    }
    historyTmpHighestScore <- c(historyTmpHighestScore, 
                                getScoreValue (scoringMetric = scoringMetric, 
                                               aucValue = tmpHighestAuc,
                                               accValue = tmpHighestAcc,
                                               mseValue = tmpHighestMse,
                                               loglossValue = tmpHighestLogloss))
    
    historyTmpHighestScoreFeatures <- c(historyTmpHighestScoreFeatures, tmpBestNewFeatures)
    historyTmpHighestScoreIndex <- c(historyTmpHighestScoreIndex, currIndex)
    
    highestScoreValue <- getScoreValue (scoringMetric = scoringMetric, 
                                        aucValue = highestAuc,
                                        accValue = highestAcc,
                                        mseValue = highestMse,
                                        loglossValue = highestLogloss)
    
    highestTmpScoreValue <- getScoreValue (scoringMetric = scoringMetric, 
                                           aucValue = tmpHighestAuc,
                                           accValue = tmpHighestAcc,
                                           mseValue = tmpHighestMse,
                                           loglossValue = tmpHighestLogloss)
    
    if (compareScoreValue (scoringMetric = scoringMetric, 
                           a = highestScoreValue, b = highestTmpScoreValue, 
                           epsilon = epsilon)) {
      highestAcc <- tmpHighestAcc 
      highestAuc <- tmpHighestAuc
      highestMse <- tmpHighestMse
      highestLogloss <- tmpHighestLogloss
      bestNewFeatures <- tmpBestNewFeatures
      
      historyHighestScore <- c(historyHighestScore, 
                               getScoreValue (scoringMetric = scoringMetric, 
                                              aucValue = highestAuc,
                                              accValue = highestAcc,
                                              mseValue = highestMse,
                                              loglossValue = highestLogloss))
      historyHighestScoreIndex <- c(historyHighestScoreIndex, currIndex)
    }
    if (is.null(bestNewFeatures)) {
      if (!acceptSetBack || noImprovmentCount > 3) {
        # no further improvement
        bestFeatures <-  setdiff(bestFeatures, noImprovmentFeatures)
        finished <- TRUE
        break
      } else {
        noImprovmentCount <- noImprovmentCount + 1
        noImprovmentFeatures <- c(noImprovmentFeatures, tmpBestNewFeatures)
        
        bestFeatures <- c(bestFeatures, tmpBestNewFeatures)
      }
    } else {
      
      bestFeatures <- c(bestFeatures, noImprovmentFeatures)
      
      noImprovmentCount <- 0
      noImprovmentFeatures <- c()
      
      if (verbose) {
        print(paste0("Best feature: ", bestNewFeatures, " (", 
                     scoringMetric, ": ", 
                     getScoreValue (scoringMetric = scoringMetric, 
                                    aucValue = highestAuc,
                                    accValue = highestAcc,
                                    mseValue = highestMse,
                                    loglossValue = highestLogloss), ")"))
      }
      bestFeatures <- c(bestFeatures, bestNewFeatures)
    }
    if (length(setdiff(featuresSet, bestFeatures)) == 0) {
      bestFeatures <-  setdiff(bestFeatures, noImprovmentFeatures)
      finished <- TRUE
      break 
    }
  }
  if (verbose) {
    print(bestFeatures)
    print(paste0("acc: ", highestAcc, ", auc: ", highestAuc, 
                 ", mse: ", highestMse, ", logloss: ", highestLogloss))
  }
  
  drawfeaturesSelectionResult <<- function(historyTmpHighestScore, 
                                           historyTmpHighestScoreIndex, 
                                           historyHighestScore, 
                                           historyHighestScoreIndex, 
                                           historyTmpHighestScoreFeatures, 
                                           ylab = "", 
                                           legendPos = "bottomright") {
    ylim <- c(min(historyTmpHighestScore), max(historyTmpHighestScore))
    xlim <- c (min(historyTmpHighestScoreIndex), max(historyTmpHighestScoreIndex))
    cex <- 0.9
    inches2lines <- (par("mar") / par("mai"))[1] #https://stackoverflow.com/questions/18040240/find-optimal-width-for-left-margin-in-r-plot
    xmargin <- max(strwidth(historyTmpHighestScoreFeatures, 
                            units="inches", 
                            cex = cex)) * inches2lines
    op <- par(mar = c(1 + xmargin, 4, 4, 2) + 0.1)
    plot (x = historyHighestScoreIndex, y = historyHighestScore, 
          type = "o", col="red", lwd=3, ylim = ylim, xlim = xlim, xaxt='n', xlab="", ylab=ylab,
          main = "Greedy Forward Selection of Features")
    par(new=TRUE)
    plot (x = historyTmpHighestScoreIndex, y = historyTmpHighestScore, 
          type = "o", col="green", xaxt='n', yaxt='n', ann=FALSE, ylim = ylim, xlim = xlim, xlab="")
    axis(1, 
         at = seq_along(historyTmpHighestScore), 
         labels = as.character(historyTmpHighestScoreFeatures), 
         las = 2, cex.axis = cex)
    abline(v = max (historyHighestScoreIndex), col = "black", lty = 2)
    # pos: 1=below, 2=left, 3=above, 4=right
    ypos <- (ylim[1] + ylim[2]) / 2
    if (grepl("^top", legendPos)) {
      ypos <- ylim[1] + (ylim[2] - ylim[1]) * 0.9
    } else if (grepl("^bottom", legendPos)) {
      ypos <- ylim[1] + (ylim[2] - ylim[1]) * 0.1
    }
    rotation <- 0 # 90
    text(x = max (historyHighestScoreIndex), y = ypos, "Selected", pos = 2, srt = rotation, col = "firebrick4")
    text(x = max (historyHighestScoreIndex), y = ypos, "Ignored", pos = 4, srt = rotation, col = "forestgreen")
    par(op)
  }
  
  # "export" those variables to the global context,
  # so that we can redraw the graph easily at the end
  historyHighestScore <<- historyHighestScore
  historyTmpHighestScore <<- historyTmpHighestScore
  
  historyHighestScoreIndex <<- historyHighestScoreIndex
  historyTmpHighestScoreIndex <<- historyTmpHighestScoreIndex
  historyTmpHighestScoreFeatures <<- historyTmpHighestScoreFeatures
  
  legendPos <- "bottomright"
  if (scoringMetric %in% c ("mse", "logloss")) {
    legendPos <- "topright"
  }
  drawfeaturesSelectionResult (historyTmpHighestScore = historyTmpHighestScore, 
                               historyTmpHighestScoreIndex = historyTmpHighestScoreIndex, 
                               historyHighestScore = historyHighestScore, 
                               historyHighestScoreIndex = historyHighestScoreIndex,
                               historyTmpHighestScoreFeatures = historyTmpHighestScoreFeatures,
                               ylab = scoringMetric, legendPos = legendPos)
  
  return (bestFeatures)
}


predictors <- getGreedyForwardlyBestFeaturesSet (featuresSet = predictors, 
                                                 trainingFrame = trainFeatureSelection, 
                                                 response = response,
                                                 newdata =  validFeatureSelection, 
                                                 scoringMetric = kFeaturesSelectionScoringMetric,
                                                 acceptSetBack = TRUE, epsilon = 0.000001)

# RF ----------------------------

hyperParams <- list ()

hyperParams$ntree <- c(50, 100, 200, 400, 500)
hyperParams$replace <- c (TRUE)
hyperParams$nodesize <- c(1, 2, 3, 4, 5)
hyperParams$maxnodes <- c(12)
hyperParams$mtry <- c(2)

grideSize <- prod (lengths(hyperParams))

formula <- getFormula (response = response, features = predictors)

set.seed(1)
validAccs <- c()
oobs <- c()
predictions <- list()
predIndexes <- c()
count <- 0

grid <- expand.grid(hyperParams)
for(i in 1:nrow(grid)) {
  row <- grid[i, ]

  rf <- randomForest(formula, data = train,
                     ntree = row$ntree, nodesize = row$nodesize,
                     mtry = row$mtry, replace = row$replace,
                     maxnodes = row$maxnodes)
  
  predValid <- predict(rf, newdata = valid)
  validAcc <- mean(valid$Survived == predValid)
  
  predTest <- predict(rf, newdata = test_t)
  
  validAccs <- c (validAccs, validAcc)
  oobs <- c(oobs, mean (rf$err.rate[, "OOB"]))
  predIndexes <- c(predIndexes, i)
  
  predictions[[paste0(i)]] <- predTest
  
  count <- count + 1
  
  if (count == 1) {
    print("Processing...")
  }
  
  if (count == grideSize) {
    cat(": Done")
    cat('\n')
  } else {
    cat('\r')
    cat(paste0(round(count / grideSize * 100), "% completed"))
  }
}

# add result columns to the grid
grid$validAcc <- validAccs
grid$oob <- oobs
grid$index <- predIndexes

# sort the grid
ordering <- order(grid$validAcc, -grid$oob, decreasing = TRUE)
grid <- grid[ordering, ]
grid

# save the best result
pathExt <- paste0("-", Sys.Date(), ".csv")
tryCatch({
  
  if(!file.exists(kTmpFolder)){
    dir.create(kTmpFolder, showWarnings = FALSE, recursive = TRUE)
  }
  
  saveOutPutCsv (Xt$PassengerId, factor2Numeric(predictions[[grid[1, ]$index]]), 
                 file.path(kTmpFolder, 
                           paste0("result", pathExt), 
                           fsep = .Platform$file.sep) )
}, error = function(e) {
  print(e)
})


