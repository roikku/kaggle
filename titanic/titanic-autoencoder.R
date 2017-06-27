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

pkgs <- c("h2o", "h2oEnsemble", "parallel", 
          "corrplot", "car", "heplots", "plyr",
          "stringr", "caret", "dplyr", "gridExtra") 
for (pkg in pkgs) {
  if (! (pkg %in% rownames(installed.packages()))) { install.packages(pkg) }
  require(pkg, character.only = TRUE)
}
Sys.setenv(LANG = "en")
rm(pkgs)
rm(pkg)

tryCatch(
  if (h2o.clusterIsUp()) {
    h2o.shutdown(prompt=FALSE)
    Sys.sleep(5)
  }, error = function(e) {
    
  })

h2o.init(nthreads = parallel:::detectCores(), 
         max_mem_size = "6g", min_mem_size = "1g")
h2o.clusterStatus()


# data ---------------------------

rm(list=ls(all=TRUE))
h2o.removeAll()
if (!is.null(dev.list()["RStudioGD"])) {
  dev.off(dev.list()["RStudioGD"])
}

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
levels(Xt$Survived) <- as.numeric(levels(X$Survived))

# move the response to the end
X <- X[, c(setdiff(names(X),"Survived"), "Survived")]
Xt <- Xt[, c(setdiff(names(Xt),"Survived"), "Survived")]


# Settings -----------------------------

kSplitRatios <- c(0.95)  


# Functions & Others  -----------------------------

saveOutPutCsv <- function (id, pred, file) {
  res <- data.frame(PassengerId=id)
  res[["Survived"]] <- as.numeric(as.vector(pred))
  write.csv(res, file=file, row.names=FALSE,  quote = FALSE)
}

showBarPlot <- function (feature, y="Survived", data = X, legendPos = "topleft", stack = FALSE, normalize = FALSE, labels = c("Died", "Survived")) {
  tb <- table(data[[y]], data[[feature]])
  if (normalize) {
    for (k in 1:dim(tb)[2]) {
      m <- sum(tb[, k])
      tb[1, k] <- tb[1, k] * 100 / m
      tb[2, k] <- tb[2, k] * 100 / m
    }
  }
  barplot(tb,  beside = !stack, legend = FALSE, 
          xlab=feature, ylab="", col=c("firebrick", "darkgreen"))
  if (length(labels) > 0) {
    legend(legendPos, 
           legend = labels, 
           fill = c("firebrick", "darkgreen"))
  }
  return (tb)
}

factor2Numeric <- function (f) {
  return (as.numeric(levels(f))[f])
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
       xlim = c(ymin, ymax), xlab = "Reconstructed")
  hist(as.vector(X[[original]]), breaks = 100, main=original, 
       xlim = c(ymin, ymax), xlab = paste0("Original (", format(sum(is.na(X[[original]])) / dim(X)[1] * 100, digits = 3), "% of missing values)"))
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

opar <- par(mar=c(5, 5.2, 4, 2) + 0.1)
boxplot(FarePred ~ Survived, data = X,  
        xlab = "Survived", ylab = "FarePred\n(outliers removed for visibility)", 
        grid = FALSE, col = c("firebrick", "darkgreen"), outline=FALSE)
par(opar)

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
  X$Gender[(X$Title %in% c("Miss", "Mrs"))] <- "Woman"
  X$Gender[(X$Title %in% c("Mr"))] <- "Man"
  X$Gender[(X$Title %in% c("Child"))] <- "Child"
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


# Convert to h2o frames and split ------------------

unselect <- c("PassengerId")
response <- "Survived"

# H2o is buggy regarding conversion of NA columns...
Xt$Survived <- NA
Xt$Survived[1] <- 1
Xt$Survived[2] <- 0
test <- as.h2o(Xt)
test[, response] <- h2o.asfactor(test[, response])

train <- as.h2o(X)

predictors <- setdiff(names(train), c(response, unselect)) 

trainUnsupervised <- h2o.rbind(train, test) 

splits <- h2o.splitFrame(
  data = train, 
  ratios = kSplitRatios,  
  seed = 1234
)
trainSupervised <- splits[[1]]
validSupervised <- splits[[2]]


# Feature selection -----------------

predictors <- c("Title", "FarePred", "EmbarkedQorS", "TicketNum", "FamilyMembers", "Pclass")  

colToRemove <- setdiff (names(X), c(response, predictors, unselect))
for (col in colToRemove) {
  X[[col]] <- NULL
  Xt[[col]] <- NULL
}
trainUnsupervised <- trainUnsupervised[c(response, predictors)]
trainSupervised <- trainSupervised[c(response, predictors)]


# Autoencoder -------------------------

# Autoencoder to leverage unlabelled data 

hyperParamsAutoencoder = list( 
  hidden = list(c(11, 4, 11), c(10, 4, 10), c(9, 5, 9), c(9, 4, 9), 
                c(7, 4, 7), c(8, 5, 8), c(8, 4, 8), c(8, 3, 8), c(7, 3, 7)),
  activation = c("Tanh") 
)

h2o.rm("grid_autoencoder")
gridAutoencoder <- h2o.grid(
  x = predictors,
  autoencoder = TRUE,
  training_frame = trainUnsupervised,
  hyper_params = hyperParamsAutoencoder,
  search_criteria = list(strategy = "Cartesian"),
  algorithm = "deeplearning",
  grid_id = "grid_autoencoder", 
  reproducible = TRUE, 
  seed = 1,
  variable_importances = TRUE,
  categorical_encoding = "AUTO",
  score_interval = 10,
  epochs = 800,
  adaptive_rate = TRUE,
  standardize = TRUE,
  ignore_const_cols = FALSE)

sortedGridAutoencoder <- h2o.getGrid("grid_autoencoder", sort_by = "mse", decreasing = FALSE)

tmpDf <- as.data.frame(sortedGridAutoencoder@summary_table)
head(tmpDf[, -grep("model_ids", colnames(tmpDf))])

bestAutoencoder <- h2o.getModel(sortedGridAutoencoder@model_ids[[1]])

bestAutoencoderErr <- as.data.frame(h2o.anomaly(bestAutoencoder, 
                                                trainUnsupervised, 
                                                per_feature = FALSE))

plotReconstructionError <- function (error) {
  cut <- 0.5 * sd(error)
  sortedErr <- sort(error)
  sortedErrFrame <- data.frame (index = seq(0, length(sortedErr)-1), error = sortedErr)
  ylim <- c(min(sortedErrFrame$error), max(sortedErrFrame$error))
  xlim <- c(min(sortedErrFrame$index), max(sortedErrFrame$index))
  # could do as in https://stackoverflow.com/questions/11838278/plot-with-conditional-colors-based-on-values-in-r
  plot (x = sortedErrFrame$index[which(sortedErrFrame$error <= cut)], 
        y = sortedErrFrame$error[which(sortedErrFrame$error <= cut)], 
        type = "o", col="forestgreen", lwd=1, ylim = ylim, xlim = xlim, ylab="mse",
        main = "Reconstruction Error",
        xlab = "Sorted Index")
  par(new=TRUE)
  plot (x = sortedErrFrame$index[which(sortedErrFrame$error > cut)], 
        y = sortedErrFrame$error[which(sortedErrFrame$error > cut)],  
        type = "o", col="firebrick3", xaxt='n', yaxt='n', ann=FALSE, ylim = ylim, xlim = xlim, xlab="")
  return (0)
}

#opar <- par(mfrow = c(1, 2))
layout(matrix(c(1,2,2), 1, 3, byrow = TRUE))
plotReconstructionError (bestAutoencoderErr$Reconstruction.MSE)

# https://stackoverflow.com/questions/21858394/partially-color-histogram-in-r
h <- hist(x = bestAutoencoderErr$Reconstruction.MSE, breaks = 100, plot = FALSE)
cuts <- cut(h$breaks, c(-Inf, 0.5 * sd(bestAutoencoderErr$Reconstruction.MSE), Inf))
plot(h, col = c("forestgreen","firebrick3")[cuts], main = "Reconstruction Error", xlab = "mse",
     lty="blank")
#par(opar)


# Deep Learning -----------------------

getDlClassifier <- function (autoencoder, predictors, response, trainSupervised) {
  
  dlSupervisedModel <- h2o.deeplearning(
    y = response, x = predictors,
    training_frame = trainSupervised, 
    pretrained_autoencoder = autoencoder@model_id,
    reproducible = TRUE, 
    balance_classes = TRUE, 
    ignore_const_cols = FALSE,
    seed = 1,
    hidden = autoencoder@parameters$hidden, 
    categorical_encoding = "AUTO",
    epochs = autoencoder@parameters$epochs, 
    standardize = TRUE,
    activation = autoencoder@parameters$activation)
  
  return (dlSupervisedModel)
}

# sequence of thresholds 
kThresholdsSequence <- seq(0.4, 0.6, 0.1)

count <- 1
models <- c()
modelThresholds <- c()
modelAccs <- c()
modelPretrainMse <- c()
modelHidden <- list()
testPreds <- list()
for (i in 1:length(sortedGridAutoencoder@model_ids)) {

  autoencoder <- h2o.getModel(sortedGridAutoencoder@model_ids[[i]])
  dlSupervisedModel <- getDlClassifier (autoencoder, predictors, response, trainSupervised)
  
  pred <- h2o.predict(object = dlSupervisedModel, newdata = validSupervised)

  # try to find an adequate threshold
  finalAcc <- 0
  finalTh <- 0
  for (th in kThresholdsSequence) {
    a <- mean (as.vector(ifelse(pred$p1 < th, 0, 1)) 
               == as.numeric(as.vector(validSupervised[, response]))) 
    if (finalAcc < a) {
      finalAcc <- a
      finalTh <- th
    }
  }

  models <- c(models, dlSupervisedModel)
  modelThresholds <- c(modelThresholds, finalTh)
  modelAccs <- c(modelAccs, finalAcc)
  modelPretrainMse <- c(modelPretrainMse, h2o.mse(autoencoder))
  modelHidden[[paste0(count)]] <- autoencoder@parameters$hidden
  
  predTest <- h2o.predict(object = dlSupervisedModel, newdata = test)
  testPreds[[paste0(count)]] <- as.vector(ifelse(predTest$p1 < modelThresholds[count], 0, 1))
  
  count <- count + 1
}

dfResults <- data.frame (validAcc = modelAccs, 
                         Threshold = modelThresholds, 
                         HiddenLayers = unlist(lapply (modelHidden, function (x) paste0("(", paste(x, collapse=", "), ")"))),
                         PretrainMse = modelPretrainMse,
                         index = seq(1, length (modelAccs), 1))

# sort the results
ordering <- order(dfResults$validAcc, -dfResults$PretrainMse, decreasing = TRUE)
dfResults <- dfResults[ordering, ]
print(head(dfResults), row.names = FALSE)

# save the best result
tryCatch({
  pathExt <- paste0("-", Sys.Date(), ".csv")
  saveOutPutCsv (Xt$PassengerId, testPreds[[dfResults[1, ]$index]], 
                 file.path(".", paste0("result", pathExt), 
                           fsep = .Platform$file.sep) )
}, error = function(e) {
  print(e)
})
