rm(list=ls())

require(rpart)


churndata <- read.table("churndata.csv", sep=",", header=T)
data <- na.omit(churndata)
data$area <- factor(data$area)
nobs <- nrow(data)
data$target <- rep(-1, nobs)                  #first set all target vars to -1 (churn=No)  
data$target[data["churn"] == "Yes"] <- 1      #then set those with churn=Yes to 1
head(data[, c("churn", "target")], 20)        #observe the result
data <- data[, -which(names(data)=="churn")]  #remove old target - we'll use target instead


set.seed(5082)
# Create training and test sets
trainrows <- sample(nrow(data), 0.7*nobs)
train <- data[trainrows, ]
test <- data[-trainrows, ]


# Initialize some vectors
w <- rep(1/length(trainrows), length(trainrows))    #initial weights (all = 1/n) 
m <- vector("list", 1000)                           #List to hold models
e <- c(1:1000)                                      #Error vector
a <- c(1:1000)                                      # Alpha (accuracy) vector

for (i in 1:1000) {
	# Build Model, a simple rpart model with 1 branch (a decision stump)
  # weighted sample
	m[[i]] <- rpart(formula=target ~ ., 
	                data=train, 
	                weights=w/mean(w), 
	                control=rpart.control(maxdepth=1, cp=0),
	                method="class")
	
	# miss is a vector of misclassified indices - 
	# col 2 of prediction is prob that observation is 1 (yes)
	# change logical value to -1 and 1 
	# If yes > 0.5, then 1*2-1 = 1
	# else if yes < 0.5, then 0*2-1 = -1
		
	miss <- which((predict(m[[i]])[, 2]>0.5)*2-1 != train["target"]) 

	
	e[i] <- sum(w[miss])/sum(w)    	  #compute error rate (epsilon)
	
	a[i] <- 0.5*log((1-e[i])/e[i])    #compute alpha
	
	w[miss] <- w[miss]*exp(a[i])      #adjust weights for misses  
	
	if (e[i] > 0.49) {
		print(paste("Error rate of", round(e[i], 4), "encountered after", i, "models built"))
		print("Model model weights are as follows:")
		print(a[1:i])
		break
	}
}

prediction <- rep(0,nrow(test))

for(j in 1:i) {
  m.pred <- (predict(m[[j]], newdata=test)[, 2]>0.5)*2-1
  prediction <- prediction + m.pred * a[j]  # sum weighted models
}

weighted.prediction <- rep(-1, nrow(test))
weighted.prediction[prediction>0] <- 1
mean(weighted.prediction != test$target) #error rate
