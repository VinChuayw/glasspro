library(mlbench)
library(dplyr)
library(caret)
library(ggplot2)
library(corrplot)
library(class)
data(Glass)
str(Glass)
summary(Glass)
#convert the Type variable into yType factor with only 2 levels and select only Type 1 and 2 
Glass1 <- subset(Glass, Type==1 | Type ==2)
Glass1 <- Glass1%>%mutate(yType=factor(ifelse(Type=="2", 1,0)))%>%select(RI:Fe, yType)

########### Prediction ###############
###########1. kNN regression ###############
#form Glass2 for finding the correlation matrix between RI and other variables
Glass2 <- Glass%>%select(RI:Fe)
#split data into training and data sets
set.seed(100)
training.idx <- sample(1: nrow(Glass2), size=nrow(Glass2)*0.8)
train.data <-Glass2[training.idx, ]
test.data <- Glass2[-training.idx, ]
# Fit the model on the training set
set.seed(101)
model <- train(
  RI~., data = train.data, method = "knn",
  trControl = trainControl("cv", number = 10),
  preProcess = c("center","scale"),
  tuneLength = 10
)
# Plot model error RMSE vs different values of k
plot(model)
model$bestTune
predictions <-predict(model, test.data)
head(predictions)
#1.516316 1.518658 1.517638 1.517798 1.517446 1.516818
#Compute the prediction error RMSE
RMSE(predictions, test.data$RI)
#0.002189306 
#visualize the performance of kNN reg,we plot predictedRI vs RI in the test data
par(mfrow=c(1,1))
plot(test.data$RI, predictions,main="Prediction performance of kNN regression")
#add a reference line
fit <- lm(predictions~test.data$RI,data = Glass2)
abline(fit, col="red")

###########2. Linear regression ###############
lmodel<-lm(RI~., data = train.data)
#print details of model fitting
summary(lmodel)
# Make predictions on the test data
predictions <-predict(lmodel, test.data)
RMSE(predictions, test.data$RI)
#0.001283655
plot(test.data$RI, predictions,main="Prediction performance of linearregression")
#add a reference line 
fit <- lm(predictions~test.data$RI,data = Glass2)
abline(fit, col="red")
#calculate residuals
residuals(lmodel)
#create multiple plots on the same page
par(mfrow=c(2,2))
plot(lmodel)
#Visualize the correlation between the outcome RI and each predictor
par(mfrow=c(1,1))
corrplot(cor(train.data),type="upper",method="color",addCoef.col="black",number.cex=0.8)
#remove outliers from the training data
Glass1<-Glass1[-c(48,125,188,173,202,208),]
set.seed(100)
training.idx <-sample(1: nrow(Glass1), size=nrow(Glass1)*0.8)
train.data  <-Glass1[-training.idx,]
test.data <-Glass1[-training.idx, ]
p2model<-lm(RI~Na+Mg+Al+Si+K+Ca+Ba+Fe+I(Ca^2)+I(Si^2),data = train.data)
#print details of model fitting
summary(p2model)
# Make predictions on the test data
predictions <-predict(p2model, test.data)
RMSE(predictions, test.data$RI)
#0.0002637537
#create multiple residual plots on the same page
par(mfrow=c(2,2))
plot(p2model)

########### Data Visualisation with boxplot ###############
ggplot(Glass2, aes(x=Ca, y=RI))+geom_point()+geom_smooth(method='lm')
ggplot(Glass1, aes(x=yType, y=RI)) +geom_boxplot()

########### Classification ###############
###########1. Logistic regression ###############
set.seed(100)
training.idx <- sample(1: nrow(Glass1), size=nrow(Glass1)*0.8)
train.data <-Glass1[training.idx, ]
test.data <- Glass1[-training.idx, ]
mlogit <- glm(yType~RI+Na+Mg+Al+Si+K+Ca+Ba+Fe, data = train.data,family = "binomial")
summary(mlogit)
Pred.p <-predict(mlogit, newdata =test.data, type = "response")
y_pred_num <-ifelse(Pred.p > 0.5, 1, 0)
y_pred <-factor(y_pred_num, levels=c(0,1))
#Accuracy of the classification
mean(y_pred ==test.data$yType )
#0.7931034
#Create the confusion matrix with row-y_pred col=y
table(y_pred,test.data$yType)

###########2. kNN classification ###############
#Normalize numeric variables
nor <-function(x) { (x -min(x))/(max(x)-min(x)) }
Glass1[,1:9] <- sapply(Glass1[,1:9], nor)
#split data
set.seed(100)
training.idx <- sample(1: nrow(Glass1), size=nrow(Glass1)*0.8)
train.data <-Glass1[training.idx, ]
test.data <- Glass1[-training.idx, ]
#kNN classification
set.seed(101)
knn1<-knn(train.data[,1:9], test.data[,1:9], cl=train.data$yType, k=5)
mean(knn1 ==test.data$yType)
#0.8275862
table(knn1,test.data$y)
#try different k to find the best classfier
ac<-rep(0, 30)
for(i in 1:30){
  set.seed(101)
  knn.i<-knn(train.data[,1:9], test.data[,1:9], cl=train.data$yType, k=i)
  ac[i]<-mean(knn.i ==test.data$yType)
  cat("k=", i, " accuracy=", ac[i], "\n")
}
#Accuracy plot
par(mfrow=c(1,1))
plot(ac, type="b", xlab="K",ylab="Accuracy")
#k=5 best 

########### Clustering ###############
###########1. K-means clustering ###############
k2 <- kmeans(Glass, centers = 2, nstart = 25)
str(k2)
k2
# function to compute total within-cluster sum of square
wcss <- function(k) {
  kmeans(Glass, k, nstart = 10 )$tot.withinss
}
# Compute and plot wss for k = 1 to k = 15
k.values <- 1:15 
set.seed(100)
# apply wcss to all k values
wcss_k<-sapply(k.values, wcss)
plot(k.values, wcss_k, type="b", pch = 19, frame = FALSE,xlab="Number of clusters K",ylab="Total within-clusters sum of squares")
# 5
#final clustering with k=5
set.seed(100)
k5.final <- kmeans(Glass2, 5, nstart = 25)
k5.final
Glass %>% mutate(Cluster = k5.final$cluster) %>% group_by(Cluster) %>% summarise_all("mean")
