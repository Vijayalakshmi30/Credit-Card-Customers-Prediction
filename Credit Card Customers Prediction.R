bank_churners <- read.csv("Bank_Churners.csv")

library(tidyverse)
bank_churners <- bank_churners %>% select(-c(1,10,22,23)) 
str(bank_churners)

#There are 23 variables out of that 2 are from Naive Bayes, so I'll omit those 2 columns and other 2 variables which are irrelevant to the prediction.
#Total no. of variables: 19
#Number of data points: 10127


#Binary Classification problem
#Target variable: Attrition_Flag
#No. of classes: 2 ("Existing Customer", "Attrited Customer")

#Data Exploration:

#Converting appropriate variables to factors
  
bank_churners$Attrition_Flag <- as.factor(bank_churners$Attrition_Flag)
bank_churners$Gender <- as.factor(bank_churners$Gender)
bank_churners$Education_Level <- as.factor(bank_churners$Education_Level)
bank_churners$Marital_Status <- as.factor(bank_churners$Marital_Status)
bank_churners$Income_Category <- as.factor(bank_churners$Income_Category)
bank_churners$Card_Category <- as.factor(bank_churners$Card_Category)
str(bank_churners)

##Summary statistics:

#5 number summary
  
summary(bank_churners)

#Totally 1627 customers have churned the credit card service
#There are no missing values and will work with outliers using boxplot while cleaning data.

##Visualizations:

#i) Understanding the distribution of each variables

#There are some outliers above age 65.That might be because very few senior citizens have been using credit card.
library(ggplot2)
ggplot(bank_churners, aes(Customer_Age)) + geom_histogram()

#Majority of the clients seemsto have 2 or 3 dependents
ggplot(bank_churners, aes(Dependent_count)) + geom_histogram()

#Most of the customers have 3 products from the bank
ggplot(bank_churners, aes(Total_Relationship_Count)) + geom_histogram()

#Most of the customers being inactive for 3 months
ggplot(bank_churners, aes(Months_Inactive_12_mon)) + geom_histogram()

ggplot(bank_churners, aes(Contacts_Count_12_mon)) + geom_histogram()

#Credit Limit variable is right skewed
ggplot(bank_churners, aes(Credit_Limit)) + geom_histogram()

#Most of the clients have 0 Revolving Balance
ggplot(bank_churners, aes(x=Total_Revolving_Bal)) + geom_histogram()

#Again Avg_Open_To_Buy is having right skewed distribution
ggplot(bank_churners, aes(x=Avg_Open_To_Buy)) + geom_histogram()

#There are some outliers over here
ggplot(bank_churners, aes(x=Total_Amt_Chng_Q4_Q1)) + geom_histogram()

#This distribution is little weird
ggplot(bank_churners, aes(x=Total_Trans_Amt)) + geom_histogram()

#Here, I can see bimodal distribution
ggplot(bank_churners, aes(x=Total_Trans_Ct)) + geom_histogram()

#Again some outliers over here
ggplot(bank_churners, aes(x=Total_Ct_Chng_Q4_Q1)) + geom_histogram()

#Right skewed distribution
ggplot(bank_churners, aes(x=Avg_Utilization_Ratio)) + geom_histogram()

#ii) Getting insights
  
#Most the customers have churned after being inactive for 2 or 3 months. 
p <- ggplot(bank_churners, aes(x=Months_Inactive_12_mon, fill=Attrition_Flag))
p + geom_bar(position="stack")

#The clients who held the blue card have churned the most.
af <- ggplot(bank_churners, aes(x=Card_Category, fill=Attrition_Flag))
af + geom_bar(position="stack")

#About 900 customers with 0 revolving balance have quitted the credit card service
q <- ggplot(bank_churners, aes(x=Total_Revolving_Bal, fill=Attrition_Flag))
q + geom_bar(position="stack")

s <- ggplot(bank_churners, aes(x=Income_Category, fill=Attrition_Flag))
s + geom_bar(position="stack")

unique(bank_churners$Income_Category)

#People with income category $80k - $120k are given more credit limit and thereby we can see more existing customers in this category but still some of them have churned the credit card service.
r <- ggplot(bank_churners, aes(x=Income_Category, y=Credit_Limit, fill=Attrition_Flag))
r + geom_col()

#iii) Understanding relationship between variables
  
# Integers and numeric variables
bint <- bank_churners %>% select(c(2,4,9,10,11,12,13,14,15,16,17,18,19))

# Randomly sampling 1000 data to visualize scatterplot
library(dplyr)
sam <- sample_n(bint, 1000)
plot(sam)

#From the above plot, I've seen following variable relationships:
#Credit_Limit and Avg_Open_To_Buy
#Credit_Limit and Avg_Utilization_Ratio
#Avg_Open_To_Buy and Avg_Utilization_Ratio
#Total_Trans_Amt, Total_Trans_Ct

#Performing scatterplots to find the relationships between variables

ggplot(bint, aes(Credit_Limit, Avg_Open_To_Buy)) + geom_point()

ggplot(bint, aes(Credit_Limit, Avg_Utilization_Ratio)) + geom_point()

ggplot(bint, aes(Avg_Open_To_Buy, Avg_Utilization_Ratio)) + geom_point()

ggplot(bint, aes(Total_Trans_Amt, Total_Trans_Ct)) + geom_point()

#Pearson correlation

cor(bint)

#Strongly Correlated variables:
#-> Credit_Limit and Avg_Open_To_Buy => 0.995980544 (99.59% positively correlated)
#-> Total_Trans_Amt, Total_Trans_Ct => 0.80719203 (80.71% positively correlated)
#-> Total_Revolving_Bal and Avg_Utilization_Ratio => 0.624021991 (62.4% positively correlated)
#-> Credit_Limit and Avg_Utilization_Ratio => -0.482965071 (48.29% negatively correlated)
#-> Avg_Open_To_Buy and Avg_Utilization_Ratio => -0.538807748 (53.88% negatively correlated)

#Cross tabulation to find what kind of groups to concentrate in order to avoid further churn*

#1519 customers owning blue card have been churned.
xt1 <- bank_churners %>% group_by(Card_Category) %>% select(Card_Category, Attrition_Flag) %>% table()
xt1
 
#From the data, age group of 40-55 is more likely to quit the credit card service
xt2 <- bank_churners %>% group_by(Customer_Age) %>% select(Customer_Age, Attrition_Flag) %>% table()
xt2

#People with income less than $40k had attrited the most.
xt3 <- bank_churners %>% group_by(Income_Category) %>% select(Income_Category, Attrition_Flag) %>% table()
xt3

#930 females and 697 males have churned. Since both the numbers are high.
xt4 <- bank_churners %>% group_by(Gender) %>% select(Gender, Attrition_Flag) %>% table()
xt4

xt5 <- bank_churners %>% group_by(Education_Level) %>% select(Education_Level, Attrition_Flag) %>% table()
xt5

xt6 <- bank_churners %>% group_by(Marital_Status) %>% select(Marital_Status, Attrition_Flag) %>% table()
xt6

#I don't think education level and Marital status would affect the churning

#Data Cleaning

#Checking for outliers:
  
boxplot(bank_churners$Avg_Utilization_Ratio)

boxplot(bank_churners$Total_Ct_Chng_Q4_Q1)

boxplot(bank_churners$Total_Trans_Ct)

boxplot(bank_churners$Total_Trans_Amt)

boxplot(bank_churners$Total_Amt_Chng_Q4_Q1)

boxplot(bank_churners$Avg_Open_To_Buy)

boxplot(bank_churners$Total_Revolving_Bal)

boxplot(bank_churners$Credit_Limit)

boxplot(bank_churners$Contacts_Count_12_mon)

boxplot(bank_churners$Total_Relationship_Count)

boxplot(bank_churners$Dependent_count)

boxplot(bank_churners$Customer_Age)

#Removing Outliers
  
#Total_Ct_Chng_Q4_Q1
  
new <- bank_churners
new1 <- new %>% filter(Total_Ct_Chng_Q4_Q1 > 0.25 & Total_Ct_Chng_Q4_Q1 < 1.14)
boxplot(new1$Total_Ct_Chng_Q4_Q1)

#Total_Trans_Ct
  
new2 <- new1 %>% filter(Total_Trans_Ct < 134)
boxplot(new2$Total_Trans_Ct)

#Total_Trans_Amt
  
new3 <- new2 %>% filter(Total_Trans_Amt < 8087)
boxplot(new3$Total_Trans_Amt)

#Total_Amt_Chng_Q4_Q1

new4 <- new3 %>% filter( Total_Amt_Chng_Q4_Q1 < 1.16 & Total_Amt_Chng_Q4_Q1 > 0.28)
boxplot(new4$Total_Amt_Chng_Q4_Q1)

#Avg_Open_To_Buy
  
new5 <- new4 %>% filter(Avg_Open_To_Buy < 6965)
boxplot(new5$Avg_Open_To_Buy)

#Credit_Limit

new6 <- new5 %>% filter(Credit_Limit < 6500)
boxplot(new6$Credit_Limit)

#Contacts_Count_12_mon
  
new7 <- new6 %>% filter(Contacts_Count_12_mon < 5 & Contacts_Count_12_mon > 0.1)
boxplot(new7$Contacts_Count_12_mon)

#Months_Inactive_12_mon
  
new8 <- new7 %>% filter(Months_Inactive_12_mon < 5 & Months_Inactive_12_mon > 0.1)
boxplot(new8$Months_Inactive_12_mon)

#Customer_Age
  
new9 <- new8 %>% filter(Customer_Age<70)
boxplot(new9$Customer_Age)

#Dataset with no outliers
str(new9)

#No. of features: 19
#No. of instances: 4830

summary(new9)

#Data Preprocessing

##Scaling
library(caret)

scaling <- preProcess(new9, method=c("center", "scale"))
std_churn <- predict(scaling, new9)
summary(std_churn)

##Feature Selection

#Converting categorical variables to numerical variable using dummyVars function
dummy <- dummyVars(Attrition_Flag ~ ., data = std_churn)
dummies <- as.data.frame(predict(dummy, newdata = std_churn))
head(dummies)


#Removing near zero variance predictors
nzv <- nearZeroVar(dummies)
nzv

#Checking zero variance columns
  names(dummies)

#The following are zero variance columns:
#6 - "Education_Level.Doctorate"
#16 - "Income_Category.$120K +" 
#22 -  "Card_Category.Blue"  
#23 - "Card_Category.Gold"   
#24 - "Card_Category.Platinum" 
#25 -  "Card_Category.Silver"

#Removing card category
bc_dummy <- dummies %>% select(-c(22,23,24,25))
nzv1 <- nearZeroVar(bc_dummy)
length(nzv1)


#Performing PCA
churn.pca <- prcomp(bc_dummy)
summary(churn.pca)

#Scree plot
screeplot(churn.pca, type = "l") + title(xlab = "PCs")

#Here, the elbow is at 4. So, I'm choosing 4 principal components.

# We don't want to include a prediction target variable in PCA, so we'll separate it

# Create the components
pre_pca <- preProcess(bc_dummy, method="pca", pcaComp=4)
churn.pc <- predict(pre_pca, bc_dummy)
# Put back target column
churn.pc$Attrition_Flag <- std_churn$Attrition_Flag
# Make sure that we have the PCs as predictors
head(churn.pc)

#Clustering

# Removing label

pcwol <- churn.pc %>% select(-c(5))
head(pcwol)

##KMeans Clustering

#Finding k (no. of clusters)

#Knee Plot
library(stats)
library(factoextra)
#From knee plot, no. of clusters = 4
fviz_nbclust(pcwol, kmeans, method = "wss")

#Silhouette plot:
#From silhouette, no. of clusters = 3.
fviz_nbclust(pcwol, kmeans, method = "silhouette")

#I'm going to use 3 clusters for my clustering process
  
#Fitting the data
fit <- kmeans(pcwol, centers = 3, nstart = 25)
fit

#Cluster Plot
fviz_cluster(fit, data = pcwol)

#Displaying PCA plot with clusters expressed in colors**
pca = prcomp(bc_dummy)
rotated_data = as.data.frame(pca$x)
rotated_data$color <- std_churn$Attrition_Flag
ggplot(data = rotated_data, aes(x = PC1, y = PC2, col = color)) + geom_point(alpha = 0.3)

#Displaying plot with cluster results

# Assign clusters as a new column
rotated_data$Clusters = as.factor(fit$cluster)
# Plot and color by labels
ggplot(data = rotated_data, aes(x = PC1, y = PC2, col = Clusters)) + geom_point()

##Hierarchical Clustering

#Dissimilarity matrix
  
library(cluster)
dist_mat2 <- daisy(pcwol, metric = "gower")
# Result is a dissimilarity matrix
summary(dist_mat2)

#Knee plot
#From knee plot, no. of clusters for hierarchical clustering is 3
fviz_nbclust(pcwol, FUN = hcut, method = "wss")

#Silhouette plot
#From silhouette plot, no. of clusters for hierarchical clustering is 3
fviz_nbclust(pcwol, FUN = hcut, method = "silhouette")

#Building HAC Cluster model
hfit <- hclust(dist_mat2, method = 'complete')
h3 <- cutree(hfit, k=3)

#Visualizing HAC Cluster
  
fviz_cluster(list(data = pcwol, cluster = h3))

#Displaying cluster results with PCA projection
  
# Assign clusters as a new column
rotated_data$Clusters = as.factor(h3)
# Plot and color by labels
ggplot(data = rotated_data, aes(x = PC1, y = PC2, col = Clusters)) + geom_point()

#Displaying PCA plot with clusters expressed in colors
ggplot(data = rotated_data, aes(x = PC1, y = PC2, col = color)) + geom_point()

#Comparing the results
  
# Create a dataframe
result <- data.frame(Status = std_churn$Attrition_Flag, HAC3 = h3, Kmeans = fit$cluster)
# View the first 20 cases one by one
head(result, n = 20)

#Clustering results In Cross-tabulation
  
# Crosstab for HAC
result %>% group_by(HAC3) %>% select(HAC3, Status) %>% table()

# Crosstab for K Means
#Here K Means seemed to get relatively better clusters on the data according to both cross tabulation and the one-by-one comparison on the table. HAC placed most of the data in the second cluster while struggling to cluster Attrited customers.
result %>% group_by(Kmeans) %>% select(Kmeans, Status) %>% table()

#Classification:

##SVM Classifier

#Using 10 fold stratified cross validation
folds <- 10

# Generate stratified indices
idx <- createFolds(churn.pc$Attrition_Flag, folds, returnTrain = T)

# Evaluation method
train_control_strat <- trainControl(index = idx, method = 'cv', number =
                                      folds)

# Fit the model
svm_strat <- train(Attrition_Flag ~., data = churn.pc, method = "svmLinear", 
                   trControl = train_control_strat)

# Evaluate fit
svm_strat # Accuracy: 87.68%
  
#Confusion Matrix
confusionMatrix(svm_strat, churn.pca$Attrition_Flag)

#It has wrongly predicted 446 Attrited customers as Existing customers.

#Grid Search
grid <- expand.grid(C = seq(0.001, 2, length = 20))

# Fit the model
svm_stratcv_grid <- train(Attrition_Flag ~., data = churn.pc, method = "svmLinear", 
                          trControl = train_control_strat, tuneGrid = grid)

# View grid search result
svm_stratcv_grid  # Accuracy: 87.78%

##KNN Classifier
set.seed(123)

# 10 fold cross-validation
ctrl <- trainControl(method="cv", number = 10) 
knnFit <- train(Attrition_Flag ~ ., data = churn.pc, 
                method = "knn", 
                trControl = ctrl)

#Output of kNN fit
knnFit   # Accuracy: 89.04%

#Accuracy vs No. of Neighbors
knnFit1 <- train(Attrition_Flag ~ ., data = churn.pc, method = "knn", trControl = ctrl, tuneLength = 10)

plot(knnFit1) # Accuracy is high at k=9

# Accuracy of SVM: 87.78%
# Accuracy of KNN: 89.04%
  
#Evaluation:
#Better classifier: KNN

#Generating Confusion Matrix for KNN
pred_knn<- predict (knnFit, churn.pc)
confusionMatrix(churn.pc$Attrition_Flag, pred_knn)

#151 Attrited customers were wrongly predicted as Existing customers. From the confusion matrix, this error is critical.

#From Confusion Matrix:
#Positive class: Attrited Customers
#Negative class: Existing Customers

#TP: 517     FN: 151
#FP: 287     TN: 3875

#Calculating Precision
  
#Precision = TP / (TP + FP)
#= 517 / (517 + 287)
#= 0.643

#Calculating Recall
#Recall = TP / (TP + FN)
#= 517 / (517 + 151)
#= 0.7739

#ROC Plot
  
#Target class
str(churn.pc$Attrition_Flag)
unique(churn.pc$Attrition_Flag)

library(pROC)

# Get class probabilities for KNN
pred_prob <- predict(knnFit, churn.pc, type = "prob")
head(pred_prob)

# And now we can create an ROC curve for our model.
roc_obj <- roc((churn.pc$Attrition_Flag), pred_prob[,1])
plot(roc_obj, print.auc=TRUE)

#Accuracy: 90.93% 

#Precision: 64.3%
#64.3% of postive class was predicted correctly among all positives

#Recall: 77.39%
#77.39% of actual positives were correctly predicted as positive
#AUC: 0.952 
#This value seems good but it is misleading as the data has class imbalance problem
#Sensitivity : 77.4% (True Positive Rate)
#Specificity : 93.10% (True Negative Rate)
#Balanced Accuracy: 85.25%