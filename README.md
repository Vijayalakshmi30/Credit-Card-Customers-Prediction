# Credit-Card-Customers-Prediction

**Problem:** 
A manager at the bank is disturbed with more and more customers leaving their credit card services. They would really appreciate if one could predict for them who is gonna get churned so they can proactively go to the customer to provide them better services and turn customers' decisions in the opposite direction.

A business manager of a consumer credit card portfolio is facing the problem of customer attrition. They want to analyze the data to find out the reason behind this and leverage the same to predict customers who are likely to drop off.

Attrited customers are the group of customers that has been lost through time for any reason. The process of losing a client is referred to as 
customer attrition, also known as customer churn.

**Goal:** To predict churning customers

**Report:**

*From data exploration:*

1.Most of the customers have churned after being inactive for 2 or 3 months.So, target on the people who are inactive for 2-3 months. As they're likely to be attrited. 

2.The clients who held the blue card have churned the most. So, it's recommended to change the benefits of the card making favourable to the customers.1519 customers owning blue card have been churned.

3.Most of the clients having 0 Revolving Balance are likely to be churned.About 900 customers with 0 revolving balance have quitted the credit card service.

4.People with income category $80k - $120k are given more credit limit and thereby we can see more existing customers in this category but still some of them have churned the credit card service.

5.People with age group of 40-55 is more likely to quit the credit card service, so have to turn focus on them too.

Approach to following groups of people to minimize the churning rate to some extent.

Card category: Blue

Age: 40-55

Income : Less than $40K

Blue card holders

**From Clustering:**

K Means seemed to get relatively better clusters on the data. HAC placed most of the data in the second cluster while struggling to cluster Attrited customers.

**From Classifiers:**

KNN seems to be good as it has high accuracy but it is misleading due to the class imbalance problem in data as there were more existing customers than credit card service churned customers

**From Evaluation:**

Accuracy, Balnaced Accuracy and AUC values are good. Intolerable error from the confusion matrix is that 151 Attrited customers were wrongly predicted as Existing customers. 
 
