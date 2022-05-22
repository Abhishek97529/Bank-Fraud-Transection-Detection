# Bank-Fraudulent-Transection-Detection
## The challenge is to recognize fraudulent transactions so that the customers of Bank do not suffer for the mistake they not have done. 
## Main challenges involved in fraudulent Transaction detection are:  
### 1.)Enormous Data is processed every day and the model build must be fast enough to respond to the scam in time.
### 2.)Imbalanced Data i.e most of the transactions (99.8%) are not fraudulent which makes it really hard for detecting the fraudulent ones
### 3.)Data availability as the data is mostly private.
### 4.)Misclassified Data can be another major issue, as not every fraudulent transaction is caught and reported. 
### 5.)Adaptive techniques used against the model by the scammers.
### 6.)Too many features are there in the Dataset so feature selection is a problem

## ALGORITHMS TESTED
LogisticRegression,
SVC,
DecisionTreeClassifier,
KNeighborsClassifier,
RandomForestClassifier,
AdaBoostClassifier,
BaggingClassifier,
ExtraTreesClassifier,
GradientBoostingClassifier,

## BEST PERFORMANCE
## RandomForestClassifier
![image](https://user-images.githubusercontent.com/91778239/169697912-90fd22d2-7ef2-4d59-8987-5f2df4a485b0.png)

## Used SMOTE for high-dimensional class-imbalanced data
SMOTE is an oversampling technique that generates synthetic samples from the minority class. It is used to obtain a synthetically class-balanced or nearly class-balanced training set, which is then used to train the classifier.

Classification using class-imbalanced data is biased in favor of the majority class. The bias is even larger for high-dimensional data, where the number of variables greatly exceeds the number of samples. The problem can be attenuated by undersampling or oversampling, which produce class-balanced data. Generally undersampling is helpful, while random oversampling is not. Synthetic Minority Oversampling TEchnique (SMOTE) is a very popular oversampling method that was proposed to improve random oversampling but its behavior on high-dimensional data has not been thoroughly investigated. In this paper we investigate the properties of SMOTE from a theoretical and empirical point of view, using simulated and real high-dimensional data.
![image](https://user-images.githubusercontent.com/91778239/169697831-437d31b1-8f57-4860-8380-8a942d0b6737.png)

## Understand Weight of Evidence and Information Value for Feature Selection
### Weight of Evidence-
The formula to calculate the weight of evidence for any feature is given by

![image](https://user-images.githubusercontent.com/91778239/169698268-6c63c612-50a9-4cad-87a9-fe6f03863a79.png)

let us take a dummy example:

![image](https://user-images.githubusercontent.com/91778239/169698236-498cb588-9ef8-4fe6-94be-4f5c3823ed0a.png)

### Information Value-
The WoE value tells us the predictive power of each bin of a feature.However, a single value representing the entire feature’s predictive power will be useful in feature selection.

![image](https://user-images.githubusercontent.com/91778239/169698414-85a5aa06-cc65-4f8f-b60c-e5a072826deb.png)

Note that the term (percentage of events – the percentage of non-events) follows the same sign as WoE hence ensuring that the IV is always a positive number.

How do we interpret the IV value?

The table below gives you a fixed rule to help select the best features for your model

Information Value	Predictive power

< 0.02	Useless

0.02 to 0.1	Weak predictors

0.1 to 0.3	Medium Predictors

0.3 to 0.5	Strong predictors

> 0.5	Suspicious


## Accuracy, Precision, Recall & F1 Score: Interpretation of Performance Measures

![image](https://user-images.githubusercontent.com/91778239/169699302-203e5cd7-a53e-4378-a2e1-d59fe77d1647.png)

Accuracy - Accuracy is the most intuitive performance measure and it is simply a ratio of correctly predicted observation to the total observations. One may think that, if we have high accuracy then our model is best. Yes, accuracy is a great measure but only when you have symmetric datasets where values of false positive and false negatives are almost same. Therefore, you have to look at other parameters to evaluate the performance of your model. For our model, we have got 0.803 which means our model is approx. 80% accurate.

### Accuracy = TP+TN/TP+FP+FN+TN

Precision - Precision is the ratio of correctly predicted positive observations to the total predicted positive observations. The question that this metric answer is of all passengers that labeled as survived, how many actually survived? High precision relates to the low false positive rate. We have got 0.788 precision which is pretty good.

### Precision = TP/TP+FP

Recall (Sensitivity) - Recall is the ratio of correctly predicted positive observations to the all observations in actual class - yes. The question recall answers is: Of all the passengers that truly survived, how many did we label? We have got recall of 0.631 which is good for this model as it’s above 0.5.

### Recall = TP/TP+FN

F1-score - F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account. Intuitively it is not as easy to understand as accuracy, but F1 is usually more useful than accuracy, especially if you have an uneven class distribution. Accuracy works best if false positives and false negatives have similar cost. If the cost of false positives and false negatives are very different, it’s better to look at both Precision and Recall. In our case, F1 score is 0.701.

### F1-Score = 2*(Recall * Precision) / (Recall + Precision)


## ROC Curve and AUC

What is the AUC - ROC Curve?

AUC - ROC curve is a performance measurement for the classification problems at various threshold settings. ROC is a probability curve and AUC represents the degree or measure of separability. It tells how much the model is capable of distinguishing between classes. Higher the AUC, the better the model is at predicting 0 classes as 0 and 1 classes as 1. By analogy, the Higher the AUC, the better the model is at distinguishing between patients with the disease and no disease.

The ROC curve is plotted with TPR against the FPR where TPR is on the y-axis and FPR is on the x-axis.

![image](https://user-images.githubusercontent.com/91778239/169699934-bcccfd9b-b050-435e-a13a-37a8c6792861.png)

Defining terms used in AUC and ROC Curve.

TPR (True Positive Rate) / Recall /Sensitivity-

![image](https://user-images.githubusercontent.com/91778239/169699966-273b0a35-9e36-419b-a619-f2065f8b1b92.png)

Specificity-

![image](https://user-images.githubusercontent.com/91778239/169700068-94d9f081-70ae-49c6-a968-b06a0d7a5570.png)

FPR-

![image](https://user-images.githubusercontent.com/91778239/169700080-fce45cb3-5c04-48ef-8a30-2a53224f24fd.png)

How to speculate about the performance of the model?

An excellent model has AUC near to the 1 which means it has a good measure of separability. A poor model has an AUC near 0 which means it has the worst measure of separability. In fact, it means it is reciprocating the result. It is predicting 0s as 1s and 1s as 0s. And when AUC is 0.5, it means the model has no class separation capacity whatsoever.












