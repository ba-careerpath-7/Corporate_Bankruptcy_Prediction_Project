# 👔 Corporate Bankruptcy Prediction Project

---

## 1. ⭐ What is this Corporate Bankruptcy Project about?

This project is a deep-dive into financial risk assessment. Using a dataset of over 95 financial predictors, I built and compared several Machine Learning (ML) models to see if we can catch the "warning signs" of corporate failure before it happens. 

---

## 2. 💵 The Business Problem: The cost of missing a Warning Sign.

* For banks and investors, a bankrupt company isn't just a loss. It's a "missed signal." Often, bankrupt firms are rare (imbalanced data), making them hard to spot.

* My goal was to create a model that helps predict or locate bankrupt firms.

* The warning signs would be certain values from the 95 predictors.




---
## 3. 💡 Key Insights and Final Conclusions

### The Challenges:
* The first challenge of this project is that we have 95 predictors, so doing regular Exploratory Data Analysis of comparing the respone variable of bankrupt status and each predictor is not efficient.

* The second challenge of this project is that there are only about 3.2% bankrupt companies. So when making a training set and testing set, there is a scenario where the testing set contains all the bankrupt companies and the training set has none of them. One way to avoid this circumstance is by using **stratified cross validation**.

* **Stratified Cross Validation (CV)** example: Before the  make the Stratified CV makes folds, look at the 5 bankrupt companies and make sure they are distributed evenly with the 100 companies."

Fold 1: 20 companies (1 bankrupt)

Fold 2: 20 companies (1 bankrupt)

Fold 3: 20 companies (1 bankrupt)

Fold 4: 20 companies (1 bankrupt)

Fold 5: 20 companies (1 bankrupt)

Now, every single training set and testing set has exactly the same proportion of 5% of the minority class. 



### The "Safe Zone" and "Danger Zone": 
Using 3D PCA plots, I successfully visualized a "Safe Zone" and a "Danger Zone," providing a clear spatial representation of how healthy and failing firms differ.



[insert some 3D PCA plots here]



**1. The Stability Cluster (The Safe Zone):**

Most non-bankrupt companies tend to be in the cloud of points. These companies are considered the "Safe Zone".

**2. The Outlier Signal (The Danger Zone):**

Meanwhile companies outside of this cloud are in danger of being bankrupt! They are in the "danger zone". (Notice that many bankrupt companies are in the outer part of the cloud, but not in the inner core of the cloud.)

Most of the bankrupt companies appear to be in these spatial coordinates:
* PC1: -20 to 0
* PC2: -20 to 20
* PC3: -20 to 20 

It appears that negative PC1 values are the most important coordinate in determining if a company is bankrupt. Some bankrupt companies are inside the stability cloud and yet if they have negative PC1 values, they could still bankrupt.

The inner core of the cloud and positve PC1 values have virtually not bankrupt companies.  

(NOTE: If you want a interactive 3D PCA plot to view the bankrupt companies in detail, please view the project in the Google Colab Link in the start of the project.) 

### Operational Application: Real Time Risk Screening:

There can be a scenario where we have new companies that do not have a bankrupt label, but have the information of the predictor variables. We add them to the data set and perform PCA.

**Internal Mapping:** For any new companies that are inside the safe zone, we can infer that they are fiancially stable companies. Their 95 predictor values align with successful firms.

**External Mapping:** In contrast, for any new companies that are or drifting in the danger zone, we can deduce that they are in risk of being bankrupt. They should be flagged for financial investigation or considered for divestment.

(Divestment is the opposite of investment. Investment is when someone buys stock or property of a company. Divestment is when someone sells stock or property of a company.) 


### The Quantitative Limit:
XGBoost did a good job of predicting bankrupt companies, but it is not perfect. 


[Inser bar plot of all classification models]


[insert the confusion matrix plot]


[insert pr auc plot]



### The Human Element:
Real-world bankruptcy is often driven by qualitative shifts (like the Blockbuster/Netflix case) that tabular data cannot see. A truly robust strategy must combine Machine Learning with Contextual Analysis.




---
## 4. 📔 The Methodology of what I did: 

### Firstly, I did exploratory data analysis.

Plots of predictor variables against response variables were made.
Additionally, predictor variables against other predictor variables were made.

### Secondly, ML models and plots were created.
  
* **Stack:** Python (NumPy, pandas, matplotlib, seaborn, scikit-learn, XGBoost, plotly, SciPy)

### Thirdly, I tried to gather insights about how to detect bankrupt companies and any reasons companies can be bankrupt.

For a refresher, check them out at [point 3!](#3--key-insights-and-final-conclusions) 


---
## 5. 💻 Technical Log 

### Un-Supervised 🧩:
1. PCA
2. LLE 


### Classification 📂:
1. Logistic Regression
2. Logistic Ridge
3. Logistic LASSO
4. Random Forest
5. XGBoost
6. KNN using PCA 


