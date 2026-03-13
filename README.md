# 👔 Corporate Bankruptcy Prediction Project
 


---

## 1. ⭐ What is this Corporate Bankruptcy Project about?

This project is a deep-dive into financial risk assessment. Using a dataset of over 90 financial ratios, I built and compared several machine learning models to see if we can catch the "warning signs" of corporate failure before it happens.

---

## 2. 💵 The Business Problem: The cost of missing a Warning Sign.


---
## 3. 💡 Key Insights and Final Conclusions

* The "Zones" of Risk: Using 3D PCA plots, I successfully visualized a "Safe Zone" and a "Danger Zone," providing a clear spatial representation of how healthy and failing firms differ.

* The Quantitative Limit: While XGBoost was highly effective at spotting patterns in the 95 variables, I concluded that ML is not a "silver bullet."

* The Human Element: Real-world bankruptcy is often driven by qualitative shifts (like the Blockbuster/Netflix case) that tabular data cannot see. A truly robust strategy must combine Machine Learning with Contextual Analysis.


---
## 4. 📔 The Methodology of what I did: 

### Firstly, I did exploratory data analysis.

Plots of predictor variables against response variables were made.
Additionally, predictor variables against other predictor variables were made.

### Secondly, Machine Learning models and plots were created.
  
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
4. XGBoost
5. KNN using PCA


