# Problem
- Whether water is utilised for drinking, domestic use, food production, or recreational activities, it is crucial for the public's health to have access to safe, readily available water. The decrease of poverty and the expansion of the economy can be significantly aided by better water supply, sanitary conditions, and water resource management. It is well recognised that unsanitary conditions and tainted water contribute to the spread of illnesses like cholera, diarrhoea, dysentery, hepatitis A, typhoid, and polio. Health concerns for those without access to sanitary facilities and clean water can be avoided.

# Goal
- to distinguish between potable and non-potable water using machine learning techniques, and to comprehend what makes water safe to drink.

# Dataset
The dataset contains data of tumor for 569 persons.
- id
- diagnosis	
- radius_mean	
- texture_mean	
- perimeter_mean	
- area_mean	
- smoothness_mean	
- compactness_mean	
- concavity_mean	
- concave 
- points_mean	...	
- texture_worst
- perimeter_worst	
- area_worst	
- smoothness_worst	
- compactness_worst	
- concavity_worst	
- concave 
- points_worst	
- symmetry_worst	
- fractal_dimension_worst

# Models used for training

- **Logistic Regression** - Logistic Regression is named for the function used at the core of the method, the logistic function.
    
    The [logistic function](https://en.wikipedia.org/wiki/Logistic_function), also called the sigmoid function, was developed by statisticians to describe properties of population growth in ecology, rising quickly and maxing out at the carrying capacity of the environment. It’s an S-shaped curve that can take any real-valued number and map it into a value between 0 and 1, but never exactly at those limits.
    
    <p align="left" width="100%">
    <img width="500" height="350" src="https://miro.medium.com/max/640/1*OUOB_YF41M-O4GgZH_F2rw.png">
</p>
    
- **Support Vector Classifier** - The objective of a Linear SVC (Support Vector Classifier) is to fit the data you provide, returning a "best fit" hyperplane that divides, or categorizes your data.

    <p align="left" width="100%">
    <img width="500" height="350" src="http://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1526288453/index3_souoaz.png">
</p>

- **Random Forest Classifier** - A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.

    <p align="left" width="100%">
    <img width="500" height="350" src="https://www.freecodecamp.org/news/content/images/2020/08/how-random-forest-classifier-work.PNG">
</p>

- **XGBoost -** XGBoost is an optimized distributed gradient boosting library designed to be highly **efficient**, **flexible** and **portable**. It implements machine learning algorithms under the [Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting) framework. XGBoost provides a parallel tree boosting (also known as GBDT, GBM).

# Requirements 
### Libraries
The following python libraries were used to perform the various actions on the dataset from loading to preprocessing to visualizing and predicting the results.
 ```
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from itertools import product
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
```
[Numpy](https://github.com/numpy/numpy)- NumPy is the fundamental package needed for scientific computing with Python.

[Pandas](https://github.com/pandas-dev/pandas) - Python library used to analyze data.

[Matplotlib](https://github.com/matplotlib/matplotlib) - Most of the Matplotlib utilities lies under the pyplot submodule.

[Seaborn](https://seaborn.pydata.org/) - An open-source Python library built on top of matplotlib. It is used for data visualization and exploratory data analysis.

