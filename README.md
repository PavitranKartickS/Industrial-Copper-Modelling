# Guvi Capstone Project 5: Industrial-Copper-Modelling

## Introduction:
This Industrial Copper Modelling project mainly focuses on the process of price prediction and status prediction of provided data points of copper based on the data provided along with the problem statement.By exploring the dataset, applying appropriate pre processing techniques we have been able to create the data suitable for machine learning training process.The models will utilize advanced techniques such as data normalization, outlier detection and handling, handling data in the wrong format, and leveraging tree-based models, specifically the decision tree algorithm, to predict the selling price and leads accurately.

## Project Takeaway:
The main takeaways from this project are as follows:

- Exploring Skewness and Outliers: Analyze the distribution of variables in the dataset and identify skewness and outliers. This step helps in understanding the data quality and potential issues that may affect the model performance.

- Data Transformation and Cleaning: Transform the data into a suitable format for analysis and perform necessary cleaning steps. This includes handling missing values, encoding categorical variables, and scaling numerical features.

- Machine Learning Regression Algorithms: Apply various machine learning regression algorithms to predict the selling price of industrial copper. Compare the performance of algorithms such as linear regression, decision trees, random forests, or gradient boosting.

- Machine Learning Classification Algorithms: Apply different machine learning classification algorithms to predict the status (won or lost) of copper transactions. Explore algorithms such as logistic regression, support vector machines, or random forests to classify the outcomes.

- Evaluation and Model Selection: Evaluate the performance of regression and classification models using appropriate metrics such as mean squared error (MSE), accuracy, precision, and recall. Select the best-performing models based on these metrics.


## Requirements:

To run this project, the following libraries are needed:

* NumPy: A library for numerical computations in Python.
* Pandas: A library for data manipulation and analysis.
* Scikit-learn: A machine learning library that provides various regression and classification algorithms.
* Matplotlib: A plotting library for creating visualizations.
* Seaborn: A data visualization library built on top of Matplotlib.


### Regression model details
The copper industry deals with less complex data related to sales and pricing. However, this data may suffer from issues such as skewness and noisy data, which can affect the accuracy of manual predictions. Dealing with these challenges manually can be time-consuming and may not result in optimal pricing decisions. A machine learning regression model can address these issues by utilizing advanced techniques such as data normalization, outlier detection and handling, handling data in wrong format, identifying the distribution of features, and leveraging tree-based models, specifically the decision tree algorithm.

### Classification model details
Another area where the copper industry faces challenges is in capturing the leads. A lead classification model is a system for evaluating and classifying leads based on how likely they are to become a customer. You can use the STATUS variable with WON being considered as Success and LOST being considered as Failure and remove data points other than WON, LOST STATUS values.
