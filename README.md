# Poisonous-Mushroom-Prediction-with-LGBM
Building and leveraging a LGBM model to predict whether a mushroom is edible or poisonous

![image](https://github.com/user-attachments/assets/72c43b0e-7fc6-4027-be4a-1c31523cd043)

---

### Objective
With over 150,000 known species of fungi, any mushroom consumer would likely want to know whether or not a given mushroom is edible or poisonous. The objective of this project is to develop and leverage a LightGBM (LGBM) model to tackle this binary classification task. This project was originally completed on Kaggle, as part of Kaggle's Playground Series Competitions (Season 4, Episode 8). The dataset was provided by Kaggle and was generated from a deep learning model trained on the original UCI Mushroom dataset. The full 'train' dataset includes 22 columns with roughly 3.12 million entries/rows of data. It is important to note that the data has not been pre-cleaned. Thus, another objective of this project is to remedy the large amount of missing/null values found in many of the columns. The target variable 'class' is either 'e' (edible) or 'p' (poisonous). There are 20 feature/predictor variables. 

---

### Methods
Libraries Used
- pandas
- numpy
- seaborn
- matplotlib
- sklearn (train_test_split, accuracy_score, confusion_matrix, classification_report)
- lightgbm (LGBMClassifier, plot_importance)

Data Preprocessing
- Dropping columns with >80% missing/null values
- Filling missing values in object type columns with str 'missing'
- Filling missing values in float type columns with median value
- Converting object type columns to categorical type

Exploratory Data Analysis (EDA)
- Visualizing the distribution of the target variable 'class' with donut and count plots
- Visualizing various categorical variables with donut and count plots
- Visualizing the distribution of numerical variables with kernel density estimation plots
- Visualizing potential correlation amongst numerical variables with heatmap

Building and Training the Model
- Training set (80%), validation set (20%)
- LGBMClassifier parameters
  - Objective: 'binary'
  - Metric: 'binary_error'
  - Num_leaves: 81
  - Learning_rate: 0.05
  - N_estimators: 600
  - Max_depth: 9
  - Random_state: 42
 
  ---

  ### General Results
  - 99.18% accuracy on validation data
  - 0.99 precision, recall, f1-score
  - Confusion matrix
  
  ![image](https://github.com/user-attachments/assets/fa3313ab-510d-4788-937a-c134a2107b0c)
  - Feature Importance
  
  ![image](https://github.com/user-attachments/assets/621278b0-e320-407f-9856-a52b4abc1032)

