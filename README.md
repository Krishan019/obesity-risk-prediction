# Multi-Class Prediction of Obesity Risk
Exploratory Data Analysis (EDA), Different Machine Learning and Boosting Models for Multi-Class Prediction of Obesity Risk

##  Goal:
The goal of this project is to develop a machine learning model capable of accurately predicting obesity risk in individuals by utilising various parameters/features.

## Dataset Description:
The dataset (both train and test) was generated from a deep learning model trained on the Obesity or CVD risk dataset. Feature distributions are close to, but not exactly the same as, the original. This dataset is particularly well-suited for visualizations, clustering, and general EDA.
Dataset Link: https://www.kaggle.com/competitions/playground-series-s4e2/data
### Files:
**train.csv** - the training dataset; NObeyesdad is the categorical target

**test.csv** - the test dataset; Objective is to predict the class of NObeyesdad for each row

### Dataset Dictionary:

| Feature                        | Description |
|--------------------------------|-------------|
| `id`                           | Unique identifier |
| `Gender`                       | Gender |
| `Age`                          | Age of the individual |
| `Height`                       | Height in meters |
| `Weight`                       | Weight (between 39 and 165) |
| `family_history_with_overweight` | Family history with overweight (Yes/No) |
| `FAVC`                         | Frequent consumption of high-calorie food (Yes/No) |
| `FCVC`                         | Frequency of vegetable consumption |
| `NCP`                          | Number of main meals |
| `CAEC`                         | Consumption of food between meals |
| `SMOKE`                        | Smoking habit (Yes/No) |
| `CH2O`                         | Daily water consumption |
| `SCC`                          | Calories consumption monitoring (Yes/No) |
| `FAF`                          | Physical activity frequency |
| `TUE`                          | Time spent using technology devices (for health tracking) |
| `CALC`                         | Alcohol consumption |
| `MTRANS`                       | Mode of transportation used |
| `NObeyesdad`                   | **Target variable** – Obesity risk category |

## Target Variable: `NObeyesdad`

| Category              | BMI Range (kg/m²)      |
|-----------------------|-------------------------|
| `Insufficient_Weight` | Less than 18.5          |
| `Normal_Weight`       | 18.5 – 24.9             |
| `Overweight_Level_I`  | 25.0 – 29.9             |
| `Overweight_Level_II` | 25.0 – 29.9             |
| `Obesity_Type_I`      | 30.0 – 34.9             |
| `Obesity_Type_II`     | 35.0 – 39.9             |
| `Obesity_Type_III`    | Higher than 40          |

## Project Workflow
1.  **Exploratory Data Analysis (EDA):** Investigated the distributions of categorical and numerical variables to understand data patterns and relationships with the target variable.
2.  **Data Preprocessing:** Handled features data by converting categorical features into a numerical format. Ordinal columns were mapped to integers using the `.replace()` method, while nominal columns were transformed using one-hot encoding with `pd.get_dummies()`.
3.  **Model Building & Comparison:**
    * Trained and evaluated several baseline machine learning models.
    * Implemented advanced boosting algorithms for improved performance.
4.  **Model Evaluation:** Assessed model performance using accuracy scores and classification reports to select the best-performing model for prediction on the test dataset.

## Models Implemented
A variety of models were trained and evaluated to find the best classifier for this multi-class obesity prediction problem:

**Machine Learning Models:**
* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

**Advanced Boosting Models:**
* XGBoost Classifier
* LightGBM Classifier
* CatBoost Classifier
* Voting Classifier (ensemble of multiple models)  

## Results
The best-performing model was the **LightGBM**, which achieved an average accuracy of **0.9049** on the evaluation data. Although all the boosting classifiers and the voting classifier had very close accuracies, around 90%.
| Model                 | Average Accuracy        |
|-----------------------|-------------------------|
| `Decision Tree Classifier` | 0.8420          |
| `Random Forest Classifier`       | 0.8918             |
| `LightGBM Classifier`  | 0.9049             |
| `XGBoost Classifier` | 0.9043            |
| `CatBoost Classifier`      | 0.9047             |
| `Voting Classifier`     | 0.9036             |

## How to Run This Project
1.  Clone the repository to your local machine:
    ```bash
    git clone [https://github.com/Krishan019/obesity-risk-prediction.git](https://github.com/Krishan019/obesity-risk-prediction.git)
    ```
2.  Navigate to the project directory:
    ```bash
    cd obesity-risk-prediction
    ```
3.  Install the required Python libraries:
    ```bash
    pip install -r requirements.txt
    ```
4.  Launch Jupyter Notebook and open the project files:
    ```bash
    jupyter notebook
    ```
5.  Run the notebooks in order:
    * `01_EDA_and_Preprocessing.ipynb`
    * `02_Model_Training_and_Evaluation.ipynb`
      
## Feedback
This repository is primarily a learning archive, but I’m happy to hear your suggestions or answer questions.  
You can open an issue or start a discussion.

## License
This repository is released under the MIT License; you are free to use and modify the code here, but please provide appropriate credit when sharing.

## Contact
If you’d like to connect, collaborate, or have any questions, feel free to reach out:
- **Email:** [sharma.19.krishan@gmail.com](mailto:sharma.19.krishan@gmail.com)  
- **LinkedIn:** [Krishan Sharma](https://linkedin.com/in/krishan-19-sharma/)

⭐ *If you liked my work or found it useful, please give this repo a star!*
