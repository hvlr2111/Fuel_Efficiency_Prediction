# 🚗 Fuel Efficiency Prediction using Neural Networks
⭐ Deep Learning Project | TensorFlow + Keras | Regression Analysis

## 📊 Project Overview
This project focuses on predicting the fuel efficiency (MPG) of vehicles using a **Neural Network** model. The dataset contains various automotive features like cylinders, displacement, horsepower, weight, and acceleration.

The model learns complex relationships between vehicle specifications and fuel consumption, providing accurate MPG predictions that can be valuable for automotive analysis and environmental impact assessment.

## 🎯 Objectives
* Load and preprocess automotive data for training.
* Build and train a Neural Network model to predict fuel efficiency.
* Implement early stopping to prevent overfitting.
* Evaluate model performance with regression metrics.
* Generate predictions on new unseen data.
* Visualize training progress and results.

## 🛠️ Tech Stack
* **Programming Language:** Python
* **Libraries:**
    * `numpy`, `pandas` → Data manipulation
    * `matplotlib`, `seaborn` → Data visualization
    * `scikit-learn` → Preprocessing, model evaluation
    * `tensorflow`, `keras` → Deep Learning model creation & training

## 📁 Dataset
* **Source:** Auto MPG Dataset
* **Description:**
    * The dataset contains various vehicle specifications and their corresponding fuel efficiency.
    * Features include technical specifications like cylinders, displacement, horsepower, etc.

| Feature | Type | Description |
| :--- | :--- | :--- |
| `mpg` | Continuous | Miles per gallon (target variable) |
| `cylinders` | Discrete | Number of cylinders |
| `displacement` | Continuous | Engine displacement |
| `horsepower` | Continuous | Engine horsepower |
| `weight` | Continuous | Vehicle weight |
| `acceleration` | Continuous | Acceleration performance |
| `model year` | Discrete | Vehicle model year |
| `origin` | Categorical | Country of origin |

## 🔬 Methodology & Steps
1️⃣ **Data Loading & Exploration**
* Loaded the dataset and inspected its structure.
* Handled missing values and performed data cleaning.
* Visualized feature distributions and correlations.
* Normalized numerical features for better training.

2️⃣ **Data Preprocessing**
* Split data into train (80%) and test (20%) sets.
* Scaled features using StandardScaler for optimal performance.
* Prepared data for neural network input.

3️⃣ **Model Architecture (Neural Network)**
```python
model = Sequential([
    Dense(64, activation='relu', input_shape=[len(train_features[0])]),
    Dense(64, activation='relu'),
    Dense(1)
])
```
* **Optimizer:** `Adam`
* **Loss Function:** `Mean Squared Error`
* **Metrics:** `Mean Absolute Error`, `Mean Squared Error`

4️⃣ **Model Training with Early Stopping**
* Trained the neural network using `model.fit()` for multiple epochs.
* Implemented early stopping callback to prevent overfitting.
* Monitored training & validation loss curves.
* Saved the best model based on validation performance.

5️⃣ **Model Evaluation**
* Evaluated on test set using `model.evaluate()`.
* Calculated regression metrics (MAE, MSE, RMSE).
* Generated predictions and visualized against actual values.
* Analyzed feature importance and model interpretability.

6️⃣ **Visualization**
* Training vs Validation loss curves.
* Predicted vs Actual MPG scatter plots.
* Residual analysis and error distribution.

## 📈 Results
* ✅ **Final Performance:** Low prediction error on test data
* ✅ **Model:** Multi-layer Neural Network
* ✅ **Optimizer:** `Adam`
* ✅ **Loss Function:** `Mean Squared Error`
* ✅ **Regularization:** Early Stopping implemented

The model effectively predicts fuel efficiency with good accuracy, demonstrating strong performance on unseen vehicle data while avoiding overfitting through early stopping.

## 🚀 How to Run

1.  **Clone the repository**

    ```bash
    git clone https://github.com/hvlr2111/Fuel_Efficiency_Prediction.git
    ```

2.  **Navigate to the project folder**

    ```bash
    cd fuel-efficiency-prediction
    ```

3.  **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the dataset**
    The dataset will be automatically downloaded during execution.

5.  **Run the notebook or script**

    ```bash
    python fuel_efficiency.py
    ```

    or open and run `Fuel_Efficiency_Prediction.ipynb` cell by cell in Jupyter Notebook.

## 🧠 Future Improvements
* Experiment with different neural network architectures.
* Implement hyperparameter tuning for optimal performance.
* Add more features and perform feature engineering.
* Deploy the model as a web service using Flask or FastAPI.
* Create an interactive dashboard for predictions.

## 👤 Author
* H.V.L. Ranasinghe
* LinkedIn: https://www.linkedin.com/in/lakshika-ranasinghe-1404ab34a/
