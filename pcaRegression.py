import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
import numpy as np

# Load dataset
df = pd.read_csv('student_scores.csv')
X = df.drop(columns=['Score'])  # Features
y = df['Score']                # Target

# Train-test split and standardization
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)

# Regression without PCA
model = LinearRegression().fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
print("Without Dimensionality Reduction - MAE:", mean_absolute_error(y_test, y_pred), 
      ", RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Regression with PCA
pca = PCA(n_components=0.95)
X_train_pca, X_test_pca = pca.fit_transform(X_train_scaled), pca.transform(X_test_scaled)
model_pca = LinearRegression().fit(X_train_pca, y_train)
y_pred_pca = model_pca.predict(X_test_pca)
print("With PCA - MAE:", mean_absolute_error(y_test, y_pred_pca), 
      ", RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_pca)))
