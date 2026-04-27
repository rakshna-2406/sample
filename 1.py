#q1
# ----------- IMPORTS -----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# ----------- LOAD DATA -----------
df = pd.read_csv("data.csv") #change filename if needed

# ----------- EDA -----------
print("HEAD:\n", df.head())
print("\nINFO:")
print(df.info())
print("\nDESCRIBE:\n", df.describe())

print("\nMISSING VALUES BEFORE:\n", df.isnull().sum())

# ----------- PREPROCESSING -----------

# Remove duplicate rows
df = df.drop_duplicates()

# Handle missing values
df = df.fillna(df.mean(numeric_only=True))

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

print("\nDATA AFTER PREPROCESSING:\n", df.head())

print("\nMISSING VALUES AFTER:\n", df.isnull().sum())

# Keep only numeric columns
df = df.select_dtypes(include=np.number)

# ----------- NORMALIZATION -----------
normalizer = MinMaxScaler()
normalized_data = normalizer.fit_transform(df)

print("\nNORMALIZED DATA SAMPLE")
print(normalized_data[:5])

# ----------- SCALING -----------
scaler = StandardScaler()
scaled = scaler.fit_transform(df)

print("\nSCALED DATA SAMPLE")
print(scaled[:5])

# ----------- DATA VISUALIZATION -----------
cols = df.columns[:6]

# Histogram
df[cols].hist(figsize=(10,8))
plt.tight_layout()
plt.show()

# Boxplot
for col in cols:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(col)
    plt.show()

# Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df[cols].corr(), annot=True, cmap='coolwarm')
plt.show()

#q2-classification
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("data.csv")
df = df.iloc[:, :6]

print("\nDATA BEFORE DECISION TREE:\n", df.head(10))

df = df.fillna(df.mean(numeric_only=True))
df = df.apply(lambda x: x.astype('category').cat.codes)

target = "Income"
df[target] = (df[target] > df[target].median()).astype(int)

X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

df['Predicted_Class'] = model.predict(X)

print("\nDATA AFTER DECISION TREE:\n", df.head(10))

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

#Q2-regression
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

df=pd.read_csv("Data.csv")
df=df.iloc[:,:6]

print("\nData before Lasso Regression:\n", df.head(10))

df=df.fillna(df.mean(numeric_only=True))
df=df.apply(lambda x:x.astype('category').cat.codes)

target="Rent"

X=df.drop(target, axis=1)
y=df[target]

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)

model=Lasso(alpha=0.1)
model.fit(X_train,y_train)

df['Predicted_Value']=model.predict(X)

print("\nData after Lasso Regression:\n", df.head(10))

y_pred=model.predict(X_test)

print("\nMAE:",mean_absolute_error(y_test,y_pred))
print("MSE:",mean_squared_error(y_test,y_pred))
print("RMSE:",np.sqrt(mean_squared_error(y_test,y_pred)))
print("R2 Score:",r2_score(y_test,y_pred))

#Q2-Kmeans
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("data.csv")

df = df.iloc[:, :3]

print("\nDATA BEFORE KMEANS:\n", df.head(10))

# Preprocessing
df = df.fillna(df.mean(numeric_only=True))
df = df.apply(lambda x: x.astype('category').cat.codes)

# Model
model = KMeans(n_clusters=3)

model.fit(df)

# Add cluster labels
df['Cluster'] = model.labels_

print("\nDATA AFTER KMEANS:\n", df.head(10))

# -------- PERFORMANCE MEASURE --------
score = silhouette_score(df.drop('Cluster', axis=1), df['Cluster'])

print("\nSilhouette Score:", score)

#Q2-TimeSeries
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv("Supermart Grocery Sales - Retail Analytics Dataset.csv")

print("\nDATA PREVIEW:\n", df.head())

# Date and value column
date_col = "Order Date"
value_col = "Sales"

df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
df = df.set_index(date_col)

# Train model
model = ARIMA(df[value_col], order=(1,1,1))
model_fit = model.fit()

# Forecast
pred = model_fit.predict(start=0, end=len(df)-1)

# Performance
mse = mean_squared_error(df[value_col], pred)

print("\nForecast Values:\n", pred.head())
print("\nMSE:", mse)

#Q3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Load dataset
df = pd.read_csv("train.csv", encoding="latin-1")

# ===== DATASET PREVIEW (ADDED) =====
print("DATASET PREVIEW:\n", df.head())
# ==================================

print(df.head())
print(df.info())

# Drop missing values
df = df.dropna()

# IMPORTANT: Change column names if needed
text_col = "text"
label_col = "sentiment"

# Features and labels
X = df[text_col]
y = df[label_col]

# Convert text to numbers using TF-IDF
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(X)

# ===== TF-IDF OUTPUT (ALREADY ADDED BEFORE) =====
print("TF-IDF Shape:", X.shape)
print("Sample Words:", tfidf.get_feature_names_out()[:10])
print("TF-IDF Sample Values:\n", X[0].toarray())
# ===============================================
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# ===== SAMPLE PREDICTIONS (ADDED) =====
print("\nSample Predictions:")
print(pd.DataFrame({
    "Actual": y_test.values[:5],
    "Predicted": y_pred[:5]
}))
# =====================================
# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))

#Q4-Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =========================
# 1. Dataset Paths
# =========================
train_dir = "train" #change this to dataset path
test_dir = "test"   #change this to dataset path

# =========================
# 2. Image Preprocessing
# =========================
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

# =========================
# 3. Build CNN Model
# =========================
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# =========================
# 4. Compile Model
# =========================
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =========================
# 5. Train Model
# =========================
history = model.fit(
    train_data,
    epochs=5,
    validation_data=test_data
)

# =========================
# 6. Evaluate Model
# =========================
loss, accuracy = model.evaluate(test_data)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# =========================
# 7. Prediction on New Image
# =========================
import numpy as np
from tensorflow.keras.preprocessing import image

img_path = "2992605.jpg"  # change this
img = image.load_img(img_path, target_size=(128,128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print("Class 1 (e.g., Pizza)")
else:
    print("Class 2 (e.g., Not Pizza)")