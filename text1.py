//Q1
# ----------- IMPORTS -----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler

# ----------- LOAD DATA -----------
df = pd.read_csv("data.csv")

# ----------- EDA -----------
print("HEAD:\n", df.head())
print("\nINFO:")
print(df.info())
print("\nDESCRIBE:\n", df.describe())

print("\nMISSING VALUES BEFORE:\n", df.isnull().sum())

# ----------- PREPROCESSING -----------
# 1. Handle missing values (numeric)
df = df.fillna(df.mean(numeric_only=True))

# 2. Handle missing values (categorical)
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# 3. Label Encoding (safe)
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

print("\nAFTER PREPROCESSING:\n", df.head())
print("\nMISSING VALUES AFTER:\n", df.isnull().sum())

# 4. Keep only numeric (VERY IMPORTANT → universal safety)
df = df.select_dtypes(include=np.number)

# 6. Scaling
scaler = StandardScaler()
scaled = scaler.fit_transform(df)
print("\nScaled Data")
print(scaled[:5])

# Select few columns (avoid clutter)
num_cols = df.columns[:6]

# Histogram
df[num_cols].hist(figsize=(10,8))
plt.tight_layout()
plt.show()

# Boxplot (one-by-one)
for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title(col)
    plt.show()

# Heatmap (bonus)
plt.figure(figsize=(8,6))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm')
plt.show()


//Q2
//classification
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("data.csv")

# Take few columns
df = df.iloc[:, :6]

print("\nDATA BEFORE CLASSIFICATION:\n", df.head())

# Preprocessing
df = df.fillna(df.mean(numeric_only=True))
df = df.apply(lambda x: x.astype('category').cat.codes)

# Split
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Add predictions to dataset
df['Predicted_Class'] = model.predict(X)

print("\nDATA AFTER CLASSIFICATION:\n", df.head())


//Reg 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data.csv")

# Few columns (fast)
df = df.iloc[:, :6]

print("\nDATA BEFORE REGRESSION:\n", df.head())

# Preprocessing
df = df.fillna(df.mean(numeric_only=True))
df = df.apply(lambda x: x.astype('category').cat.codes)

# Split
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Add predictions
df['Predicted_Value'] = model.predict(X)

print("\nDATA AFTER REGRESSION:\n", df.head())


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("data.csv")

df = df.iloc[:, :6]

print("\nDATA BEFORE REGRESSION:\n", df.head())

# Preprocessing
df = df.fillna(df.mean(numeric_only=True))
df = df.apply(lambda x: x.astype('category').cat.codes)

# Split
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Add predictions
df['Predicted_Value'] = model.predict(X)

print("\nDATA AFTER REGRESSION:\n", df.head())


//KMeans
import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv("data.csv")

df = df.iloc[:, :5]

print("\nDATA BEFORE KMEANS:\n", df.head())

# Preprocessing
df = df.fillna(df.mean(numeric_only=True))
df = df.apply(lambda x: x.astype('category').cat.codes)

# Model
model = KMeans(n_clusters=3)
model.fit(df)

# Add cluster labels
df['Cluster'] = model.labels_

print("\nDATA AFTER KMEANS:\n", df.head())


//Image
# Step 1: Import libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Step 2: Load images from folder
data = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train = data.flow_from_directory(
    "Images",
    target_size=(64,64),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

test = data.flow_from_directory(
    "Images",
    target_size=(64,64),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

# Step 3: Build simple CNN model
model = models.Sequential([
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(64,64,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(train.num_classes,activation='softmax')
])

# Step 4: Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 5: Train model
model.fit(train, epochs=5, validation_data=test)

# Step 6: Evaluate
loss, acc = model.evaluate(test)
print("Accuracy:", acc)

pip install pandas numpy matplotlib seaborn scikit-learn tensorflow 