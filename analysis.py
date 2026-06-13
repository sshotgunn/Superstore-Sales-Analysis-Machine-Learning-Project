"""
    Superstore Sales Analysis &
    Machine Learning Project
    Created by: Saha Rathor
"""

"""Importing Libraries"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""Loading Data"""
df = pd.read_csv("Superstore.csv", encoding = 'latin1')
df.head()

"""Basic Information"""
df.info()
df.describe()

"""Data Cleaning"""
# Check missing values
df.isnull().sum()

# Remove duplicates
df = df.drop_duplicates()

# Convert date column
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])

"""Feature Engineering"""
df['Profit Margin'] = df['Profit'] / df['Sales']
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month

"""Overall Sales  Trend"""
monthly_sales = df.groupby(df['Order Date'].dt.to_period('M'))['Sales'].sum()

monthly_sales.plot(figsize=(12,5))
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show()

"""Top Categories"""
sns.barplot(data=df, x='Category', y='Sales', estimator=sum)
plt.title("Sales by Category")
plt.show()

"""Profit by Region"""
region_profit = df.groupby('Region')['Profit'].sum()

region_profit.plot(kind='bar')
plt.title("Profit by Region")
plt.show()

"""Most Profitable Sub-Categories"""
subcat = df.groupby('Sub-Category')['Profit'].sum().sort_values()

subcat.plot(kind='barh', figsize=(10,6))
plt.title("Profit by Sub-Category")
plt.show()

"""Correlation Analysis"""
sns.heatmap(df[['Sales','Profit','Discount','Quantity']].corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

"""Preparing Data"""
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

df_ml = df[['Sales','Discount','Quantity','Profit']]

X = df_ml.drop('Profit', axis=1)
y = df_ml['Profit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""Training Model"""
model = RandomForestRegressor()
model.fit(X_train, y_train)

"""Evaluation"""
pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, pred))

"""Feature Importance"""
importance = pd.Series(model.feature_importances_, index=X.columns)
importance.plot(kind='bar')
plt.title("Feature Importance")
plt.show()

#df.to_csv("Superstore.csv", index=False)
