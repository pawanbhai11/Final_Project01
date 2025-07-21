# Import dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression 

df = pd.read_csv('data.csv') 
df['Month_Number'] = np.arange(len(df))
# print(df)

X = df[['Month_Number']]  # Feature
Y = df['Sales']           # Target

model = LinearRegression()
model.fit(X, Y)

df['Predicted_Sales'] = model.predict(X)
next_month = [[len(df)]]
next_month_sales = model.predict(next_month)[0]

plt.figure(figsize=(10,6))
plt.plot(df['Month'], df['Sales'], marker='o', label='Actual Sales')
plt.plot(df['Month'], df['Predicted_Sales'], linestyle='--', label='Predicted Sales')
plt.scatter(df['Month'].iloc[-1], next_month_sales, color='red', s=100, label='Next Month Prediction')
plt.title("Sales Prediction using Linear Regression")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"Predicted Sales for Next Month: {next_month_sales:.2f}")
