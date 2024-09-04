import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
add Codeadd Markdown
Loading Data
add Codeadd Markdown
df_train = pd.read_csv("/kaggle/input/black-friday-sale/train.csv")
add Codeadd Markdown
df_train.head()
add Codeadd Markdown
Data Cleaning and Wrangling
add Codeadd Markdown
df= df_train
df.info()
add Codeadd Markdown
df.columns
add Codeadd Markdown
df.isnull().sum()
add Codeadd Markdown
missing_mean = df.isnull().mean()
print(missing_mean)
add Codeadd Markdown
# Fill with the most frequent category
df['Product_Category_2'].fillna(df['Product_Category_2'].mode()[0], inplace=True)
df['Product_Category_3'].fillna(df['Product_Category_3'].mode()[0], inplace=True)
df.isnull().sum()
add Codeadd Markdown
df.isnull().sum()
add Codeadd Markdown
df.shape
add Codeadd Markdown
df.head()
add Codeadd Markdown
le = LabelEncoder()
​
df['Gender'] = le.fit_transform(df['Gender'])
add Codeadd Markdown
le = LabelEncoder()
​
df['City_Category'] = le.fit_transform(df['City_Category'])
add Codeadd Markdown
df.head()
add Codeadd Markdown
df['Age'].value_counts()
add Codeadd Markdown
df['User_ID'].value_counts()
add Codeadd Markdown
df['Product_ID'].value_counts()
add Codeadd Markdown
# Convert '4+' to 4
df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].replace('4+', 4)
​
# Convert the column to numeric
df['Stay_In_Current_City_Years'] = pd.to_numeric(df['Stay_In_Current_City_Years'])
add Codeadd Markdown
import category_encoders as ce
​
cols_to_encode = ['User_ID']
cols_present = set(cols_to_encode).intersection(set(df.columns))
encoder = ce.BinaryEncoder(cols=cols_present)
df = encoder.fit_transform(df)
add Codeadd Markdown
cols_to_encode = ['Product_ID']
cols_present = set(cols_to_encode).intersection(set(df.columns))
encoder = ce.BinaryEncoder(cols=cols_present)
df = encoder.fit_transform(df)
add Codeadd Markdown
df.info()
add Codeadd Markdown
le = LabelEncoder()
​
df['Age'] = le.fit_transform(df['Age'])
add Codeadd Markdown
df.head()
add Codeadd Markdown
df.columns
add Codeadd Markdown
predictors = ['Gender', 'Age','Occupation', 'City_Category', 'Stay_In_Current_City_Years',
              'Marital_Status', 'Product_Category_1', 'Product_Category_2','Product_Category_3']
add Codeadd Markdown
X= df[predictors]
add Codeadd Markdown
X
add Codeadd Markdown
df['Purchase'] = np.sqrt(df['Purchase'])
add Codeadd Markdown
df['Purchase'].hist()
add Codeadd Markdown
import seaborn as sns
​
# Define the variables corr and mask
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
​
# Generate a custom dark diverging colormap
cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="dark", as_cmap=True)
​
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
​
plt.show()
add Codeadd Markdown
y= df['Purchase']
add Codeadd Markdown
y
add Codeadd Markdown
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42)
add Codeadd Markdown
print(X_train)
add Codeadd Markdown
print(X_test)
add Codeadd Markdown
print(y_train)
add Codeadd Markdown
print(y_test)
add Codeadd Markdown
print(X_train)
add Codeadd Markdown
print(X_test)
add Codeadd Markdown
from sklearn.metrics import accuracy_score
add Codeadd Markdown
friday_lr = LinearRegression()
add Codeadd Markdown
friday_lr.fit(X_train, y_train)
friday_lr= friday_lr.score(X_train, y_train)
print('Linear Regression on trained data =', friday_lr)
add Codeadd Markdown
# Assuming friday_lr is your linear regression model
friday_lr = LinearRegression()
friday_lr.fit(X_train, y_train)  # You should fit the model on the training data, not the test data
​
# Calculate the score of the model on the test data
score = friday_lr.score(X_test, y_test)
​
print('Linear Regression on test data =', score)
add Codeadd Markdown
y_predict_train = friday_lr.predict(X_train)
y_predict_test = friday_lr.predict(X_test)
add Codeadd Markdown
mse_train = mean_squared_error(y_train, y_predict_train)
mse_train
add Codeadd Markdown
mse_test = mean_squared_error(y_test, y_predict_test)
mse_test
add Codeadd Markdown
rmse_train = np.sqrt(mse_train)
rmse_train
add Codeadd Markdown
rmse_test = np.sqrt(mse_test)
rmse_test
add Codeadd Markdown
df = pd.DataFrame({" Y Prediction": y_predict_test,
                   "Y Actual": y_test}).reset_index(drop=True)
df.head(10)
add Codeadd Markdown
plt.figure(figsize = (15,8))
plt.plot(df[:50])
plt.legend(["Y Prediction", "Y Actual"])
add Codeadd Markdown
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
​
rfr.score(X_train, y_train)
add Codeadd Markdown
rfr.score(X_test, y_test)
add Codeadd Markdown
y_predict_test = rfr.predict(X_test)
add Codeadd Markdown
df = pd.DataFrame({" Y Prediction": y_predict_test,
                   "Y Actual": y_test}).reset_index(drop=True)
df.head(10)
add Codeadd Markdown
plt.figure(figsize = (15,8))
plt.plot(df[:50])
plt.legend(["Y Prediction", "Y Actual"])
