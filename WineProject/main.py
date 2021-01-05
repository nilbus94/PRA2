# library we are using
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import anderson
from scipy.stats import fligner
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt


# Create the dataframe coming from the csv by using pandas.read_csv() function
df = pd.read_csv("wine_quality-red.csv")

# print the head of the dataframe
print(df.head(10))
# show the number of rows and columns of the dataframe
print("Rows, columns: " + str(df.shape))

# Check null values
print(df.isna().sum())

# Replace spaces with _ for each column to avoid future problems
df.columns = df.columns.str.replace(' ', '_')

# Print all column names
for col_name in df.columns:
    print(col_name)

# List the unique values for each column
print(df.fixed_acidity.unique())
print(df.volatile_acidity.unique())
print(df.citric_acid.unique())
print(df.residual_sugar.unique())
print(df.chlorides.unique())
print(df.free_sulfur_dioxide.unique())
print(df.total_sulfur_dioxide.unique())
print(df.density.unique())
print(df.pH.unique())
print(df.sulphates.unique())
print(df.alcohol.unique())
print(df.quality.unique())

# Create a boxplot
sns.boxplot(data=df, orient="v")
plt.show()

# Create the new dataframes grouping by quality
low_quality_df = df[df.quality.isin([3, 4])]
medium_quality_df = df[df.quality.isin([5, 6])]
high_quality_df = df[df.quality.isin([7, 8])]

# Validate that the dataframes are correctly generated
print(low_quality_df.quality.unique())
print(medium_quality_df.quality.unique())
print(high_quality_df.quality.unique())

# For each column of data we apply the Anderson Darling theorem
for column in df:
    print(column)
    print(anderson(df[column], dist="norm"))

# For each column of data we apply the Anderson Darling theorem
for column in low_quality_df:
    print(column)
    print(anderson(low_quality_df[column], dist="norm"))

# For each column of data we apply the Anderson Darling theorem
for column in medium_quality_df:
    print(column)
    print(anderson(medium_quality_df[column], dist="norm"))

# For each column of data we apply the Anderson Darling theorem
for column in high_quality_df:
    print(column)
    print(anderson(high_quality_df[column], dist="norm"))

# Apply the Flinger theory
for column in high_quality_df:
    print(column)
    print(fligner(low_quality_df[column], medium_quality_df[column], high_quality_df[column]))

# CORRELATION
# Correlations between variables
plt.figure(figsize=(10, 6)).subplots_adjust(bottom=0.25)
sns.heatmap(df.corr(), annot=True, fmt='.0%')
plt.show()

# Calculate and order correlations
plt.figure(figsize=(10, 6)).subplots_adjust(bottom=0.25)
df.corr()['quality'].sort_values(ascending=False).plot(kind='bar')
plt.show()

# Matrix correlation between all variables
plt.figure(figsize=(10, 6))
sns.stripplot(data=df, x="quality", y="alcohol", jitter=True)
plt.show()

# REGRESSION
# We separate our features from our target feature (quality) and we split data intro training and test
X1 = df.loc[:, ['alcohol', 'sulphates', 'citric_acid', 'volatile_acidity', 'total_sulfur_dioxide']]
X2 = df.loc[:, ['alcohol', 'sulphates', 'citric_acid', 'volatile_acidity']]
X3 = df.loc[:, ['alcohol', 'sulphates', 'citric_acid']]
X4 = df.loc[:, ['alcohol', 'sulphates']]
Y = df.iloc[:, 11]

# Create the test and training samples
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, Y, test_size=0.4, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, Y, test_size=0.4, random_state=42)
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, Y, test_size=0.4, random_state=42)
X4_train, X4_test, y4_train, y4_test = train_test_split(X4, Y, test_size=0.4, random_state=42)

# Fit the model and make prediction
reg1 = LinearRegression()
reg2 = LinearRegression()
reg3 = LinearRegression()
reg4 = LinearRegression()
reg1.fit(X1_train, y1_train)
reg2.fit(X2_train, y2_train)
reg3.fit(X3_train, y3_train)
reg4.fit(X4_train, y4_train)
y1_prediction_lr = reg1.predict(X1_test)
y2_prediction_lr = reg2.predict(X2_test)
y3_prediction_lr = reg3.predict(X3_test)
y4_prediction_lr = reg4.predict(X4_test)

# Evaluate our models
print(sqrt(mean_squared_error(y1_test, y1_prediction_lr)))
print(sqrt(mean_squared_error(y2_test, y2_prediction_lr)))
print(sqrt(mean_squared_error(y3_test, y3_prediction_lr)))
print(sqrt(mean_squared_error(y4_test, y4_prediction_lr)))
