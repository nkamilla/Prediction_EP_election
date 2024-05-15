
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor




election_data = pd.read_excel("Dataset EP Elections 1979-2019.xlsx")

features = ['country', 'year', 'ep_affiliation', 'seats', 'votes']

filtered_data = election_data[(election_data['year'] >= 2009) & (election_data['year'] <= 2019)]

target = "votes"  # Target variable name

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(election_data[features], election_data[target], test_size=0.2, random_state=42)

# Define preprocessing steps for numeric and categorical features
numeric_features = ['year', 'seats', 'votes']  # List of numeric feature column names
categorical_features = ['country', 'ep_affiliation']  # List of categorical feature column names

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the model
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier())])

# Train the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))