import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Read in data using Pandas
s = pd.read_csv("social_media_usage.csv")

# Define function
def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x
# Establish dataframe by coding missing values
ss = pd.DataFrame({
    "income":np.where((s["income"] >= 1) & (s["income"] < 10), s["income"], np.nan),
    "education":np.where((s["educ2"] >= 1) & (s["educ2"] < 9), s["educ2"], np.nan),
    "parent":np.where((s["par"] >= 1) & (s["par"] < 3), s["par"], np.nan),
    "married":np.where((s["marital"] >= 1) & (s["marital"] < 7), s["marital"], np.nan),
    "female":np.where((s["gender"] >= 1) & (s["gender"] < 4), s["gender"], np.nan),
    "age":np.where((s["age"] >= 1) & (s["age"] < 98), s["age"], np.nan)})

# Use function to code in target variable
ss["sm_li"] = clean_sm(s["web1h"])

# Remove NaN values from dataframe
ss = ss.dropna()

# Encode variables properly
ss["income"] = ss["income"].apply(np.int64)
ss["education"] = ss["education"].apply(np.int64)
ss["parent"] = np.where(ss["parent"] == 1, 1, 0)
ss["married"] = np.where(ss["married"] == 1, 1, 0)
ss["female"] = np.where(ss["female"] == 2, 1, 0) # Female = 2 in gender variable of "s" dataframe
ss["age"] = ss["age"].apply(np.int64)

# Target (y) and feature(s) selection (X)
y = ss["sm_li"]
X = ss[["income", "education", "parent", "married", "female", "age"]]

# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,       # same num of target in test and training set
                                                    test_size=0.2,    # keep 20% of data for testing
                                                    random_state=1789) # set seed for reproducibility

# Initialize algorithm 
lr = LogisticRegression(class_weight = "balanced")

# Fit algorithm to training data
lr.fit(X_train, y_train)

# Make predictions using the model and the testing data
y_pred = lr.predict(X_test)

# End of model setup

## Web interface
st.title("Are they on LinkedIn?")
st.write("Fill out the information below and we'll predict if that\
         person is likely to be a LinkedIn user!")
st.write("The prediction will adapt as you enter information.")

# New data for features: income, education, parent, married, female, age
person = []

# Get input from user
income_input = st.selectbox(
    'Household Income',
    ('Less than $10,000', '10 to under $20,000', '20 to under $30,000',
     '30 to under $40,000', '40 to under $50,000', '50 to under $75,000',
     '75 to under $100,000', '100 to under $150,000', '$150,000 or more'))
if income_input == 'Less than $10,000':
    income = 1
elif income_input == '10 to under $20,000':
    income = 2
elif income_input == '20 to under $30,000':
    income = 3
elif income_input == '30 to under $40,000':
    income = 4
elif income_input == '40 to under $50,000':
    income = 5
elif income_input == '50 to under $75,000':
    income = 6
elif income_input == '75 to under $100,000':
    income = 7
elif income_input == '100 to under $150,000':
    income = 8
else:
    income = 9

education_input = st.selectbox(
    'Highest Level of Education Completed',
    ('Less than high school (Grades 1-8 or no formal schooling)', 
     'High school incomplete (Grades 9-11 or Grade 12 with NO diploma)',
     'High school graduate (Grade 12 with diploma or GED certificate)',
     'Some college, no degree (includes some community college)', 
     'Two-year associate degree from a college or university', 
     'Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)',
     'Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)', 
     'Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)'))
if education_input == 'Less than high school (Grades 1-8 or no formal schooling)':
    education = 1
elif education_input == 'High school incomplete (Grades 9-11 or Grade 12 with NO diploma)':
    education = 2
elif education_input == 'High school graduate (Grade 12 with diploma or GED certificate)':
    education = 3
elif education_input == 'Some college, no degree (includes some community college)':
    education = 4
elif education_input == 'Two-year associate degree from a college or university':
    education = 5
elif education_input == 'Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)':
    education = 6
elif education_input == 'Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)':
    education = 7
else:
    education = 8

parent_input = st.toggle('Is a Parent')
if parent_input:
    parent = 1
else:
    parent = 0

married_input = st.toggle('Is Married')
if married_input:
    married = 1
else:
    married = 0

female_input = st.toggle('Identifies as Female')
if female_input:
    female = 1
else:
    female = 0

age_input = st.slider('Age', 13, 99, 25)
age = age_input
# Use native features to encode person list
person.append(income)
person.append(education)
person.append(parent)
person.append(married)
person.append(female)
person.append(age)

# Predict class, given input features
predicted_class = lr.predict([person])

# Generate probability of positive class (=1)
probs = lr.predict_proba([person])

# Print predicted class and probability
st.write(f"Predicted class: {predicted_class[0]}") # 0=not LinkedIn user, 1=LinkedIn user
st.write(f"Probability that this person is a LinkedIn user: {probs[0][1]}")

if predicted_class[0] == 1:
    st.markdown("## Yes, we think they are on LinkedIn!")
else:
    st.markdown("## No, we don't think they are on LinkedIn!")