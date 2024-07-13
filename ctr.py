import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



st.markdown(
    """
    # Ads Click Through Rate Prediction
    [![LinkedIn](https://img.shields.io/badge/LinkedIn-Visit-blue)](https://www.linkedin.com/in/bhavyachawla/)
    [![GitHub](https://img.shields.io/badge/GitHub-Visit-blue)](https://github.com/BhavyaChawlaGit)
    
    """,
    unsafe_allow_html=True,
)


pio.templates.default = "plotly_white"

data = pd.read_csv("Dataset/ad_10000records.csv")
print(data.head())


# The “Clicked on Ad” column contains 0 and 1 values, where 0 means not clicked, and 1 means clicked. 
# I’ll transform these values into “yes” and “no”:

data["Clicked on Ad"] = data["Clicked on Ad"].map({0: "No", 1: "Yes"})

# Analyzing the data, based on time spent by the users 

fig = px.box(data, 
             x="Daily Time Spent on Site",  
             color="Clicked on Ad", 
             title="Click Through Rate based Time Spent on Site", 
             color_discrete_map={'Yes':'blue',
                                 'No':'red'})
fig.update_traces(quartilemethod="exclusive")
#fig.show()


# Now analysing the click through rate based on daily internet usage

fig = px.box(data, 
             x="Daily Internet Usage",  
             color="Clicked on Ad", 
             title="Click Through Rate based on Daily Internet Usage", 
             color_discrete_map={'Yes':'blue',
                                 'No':'red'})
fig.update_traces(quartilemethod="exclusive")
#fig.show()

# Click through rate based on age

fig = px.box(data, 
             x="Age",  
             color="Clicked on Ad", 
             title="Click Through Rate based on Age", 
             color_discrete_map={'Yes':'blue',
                                 'No':'red'})
fig.update_traces(quartilemethod="exclusive")
#fig.show()

# 4th Analyzing based on income of users

fig = px.box(data, 
             x="Area Income",  
             color="Clicked on Ad", 
             title="Click Through Rate based on Income", 
             color_discrete_map={'Yes':'blue',
                                 'No':'red'})
fig.update_traces(quartilemethod="exclusive")
#fig.show()

# Calculating CTR of Ads

yes_count = data["Clicked on Ad"].value_counts()["Yes"]
print(yes_count)
click_through_rate = yes_count / 10000 * 100
print("Click through rate: ",click_through_rate)


# Click Through Rate Prediction Model

data["Gender"] = data["Gender"].map({"Male": 1,"Female": 0})

x=data.iloc[:,0:7]
x=x.drop(['Ad Topic Line','City'],axis=1)
y=data.iloc[:,9]
# Splitting the data for train and test
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=4)

# Training the model
model = RandomForestClassifier()
model.fit(x, y)


# To check the accuracy of the model
y_pred=model.predict(xtest)
print("accuracy_score: ",accuracy_score(ytest,y_pred))



# # Now using the model and Making Predictions
# print("Ads Click Through Rate Prediction : ")
# a = float(input("Daily Time Spent on Site: "))
# b = float(input("Age: "))
# c = float(input("Area Income: "))
# d = float(input("Daily Internet Usage: "))
# e = input("Gender (Male = 1, Female = 0) : ")

# features = np.array([[a, b, c, d, e]])
# print("Will the user click on ad = ", model.predict(features))


# StreamLit App

a = st.number_input("Daily Time Spent on Site:", format="%.2f")
b = st.number_input("Age:", format="%.2f")
c = st.number_input("Area Income:", format="%.2f")
d = st.number_input("Daily Internet Usage:", format="%.2f")
e = st.selectbox("Gender:", options=["Male", "Female"])

# Convert gender to 0 or 1
e = 1 if e == "Male" else 0

# Making Predictions
if st.button("Predict"):
    features = np.array([[a, b, c, d, e]])
    prediction = model.predict(features)
    prediction_text = "Will click on ad" if prediction[0] == 1 else "Will not click on ad"
    #st.success(prediction_text)
    if prediction[0] == 1:
        st.success("User will click on the ad")
    else:
        st.error("User will not click on the ad")