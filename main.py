#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import  LabelEncoder
import pickle


# In[2]:


st.header("Heart Disease Prediction")
st.text_input("Enter patient's name: ", key="name")


# In[ ]:





# In[3]:


data = pd.read_csv(r"D:\Python\ArtyvisTechnologies\heart.csv")


# In[4]:


categorical_features = ['ChestPainType', 'ST_Slope']
numeric_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
eda_df = data.loc[:, numeric_features]
eda_df.columns


# In[5]:


def one_hot_encode(df, column_dict):
  for column, prefix in column_dict.items():
    dummies = pd.get_dummies(df[column], prefix = prefix)
    df = pd.concat([df, dummies], axis = 1)
    df = df.drop(column, axis = 1)
  return df


# In[6]:


# data = one_hot_encode(data, dict(zip(categorical_features, ['CP', 'ST'])))


# In[7]:


# data["Sex"] = data.groupby("Sex").ngroup()
# #data["ChestPainType"] = data.groupby("ChestPainType").ngroup()
# #data["ST_Slope"] = data.groupby("ST_Slope").ngroup()
# data["RestingECG"] = data.groupby("RestingECG").ngroup()
# data["ExerciseAngina"] = data.groupby("ExerciseAngina").ngroup()


# In[8]:


# data


# In[9]:


svm_model = pickle.load(open(r"D:\Python\ArtyvisTechnologies\svcHeart.sav", 'rb'))


# In[10]:


if st.checkbox('Show Training Dataframe'):
    data


# In[11]:


encoder = LabelEncoder()


# In[12]:


st.subheader("Please select relevant features of the Patient:")


age = st.slider('Age', 0, max(data["Age"]), 1)

left_column, right_column = st.columns(2)
with left_column:
    sex = st.radio(
        'Sex of patient:',
        np.unique(data['Sex']))
    
left_column, right_column = st.columns(2)
with left_column:
    chestpain = st.radio(
        'Chest pain type:',
        np.unique(data['ChestPainType']))

restingbp = st.slider('Resting BP', 0, max(data["RestingBP"]), 1)

cholestrol = st.slider('Cholesterol', 0, max(data["Cholesterol"]), 1)

left_column, right_column = st.columns(2)
with left_column:
    fastingbs = st.radio(
        'Fasting BS:',
        np.unique(data['FastingBS']))
    
left_column, right_column = st.columns(2)
with left_column:
    restingecg = st.radio(
        'Resting ECG:',
        np.unique(data['RestingECG']))

maxhr = st.slider('MaxHR', 0, max(data["MaxHR"]), 1)

left_column, right_column = st.columns(2)
with left_column:
    exerciseangina = st.radio(
        'Exercise Angina:',
        np.unique(data['ExerciseAngina']))
    
oldpeak = st.slider('Oldpeak', min(data['Oldpeak']), max(data["Oldpeak"]), 0.0)

left_column, right_column = st.columns(2)
with left_column:
    stslope = st.radio(
        'ST Slope:',
        np.unique(data['ST_Slope']))

rs = pickle.load(open(r'D:\Python\ArtyvisTechnologies\rs.pkl', 'rb'))

if st.button('Make Prediction'):
    encoder.classes_ = np.load(r'D:\Python\ArtyvisTechnologies\classes_sex.npy',allow_pickle=True)
    sex = encoder.transform(np.expand_dims(sex, -1))
    encoder.classes_ = np.load(r'D:\Python\ArtyvisTechnologies\classes_chestpain.npy',allow_pickle=True)
    chestpain = encoder.transform(np.expand_dims(chestpain, -1))
    encoder.classes_ = np.load(r'D:\Python\ArtyvisTechnologies\classes_fastingbs.npy',allow_pickle=True)
    fastingbs = encoder.transform(np.expand_dims(fastingbs, -1))
    encoder.classes_ = np.load(r'D:\Python\ArtyvisTechnologies\classes_restingecg.npy',allow_pickle=True)
    restingecg = encoder.transform(np.expand_dims(restingecg, -1))
    encoder.classes_ = np.load(r'D:\Python\ArtyvisTechnologies\classes_exerciseangina.npy',allow_pickle=True)
    exerciseangina = encoder.transform(np.expand_dims(exerciseangina, -1))
    encoder.classes_ = np.load(r'D:\Python\ArtyvisTechnologies\classes_stslope.npy',allow_pickle=True)
    stslope = encoder.transform(np.expand_dims(stslope, -1))
    inputs = rs.transform(np.expand_dims(
        ([int(age), int(sex), int(chestpain), restingbp, cholestrol, int(fastingbs), int(restingecg), maxhr, int(exerciseangina), int(oldpeak), int(stslope)]), 0))
    prediction = svm_model.predict(inputs)
    print("Final pred", np.squeeze(prediction, -1))
    if prediction == 0:
        k = 'safe. Continue the healthy lifestyle :D'
    elif prediction == 1:
        k = 'at risk. Treat him/her immediately'
    st.write(f"{st.session_state.name} is " + k)

    print(inputs)
# In[13]:
#50	1	1	170	209	0	2	116	0	0.0	2

# inputs = np.expand_dims([40, 1, 140, 289, 0, 1, 172, 1, 0, 1, 0, 0, 0, 1, 0, 0], 0)
# svm_model.predict(inputs)


# In[ ]:




