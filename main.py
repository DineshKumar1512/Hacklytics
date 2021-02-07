import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import streamlit as st
import pandas as pd
st.write("""
# Malicious Vs Benign Website







""")

df=pd.read_csv('dataset.csv')


st.line_chart(df)

df.isnull().sum()

df1=df.dropna()

labelencoder=LabelEncoder()
df1['URL_N']=labelencoder.fit_transform(df1['URL'])

labelencoder=LabelEncoder()
df1['CHARSET_N']=labelencoder.fit_transform(df1['CHARSET'])

labelencoder=LabelEncoder()
df1['SERVER_N']=labelencoder.fit_transform(df1['SERVER'])

labelencoder=LabelEncoder()
df1['WHOIS_COUNRTY_N']=labelencoder.fit_transform(df1['WHOIS_COUNTRY'])

labelencoder=LabelEncoder()
df1['WHOIS_STATEPRO_N']=labelencoder.fit_transform(df1['WHOIS_STATEPRO'])

X=df1.drop(df1.columns[[0,3,4,6,7,8,9]], axis = 1) 
st.write("""
DataFrame of Features(x)







""")
st.dataframe(X)
st.write("""
Graph representation of X






""")
st.area_chart(X)
st.write("""
Dataframe of Label(y
)





""")
y=df1.Type
st.dataframe(y)
st.write("""
Graph representation of y










""")
st.area_chart(y)
st.write("""
  






""")

q= st.slider('Select the test data size', 0.0, 1.0,0.2)
st.write('selected values is ',q)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=q)

def NB(X_train,y_train,X_test,y_test):
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred=nb.predict(X_test)
    st.write('Confusion Matrix')
    st.write(confusion_matrix(y_test,y_pred))
    st.write('Accuracy Score')
    st.write(accuracy_score(y_test,y_pred))



def LR(X_train,y_train,X_test,y_test):
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred=lr.predict(X_test)
    st.write('Confusion Matrix')
    st.write(confusion_matrix(y_test,y_pred))
    st.write('Accuracy Score')
    st.write(accuracy_score(y_test,y_pred))

   

def RFC(X_train,y_train,X_test,y_test):
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    y_pred=rfc.predict(X_test)
    st.write('Confusion Matrix')
    st.write(confusion_matrix(y_test,y_pred))
    st.write('Accuracy Score')
    st.write(accuracy_score(y_test,y_pred))

x= st.slider('SELECT 1-3', 0, 3)
st.write('You have selected ',x)
st.write('Select 1 for Naive Bayes')
st.write('Select 2 for Logistic Regression')
st.write('Select 3 for Random Forest Classifier')
st.write("""




""")

st.write("""




""")

if x==1:
    st.write('You have selected Naive Bayes')
    st.write("""







""")
    NB(X_train,y_train,X_test,y_test)
elif x==2:
    st.write('You have selected Logistic Regression')
    st.write("""





""")
    LR(X_train,y_train,X_test,y_test)
elif x==3:
    st.write('You have selected Random Forest Classifier')
    st.write("""





""")
    RFC(X_train,y_train,X_test,y_test)
else:
    st.write('    ')
        
    
