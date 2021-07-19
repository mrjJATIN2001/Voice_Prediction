import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)

# Load the pickled model
model = pickle.load(open('voice_logisticmodel.pkl', 'rb')) 
dataset= pd.read_csv('Classification_Dataset11.csv')

x = dataset.iloc[:, :-1].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'mean', fill_value=None, verbose=1, copy=True)   
imputer = imputer.fit(x[:, 1:8 ])
x[:, 1:8 ]= imputer.transform(x[:, 1:8 ])

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

def predict_note_authentication(meanfreq,sd,median,IQR,skew,kurt,mode,centroid,dfrange):
  output= model.predict(sc.transform([[meanfreq,sd,median,IQR,skew,kurt,mode,centroid,dfrange]]))
  print("Gender", output)
  if output==[1]:
    prediction="Male Voice"
  else:
    prediction="Female Voice"
  print(prediction)
  return prediction

def main():
    
    html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering and Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:25px;color:white;margin-top:10px;"Machine Learning Lab</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Voice Prediction using Logistic Algorithm")
    meanfreq = st.text_input("Mean Frequency","")
    sd = st.text_input("SD","")
    median = st.text_input("Median","")
    IQR = st.text_input( "IQR","")
    skew = st.text_input( "Skew","")
    kurt = st.text_input( "Kurt","")
    mode = st.text_input( "Mode","")
    centroid= st.text_input( "Centroid","")
    dfrange= st.text_input( "DfRange","")
    resul=""
    if st.button("Prediction"):
      result=predict_note_authentication(meanfreq,sd,median,IQR,skew,kurt,mode,centroid,dfrange)
      st.success('Model has predicted {}'.format(result))
    if st.button("About"):
      st.header("Developed by Jatin Tak")
      st.subheader("Student , Department of Computer Engineering")
    html_temp = """
    <div class="" style="background-color:orange;" >
    <div class="clearfix">           
    <div class="col-md-12" style="background-color:blue;">
    <center><p style="font-size:20px;color:white;margin-top:10px;">Machine Learning</p></center> 
    </div>
    </div>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
if __name__=='__main__':
  main()
