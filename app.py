import streamlit as st
st.image("dell.jpg")
st.title("Dell Global Business Center")

st.date_input("Transaction Date")
st.radio("Your deparment:", ['AI','DS','QS','Logistic'])

st.camera_input("Case reported")