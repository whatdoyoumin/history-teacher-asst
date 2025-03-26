import streamlit as st
from util.utility import check_password


if not check_password():  
    st.stop()
    
st.title("Methodology Flow Chart")

st.image("./images/flowchart_query.png")

