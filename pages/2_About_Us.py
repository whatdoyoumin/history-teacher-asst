import streamlit as st
from util.utility import check_password

# Do not continue if check_password is not True.  
if not check_password():  
     st.stop()

st.title("About Us")

st.subheader("Problem Statement")
with st.container(border=True):
    st.write("""insert here""")

st.subheader("Proposed Solution")
with st.container(border=True):
    st.write("""With the use of LLM, it is expected to cut down the manual effort required to look and evaluate sources""")

st.subheader("Impact")
with st.container(border=True):
    st.write("""Time-Savings for Teachers""")
        
st.subheader("Data Sources")
with st.container(border=True):
    st.markdown("""
    Infopedia,
    Stories from Roots,
    Sec 1 & 2 Textbooks""")

st.subheader("Application Features")
with st.container(border=True):
    st.write("There are two main features in this application. Click below to find out more:")
    st.page_link("History_Assistant.py", label="- History Assistant", icon="🚀")
    #st.page_link("pages/1_Query_Sites.py", label="- Query Site Reports", icon="🚀")
