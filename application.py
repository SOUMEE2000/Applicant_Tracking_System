# importing required modules
import streamlit as st
import pdfplumber
from Resume_Scanner import compare

comp_pressed = False
score = 0

#Sidebar
flag = 'HuggingFace-BERT'
with st.sidebar:
    st.markdown('**Which embedding do you want to use**')
    options = st.selectbox('Which embedding do you want to use',
    ['HuggingFace-BERT', 'Doc2Vec'], label_visibility="collapsed")
    flag = options

#main content
tab1, tab2 = st.tabs(["**Home**","**Results**"])

with tab1:
    st.title("Applicant Tracking System")
    uploaded_file = st.file_uploader('**Choose your resume.pdf file:** ', type="pdf")
    st.write(uploaded_file)
    st.write("")
    JD = st.text_area("**Enter the job description:**")
    comp_pressed = st.button("Compare!")
    if comp_pressed:
        score = compare(uploaded_file, JD, flag)

with tab2:
    st.header("Results")
    if comp_pressed:
        st.write("Cosine similarity is: ", score)
    else:
        st.write("#### Throw in some Resumes to see the score :)")
