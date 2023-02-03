# importing required modules
#from PyPDF2 import PdfReader
import streamlit as st
import pdfplumber
from Resume_Scanner import cosine_similarity, compare

JD_embeddings = None
resume_embeddings = None

#Sidebar
flag = 'HuggingFace-BERT'
with st.sidebar:
    st.markdown('**Which embedding do you want to use**')
    options = st.selectbox('',
    ['HuggingFace-BERT', 'Doc2Vec'])
    flag = options

#main content
st.title("Applicant Tracking System")
uploaded_file = st.file_uploader('Choose your resume.pdf file: ', type="pdf")
st.write(uploaded_file)
JD = st.text_area("Enter the job description: ")

if st.button("Compare!"):
    compare(uploaded_file, JD, flag)
