# importing required modules
#from PyPDF2 import PdfReader
import streamlit as st
import pdfplumber
from Resume_Scanner import get_embeddings, cosine_similarity, compare

JD_embeddings = None
resume_embeddings = None


uploaded_file = st.file_uploader('Choose your resume.pdf file: ', type="pdf")
st.write(uploaded_file)
JD = st.text_area("Enter the job description: ")

if st.button("Compare!"):
    compare(uploaded_file, JD)
