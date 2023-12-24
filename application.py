import sys
import streamlit as st
import pdfplumber
from Resume_scanner import compare


def extract_pdf_data(file_path):
    data = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                data += text
    return data


def extract_text_data(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return data


# Command-line argument processing
if len(sys.argv) > 1:

    if len(sys.argv) == 3:
        resume_path = sys.argv[1]
        jd_path = sys.argv[2]

        resume_data = extract_pdf_data(resume_path)
        jd_data = extract_text_data(jd_path)

        result = compare([resume_data], jd_data, flag='HuggingFace-BERT')

    sys.exit()

# Sidebar
flag = 'HuggingFace-BERT'
with st.sidebar:
    st.markdown('**Which embedding do you want to use**')
    options = st.selectbox('Which embedding do you want to use',
                           ['HuggingFace-BERT', 'Doc2Vec'],
                           label_visibility="collapsed")
    flag = options

# Main content
tab1, tab2 = st.tabs(["**Home**", "**Results**"])

# Tab Home
with tab1:
    st.title("Applicant Tracking System")
    uploaded_files = st.file_uploader(
        '**Choose your resume.pdf file:** ', type="pdf", accept_multiple_files=True)
    JD = st.text_area("**Enter the job description:**")
    comp_pressed = st.button("Compare!")
    if comp_pressed and uploaded_files:
        # Streamlit file_uploader gives file-like objects, not paths
        uploaded_file_paths = [extract_pdf_data(
            file) for file in uploaded_files]
        score = compare(uploaded_file_paths, JD, flag)

# Tab Results
with tab2:
    st.header("Results")
    my_dict = {}
    if comp_pressed and uploaded_files:
        for i in range(len(score)):
            my_dict[uploaded_files[i].name] = score[i]
        sorted_dict = dict(sorted(my_dict.items()))
        for i in sorted_dict.items():
            with st.expander(str(i[0])):
                st.write("Score is: ", i[1])
