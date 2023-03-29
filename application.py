# importing required modules
import streamlit as st
import pdfplumber
from Resume_scanner import compare

# global values
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

# Tab Home
with tab1:
    st.title("Applicant Tracking System")
    uploaded_files = st.file_uploader('**Choose your resume.pdf file:** ', type="pdf", accept_multiple_files = True)
    #st.write(uploaded_files)
    st.write("")
    JD = st.text_area("**Enter the job description:**")
    comp_pressed = st.button("Compare!")
    if comp_pressed:
        #st.write(uploaded_files[0].name)
        score = compare(uploaded_files, JD, flag)

# Tab Results
with tab2:
    st.header("Results")
    my_dict = {}
    if comp_pressed:
        for i in range(len(score)):
            my_dict[uploaded_files[i].name] = score[i]
        print(my_dict)
        sorted_dict = dict(sorted(my_dict.items()))
        print(sorted_dict)
        for i in sorted_dict.items():
            with st.expander(str(i[0])):
                st.write("Score is: ", i[1])
    else:
        st.write("#### Throw in some Resumes to see the score :)")
