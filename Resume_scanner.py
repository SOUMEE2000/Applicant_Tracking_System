from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
import streamlit as st
import pdfplumber

def extract_data(feed):
    data = ""
    with pdfplumber.open(feed) as pdf:
        pages = pdf.pages
        for page in pages:
            data += page.extract_text()
    return data # build more code to return a dataframe

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

@st.cache
def get_embeddings(sentences):

  # Load model from HuggingFace Hub
  tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
  model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')

  # Tokenize sentences
  encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=512)

  # Compute token embeddings
  with torch.no_grad():
      model_output = model(**encoded_input)

  # Perform pooling. In this case, max pooling.
  embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

  print("Sentence embeddings:")
  print(embeddings)
  return embeddings


def cosine(embeddings1, embeddings2):
  # get the match percentage
  matchPercentage = cosine_similarity(np.array(embeddings1), np.array(embeddings2))
  matchPercentage = np.round(matchPercentage, 4)*100 # round to two decimal
  print("Your resume matches about" + str(matchPercentage[0])+ "% of the job description.")
  return matchPercentage[0][0]

def compare(uploaded_file, JD):

    JD_embeddings = None
    resume_embeddings = None

    if uploaded_file is not None:
        df = extract_data(uploaded_file)
        resume_embeddings = get_embeddings(df)

    if JD is not None:
        JD_embeddings = get_embeddings(JD)

    if JD_embeddings is not None and resume_embeddings is not None:
        cos = cosine(resume_embeddings, JD_embeddings)
        st.write("Cosine similarity is: ", cos)
