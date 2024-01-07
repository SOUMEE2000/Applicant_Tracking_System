import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec
import nltk
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
import streamlit as st

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


@st.cache_resource
def get_HF_embeddings(sentences):

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

  # print("Sentence embeddings:")
  # print(embeddings)
  return embeddings


@st.cache_data
def get_doc2vec_embeddings(JD, text_resume):
    nltk.download("punkt")
    data = [JD]
    resume_embeddings = []
    
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]
    #print (tagged_data)

    model = gensim.models.doc2vec.Doc2Vec(vector_size=512, min_count=3, epochs=80)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=80)
    JD_embeddings = np.transpose(model.docvecs['0'].reshape(-1,1))

    for i in text_resume:
        text = word_tokenize(i.lower())
        embeddings = model.infer_vector(text)
        resume_embeddings.append(np.transpose(embeddings.reshape(-1,1)))
    return (JD_embeddings, resume_embeddings)


def cosine(embeddings1, embeddings2):
  # get the match percentage
  score_list = []
  for i in embeddings1:
      matchPercentage = cosine_similarity(np.array(i), np.array(embeddings2))
      matchPercentage = np.round(matchPercentage, 4)*100 # round to two decimal
      print("Your resume matches about" + str(matchPercentage[0])+ "% of the job description.")
      score_list.append(str(matchPercentage[0][0]))
  return score_list
