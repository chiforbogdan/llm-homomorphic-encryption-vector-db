import dspy
from typing import List, Union, Optional, Callable, Tuple
from dsp.utils import dotdict
import time
from pydantic import BaseModel, Field
import datetime
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
import tenseal as ts
import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dspy.retrieve.faiss_rm import FaissRM
from evaluate import load
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from enum import Enum

DOT_PROD_THRESHOLD = 0.4
SENTENCE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
HE_VECTOR_DB_CLUSTERS = 20
DIRECTORY_PATH_PATENT = "patent_disclosure_conversation"
DIRECTORY_PATH_REGULAR = "regular_conversation"

LLAMA_3B_MODEL="ollama_chat/llama3.2:3b"
CHAT_GPT_MODEL="gpt-4o-mini"

OPENAI_API_KEY=""

class EvalModel(Enum):
   CHAT_GPT=1
   LLAMA_3B=2

class EmbeddingsType(Enum):
   FAISS=1
   HE=2

class SimilarityType(Enum):
   BERT=1
   LLM_AS_JUDGE=2

############################################################
# Homorphic Encryption (HE) Embedding database (server side)
############################################################
class HEEmbeddingsDatabase:
  embeddings: np.ndarray
  docs: List[str]
  cluster_centers = []
  cluster_ids_x = []

  def __init__(self, docs, model_name, num_clusters):
    model = SentenceTransformer(model_name)
    self.docs = docs
    self.embeddings = model.encode(docs)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(self.embeddings)
    self.cluster_centers = kmeans.cluster_centers_
    self.cluster_ids_x = kmeans.labels_

  def get_passages(self, enc_query_serialized, public_context_serialized):
    pub_context = ts.context_from(public_context_serialized)
    enc_query = ts.ckks_vector_from(pub_context, enc_query_serialized)

    dotprods = []
    for embedding in self.embeddings:
      dotprods.append(enc_query.dot(embedding))

    return [(i, dotprod.serialize()) for i, dotprod in enumerate(dotprods)]

  def get_index_passages(self, enc_query_serialized, public_context_serialized):
    pub_context = ts.context_from(public_context_serialized)
    enc_query = ts.ckks_vector_from(pub_context, enc_query_serialized)

    dotprods = []
    for embedding in self.cluster_centers:
      dotprods.append(enc_query.dot(embedding))

    return [(i, dotprod.serialize()) for i, dotprod in enumerate(dotprods)]

  def get_passages_from_cluster(self, enc_query_serialized, public_context_serialized, cluster_id):
    pub_context = ts.context_from(public_context_serialized)
    enc_query = ts.ckks_vector_from(pub_context, enc_query_serialized)

    dotprods = []
    for i, cluster in enumerate(self.cluster_ids_x):
      if cluster != cluster_id:
        continue
      dotprods.append((i, enc_query.dot(self.embeddings[i])))

    return [(dotprod[0], dotprod[1].serialize()) for dotprod in dotprods]

  def get_passage_at_index(self, index):
    return self.docs[index]

############################################################
# Homorphic Encryption (HE) Retriever model (client side)
############################################################
class Passage:
  long_text: str
  def __init__(self, text):
    self.long_text = text

class HERetriever(dspy.Retrieve):
  model: SentenceTransformer
  he_db: HEEmbeddingsDatabase
  context: ts.Context
  public_context: bytes
  index_retrieve: bool

  def __init__(self, he_db, model_name, index_retrieve):
    self.model = SentenceTransformer(model_name)
    self.he_db = he_db
    self.index_retrieve = index_retrieve

    self.context = ts.context(
      ts.SCHEME_TYPE.CKKS,
      poly_modulus_degree = 8192,
      coeff_mod_bit_sizes = [60, 40, 40, 60]
    )
    self.context.generate_galois_keys()
    self.context.global_scale = 2**40

    secret_context = self.context.serialize(save_secret_key = True)

    self.context.make_context_public()
    self.public_context = self.context.serialize()
    self.context = ts.context_from(secret_context)

  def get_dot_prods_sorted(self, dot_prods):
    dotprods_dec = []
    for dotprod in dot_prods:
      dotprods_dec.append((dotprod[0], ts.ckks_vector_from(self.context, dotprod[1]).decrypt()))

    return sorted(dotprods_dec, key = lambda x: x[1], reverse=True)

  def forward(self, query:str, k:Optional[int]) -> dspy.Prediction:
    query_embeddings = self.model.encode(query)
    enc_query_embeddings = ts.ckks_vector(self.context, query_embeddings)

    if self.index_retrieve:
      dotprods = self.he_db.get_index_passages(enc_query_embeddings.serialize(), self.public_context)
      dot_prods_sorted = self.get_dot_prods_sorted(dotprods)
      dotprods = self.he_db.get_passages_from_cluster(enc_query_embeddings.serialize(), self.public_context, dot_prods_sorted[0][0])
    else:
      dotprods = self.he_db.get_passages(enc_query_embeddings.serialize(), self.public_context)

    dot_prods_sorted = self.get_dot_prods_sorted(dotprods)

    passages = []
    for i in range(0, min(k, len(dot_prods_sorted))):
      # Apply a threshold to remove the content with low score
      # if dot_prods_sorted[i][1][0] < DOT_PROD_THRESHOLD:
      #   continue
      passage = self.he_db.get_passage_at_index(dot_prods_sorted[i][0])
      passages.append(Passage(passage))

    return passages

############################################################
# RAG pipeline
############################################################
class MailPrivateData(dspy.Signature):
  """Tell if the email message contains company private information. If yes, provide a suggestion with a sanitized (rephrased) version of the mail. Please make the distinction between a regular conversation between friend and the one which discloses private information about a company."""
  context = dspy.InputField(desc="May contain company private information. If this is empty, mail does not contain any private information.")
  mail = dspy.InputField(desc="Email message")
  is_private = dspy.OutputField(desc="Indicates if the mail message contains any private data or not. Output true if message contains any private data or false otherwise.")
  sanitized_mail_suggestion = dspy.OutputField(desc="Contains a rewritten suggestion of the original mail message without giving away any private information. Mail should be coherent enough and generic.")

class RAG(dspy.Module):
  retrieve_time_values = []
  toolbox = []

  def __init__(self, num_passages=3):
    super().__init__()

    self.retrieve = dspy.Retrieve(k = num_passages)
    self.generate_answer = dspy.ReAct(MailPrivateData, tools=[])

  def contextToList(self, contextPrediction: dspy.Prediction) -> list[str]:
    context = []
    for passage in contextPrediction:
      context.append(passage)
    return context

  def retrieve_time_avg(self):
    sum = 0.0
    for t in self.retrieve_time_values:
      sum += t
    return sum / len(self.retrieve_time_values)

  def forward(self, mail):
    start_time = time.time()
    context = self.retrieve(mail)
    end_time = time.time()
    self.retrieve_time_values.append((end_time - start_time) * 1000)

    prediction = self.generate_answer(context=self.contextToList(context), mail=mail)

    return dspy.Prediction(context=context, answer={"is_private": prediction.is_private,
                                                    "sanitized_mail_suggestion": prediction.sanitized_mail_suggestion})

############################################################
# RAG document chunking
############################################################
def len_func(text):
  return len(text)

def get_chunk_documents(text) -> list[str]:
  text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n","\n", " ", "."],
    chunk_size = 1000,
    chunk_overlap = 100,
    length_function = len_func,
    is_separator_regex=False
  )

  return text_splitter.create_documents(texts = [text])

############################################################
# Evaluate similarity between HE and FAISS
############################################################
def compute_bert_score(patent_mail_data, frm, he_rm):
    bertscore = load("bertscore")
    
    k = 3
    precision = [0.0] * k
    recall = [0.0] * k
    f1 = [0] * k

    for mail in patent_mail_data:
        mail_body = mail['patent_mail'].replace('\n','\\n')
        faiss_passages = frm(mail_body, k = 3)
        he_passages  = he_rm(mail_body, k = 3)

        faiss_passages_text = [p['long_text'] for p in faiss_passages]
        he_passages_text = [p.long_text for p in he_passages]
        min_len = min(len(faiss_passages_text), len(he_passages_text))
        faiss_passages_text = faiss_passages_text[:min_len]
        he_passages_text = he_passages_text[:min_len]

        results = bertscore.compute(predictions=he_passages_text, references=faiss_passages_text, lang="en")
        for i in range(min_len):
            precision[i] += results['precision'][i]
            recall[i] += results['recall'][i]
            f1[i] += results['f1'][i]

    for i in range(k):
        precision[i] /= len(patent_mail_data)
        recall[i] /= len(patent_mail_data)
        f1[i] /= len(patent_mail_data)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")

def llm_as_judge_score(patent_mail_data, frm, he_rm):
    llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
            """
            You will be given 2 texts.
            Your task is to provide a scoring on the semantic similarity of the 2 texts.
            Give your answer on a scale of 1 to 4, where 1 means that the texts are completely unrelated and 4 means that the answers are semantically similar.

            Here is the scale you should use to build your answer:
            1: The texts are completely unrelated.
            2: The texts have some similarities but the overall semantic meaning is different.
            3: The texts have a good semantic similarity, but the meaning is not the same.
            4: The texts have more or less the same semantic meaning.

            You MUST provide a single score value and nothing else.
            """
            ),
            HumanMessagePromptTemplate.from_template("The 2 texts are the following:\n\nText 1:\n {text1}\n\nText 2:\n {text2}"),
        ]
    )
    
    k = 3
    scores = [0.0] * k

    for mail in patent_mail_data:
        mail_body = mail['patent_mail'].replace('\n','\\n')
        faiss_passages = frm(mail_body, k = 3)
        he_passages  = he_rm(mail_body, k = 3)

        faiss_passages_text = [p['long_text'] for p in faiss_passages]
        he_passages_text = [p.long_text for p in he_passages]

        for i, (text1, text2) in enumerate(zip(faiss_passages_text, he_passages_text)):
            scores[i] += int(llm.invoke(prompt.format_prompt(text1=text1, text2=text2)).content)
            
    for i in range(0, k):
        scores[i] /= len(patent_mail_data)

    print(f"LLM as judge: {scores}")

############################################################
# Evaluate accuracy
############################################################
def compute_accuracy_patent(patent_mail_data):
    rag = RAG()

    correct_pred = 0
    for mail in patent_mail_data:
        mail_body = mail['patent_mail'].replace('\n','\\n')
        pred = rag(mail_body)
        if pred.answer['is_private'] == 'true':
            correct_pred += 1
        print(f"Accuracy: {correct_pred} of {len(patent_mail_data)} ({100 * correct_pred/len(patent_mail_data)})")

    print(f"Avg retrieve time: {rag.retrieve_time_avg()}")

def compute_accuracy_regular(mail_data):
    rag = RAG()

    correct_pred = 0
    for mail in mail_data:
        mail_body = mail.replace('\n','\\n')
        pred = rag(mail_body)
        if pred.answer['is_private'] == 'false':
            correct_pred += 1
        print(f"Accuracy: {correct_pred} of {len(mail_data)} ({100 * correct_pred/len(mail_data)})")

    print(f"Avg retrieve time: {rag.retrieve_time_avg()}")

def chunk_patent_data(limit: int = 1000):
  count = 0
  docs = []
  patent_mail_data = []
  for filename in os.listdir(DIRECTORY_PATH_PATENT):
    file_path = os.path.join(DIRECTORY_PATH_PATENT, filename)
    if not file_path.endswith('.json'):
      continue
    count += 1
    if count >= limit:
      break
    with open(file_path, 'r') as file:
      data = json.load(file)
      patent_mail_data.append(data)
      page_content_desc = [doc.page_content for doc in get_chunk_documents(data['patent_description'])]
      page_content_claim = [doc.page_content for doc in get_chunk_documents(data['patent_claims'])]
      docs.extend(page_content_desc)
      docs.extend(page_content_claim)

  return docs, patent_mail_data

def patent_evaluate_accuracy(eval_model, embeddings_type, limit: int = 1000):
    docs, patent_mail_data = chunk_patent_data()

    match embeddings_type:
      case EmbeddingsType.FAISS:
        rm = FaissRM(docs)
      case EmbeddingsType.HE:
        he_db = HEEmbeddingsDatabase(docs, SENTENCE_EMBEDDING_MODEL, HE_VECTOR_DB_CLUSTERS)
        rm = HERetriever(he_db, SENTENCE_EMBEDDING_MODEL, True)

    match eval_model:
      case EvalModel.CHAT_GPT:
        llm = dspy.OpenAI(model=CHAT_GPT_MODEL, api_key=OPENAI_API_KEY)
      case EvalModel.LLAMA_3B:
        llm = dspy.LM(LLAMA_3B_MODEL, api_base='http://localhost:11434', api_key='')
    
    dspy.settings.configure(lm=llm, rm = rm, experimental=True)

    compute_accuracy_patent(patent_mail_data)

def regular_evaluate_accuracy(eval_model, embeddings_type):
    docs, _ = chunk_patent_data()

    match embeddings_type:
      case EmbeddingsType.FAISS:
        rm = FaissRM(docs)
      case EmbeddingsType.HE:
        he_db = HEEmbeddingsDatabase(docs, SENTENCE_EMBEDDING_MODEL, HE_VECTOR_DB_CLUSTERS)
        rm = HERetriever(he_db, SENTENCE_EMBEDDING_MODEL, True)

    regular_mail_data = []
    for filename in os.listdir(DIRECTORY_PATH_REGULAR):
        file_path = os.path.join(DIRECTORY_PATH_REGULAR, filename)
        if not file_path.endswith('.json'):
            continue
        with open(file_path, 'r') as file:
            data = file.read()
            regular_mail_data.append(data)

    match eval_model:
      case EvalModel.CHAT_GPT:
          llm = dspy.OpenAI(model=CHAT_GPT_MODEL, api_key=OPENAI_API_KEY)
      case EvalModel.LLAMA_3B:
        llm = dspy.LM(LLAMA_3B_MODEL, api_base='http://localhost:11434', api_key='')
    
    dspy.settings.configure(lm=llm, rm = rm, experimental=True)

    compute_accuracy_regular(regular_mail_data)

def patent_evaluare_similarity(similarity_type):
  docs, patent_mail_data = chunk_patent_data()
  frm = FaissRM(docs)
  
  he_db = HEEmbeddingsDatabase(docs, SENTENCE_EMBEDDING_MODEL, HE_VECTOR_DB_CLUSTERS)
  he_rm = HERetriever(he_db, SENTENCE_EMBEDDING_MODEL, True)

  match similarity_type:
    case SimilarityType.BERT:
      compute_bert_score(patent_mail_data, frm, he_rm)
    case SimilarityType.LLM_AS_JUDGE:
      llm_as_judge_score(patent_mail_data, frm, he_rm)

if __name__ == "__main__":
  #patent_evaluate_accuracy(EvalModel.CHAT_GPT, EmbeddingsType.FAISS)
  #patent_evaluate_accuracy(EvalModel.CHAT_GPT, EmbeddingsType.HE)
  #patent_evaluate_accuracy(EvalModel.LLAMA_3B, EmbeddingsType.FAISS)
  patent_evaluate_accuracy(EvalModel.LLAMA_3B, EmbeddingsType.HE)

  #patent_evaluare_similarity(SimilarityType.BERT)
  #patent_evaluare_similarity(SimilarityType.LLM_AS_JUDGE)
   
  #regular_evaluate_accuracy(EvalModel.LLAMA_3B, EmbeddingsType.FAISS)
  #regular_evaluate_accuracy(EvalModel.LLAMA_3B, EmbeddingsType.HE)
  #regular_evaluate_accuracy(EvalModel.CHAT_GPT, EmbeddingsType.FAISS)
  #regular_evaluate_accuracy(EvalModel.CHAT_GPT, EmbeddingsType.HE)
