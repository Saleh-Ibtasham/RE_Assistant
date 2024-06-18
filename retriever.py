import os
from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.vectorstores import Chroma, FAISS
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings

import torch
from sentence_transformers import SentenceTransformer
from torch.nn import functional as F
import numpy as np

from pymilvus import (FieldSchema, DataType, CollectionSchema, Collection)

from pymilvus import connections,MilvusClient

from langchain_community.llms import LlamaCpp
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationSummaryBufferMemory, CombinedMemory
from langchain.chains import LLMChain
from langchain_core.documents.base import Document

import time
# from transformers import AutoTokenizer, pipeline, AutoConfig, AutoModelForCausalLM, AutoModel
from tqdm import tqdm
import json
from langchain_community.llms import CTransformers
import tiktoken

import fitz
import re, glob
import pymupdf4llm, pdfplumber
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.md import partition_md
from unstructured.chunking.title import chunk_by_title
from unstructured.chunking.basic import chunk_elements
import subprocess
from collections import Counter
import gc
import pickle
import itertools


class MilvusRetriever():
    def __init__(self,
                connection_string: str):
        # ENDPOINT=f'http://localhost:19530'
        self.client = MilvusClient(
          uri=connection_string)
        self.fields = []
    def create_collection(self, collection_name: str):
        # 1. Define a minimum expandable schema.
        fields = [
           FieldSchema("pk", DataType.INT64, is_primary=True, auto_id=True),
           FieldSchema("vector", DataType.FLOAT_VECTOR, dim=EMBEDDING_LENGTH),]
        schema = CollectionSchema(
           fields,
           enable_dynamic_field=True,)
        
        index_params = client.prepare_index_params()
        
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )
        
        client.create_collection(collection_name=COLLECTION_NAME, schema=schema, index_params=index_params)
        
        # 2. Create the collection.
        # mc = Collection("MilvusDocs", schema)
        
        # 3. Index the collection.
        # mc.create_index(
        #    field_name="vector",
        #    index_params={
        #        "index_type": "AUTOINDEX",
        #        "metric_type": "COSINE",})