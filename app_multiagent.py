import pandas as pd
import streamlit as st
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core import SimpleDirectoryReader

from prompts import SYSTEM_PROMPT, CONTEXT_PROMPT, CONDENSE_PROMPT

import os
import sys
import logging
import nest_asyncio
nest_asyncio.apply()

# Initialize node parser
splitter = SentenceSplitter(chunk_size=512)

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

Settings.system_prompt = SYSTEM_PROMPT
Settings.llm = Ollama(model="llama3.1:latest", base_url="http://127.0.0.1:11434")
Settings.embed_model = OllamaEmbedding(base_url="http://127.0.0.1:11434", model_name="mxbai-embed-large:latest")

@st.cache_resource(show_spinner=False)
def load_data(data_dir=None, csv_dir=None, vector_store=None):
    with st.spinner(text="Loading and indexing – hang tight! This should take a few minutes."):
        documents = []
        
        # Load Markdown files for KnowledgeAgent
        if data_dir:
            documents = SimpleDirectoryReader(data_dir).load_data()
            st.write(f"Loaded {len(documents)} Markdown files from {data_dir}")

        # Load CSV files for CSVAgent
        if csv_dir:
            csv_data = []
            for csv_file in os.listdir(csv_dir):
                if csv_file.endswith('.csv'):
                    csv_file_path = os.path.join(csv_dir, csv_file)
                    try:
                        df = pd.read_csv(csv_file_path, skipinitialspace=True, on_bad_lines='skip')
                        st.write(f"Loaded CSV {csv_file} with {len(df)} rows")
                        unique_documents = {}
                        for _, row in df.iterrows():
                            doc_str = str(row.to_dict())
                            if doc_str.strip() and doc_str not in unique_documents:
                                document = Document(
                                    content=doc_str,
                                    metadata={
                                        "dosen": row.get('Nama Dosen', 'Unknown'),
                                        "email": row.get('email', 'Unknown'),
                                        "nomor wa": row.get('No WA', 'Unknown'),
                                    }
                                )
                                unique_documents[doc_str] = document
                        csv_data.extend(unique_documents.values())
                    except Exception as e:
                        st.error(f"Error loading CSV {csv_file}: {e}")
            documents.extend(csv_data)

        st.write(f"Total documents after merging: {len(documents)}")

    if vector_store is None:
        index = VectorStoreIndex.from_documents(documents)
    return index

def create_chat_engine(index):
    reranker = SentenceTransformerRerank(top_n=6, model="BAAI/bge-reranker-large")
    memory = ChatMemoryBuffer.from_defaults(token_limit=16384)
    retriever = QueryFusionRetriever(
        [
            index.as_retriever(similarity_top_k=5),
            BM25Retriever.from_defaults(
                docstore=index.docstore, similarity_top_k=5
            ),
        ],
        num_queries=4,
        similarity_top_k=10,
        use_async=True,
    )
    chat_engine = CondensePlusContextChatEngine(
        verbose=True,
        system_prompt=Settings.system_prompt,
        context_prompt=CONTEXT_PROMPT,
        condense_prompt=CONDENSE_PROMPT,
        memory=memory,
        retriever=retriever,
        node_postprocessors=[reranker],
        llm=Settings.llm
    )
    return chat_engine

def multi_agent_response(prompt, agents):
    for agent_name, agent in agents.items():
        if agent["filter"](prompt):
            st.write(f"Routing to {agent_name}...")
            response_stream = agent["engine"].stream_chat(prompt)  # Streamed response
            response_content = "".join(response_stream.response_gen)  # Consume the generator
            return response_content

    return "Sorry, I couldn't find an appropriate agent to handle your query."


# Main Program
st.title("CS INFOR-SIB-DSA: ")

# Load data only for the KnowledgeAgent (Markdown) and CSVAgent (CSV)
index_knowledge = load_data(data_dir="./docs/pedoman")  # for KnowledgeAgent (Markdown)
index_csv = load_data(csv_dir="./docs/csv")  # for CSVAgent (CSV)

# Initialize chat history if empty
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Halo! ada yang bisa dibantu?"}
    ]

# Initialize the chat engines
if "chat_engines" not in st.session_state.keys():
    st.session_state.chat_engines = {
        "KnowledgeAgent": {
            "filter": lambda prompt: "kurikulum" in prompt or "pedoman" in prompt,
            "engine": create_chat_engine(index_knowledge)  # KnowledgeAgent engine using MD data
        },
        "CSVAgent": {
            "filter": lambda prompt: "dosen" in prompt or "email" in prompt or "nomor wa" in prompt or "kontak" in prompt,  # Trigger for CSV agent
            "engine": create_chat_engine(index_csv)  # CSVAgent engine using CSV data
        },
        "GeneralAgent": {
            "filter": lambda prompt: True,  # Default fallback agent
            "engine": create_chat_engine(index_knowledge)  # General agent using MD data (fallback to MD)
        },
        
    }

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Berpikir..."):
            response = multi_agent_response(prompt, st.session_state.chat_engines)
            st.markdown(response)

    # Add response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
