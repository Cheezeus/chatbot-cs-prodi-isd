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
def load_data(_arg=None, vector_store=None):
    with st.spinner(text="Loading and indexing – hang tight! This should take a few minutes."):
        # md file reader
        md_dir = "./docs"
        documents = []
        for filename in os.listdir(md_dir):
            if filename.endswith(".md"):
                file_path = os.path.join(md_dir, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    st.write(f"Loaded Markdown file: {filename}")  # Debug line
                    document = Document(
                        content=content,
                        metadata={"filename": filename}
                    )
                    documents.append(document)
        
        st.write(f"Loaded {len(documents)} Markdown files")

        # csv
        csv_file_path = "./docs/Dataset Kontak Dosen.csv"
        csv_data = None
        if csv_file_path:
            try:
                df = pd.read_csv(csv_file_path)
                st.write(f"Loaded CSV with {len(df)} rows")
                csv_data = df
            except Exception as e:
                st.error(f"Error loading CSV: {e}")
                return None

        csv_documents = []
        if csv_data is not None:
            unique_documents = {}
            for _, row in csv_data.iterrows():
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

            csv_documents = list(unique_documents.values())

        documents.extend(csv_documents)

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
        num_queries=1,
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

# Main Program
st.title("CS INFOR-SIB-DSA")

index = load_data()

# Initialize chat history if empty
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Halo! ada yang bisa dibantu?"}
    ]

# Initialize the chat engine
if "chat_engine" not in st.session_state.keys():
    # Initialize with custom chat history
    init_history = [
        ChatMessage(role=MessageRole.ASSISTANT, content="Halo! ada yang bisa dibantu?"),
    ]
    memory = ChatMemoryBuffer.from_defaults(token_limit=16384)
    st.session_state.chat_engine = create_chat_engine(index)

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
            response_stream = st.session_state.chat_engine.stream_chat(prompt)
            st.write_stream(response_stream.response_gen)

    # Add user message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_stream.response})
