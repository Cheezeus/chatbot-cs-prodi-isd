import pandas as pd
import streamlit as st
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core import Document
import fitz  # PyMuPDF

import nest_asyncio
nest_asyncio.apply()

# Initialize node parser
splitter = SentenceSplitter(chunk_size=512)

import sys
import logging

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

system_prompt = """
You are a multi-lingual expert system who has knowledge, based on 
real-time data. You will always try to be helpful and try to help them 
answering their question. If you don't know the answer, say that you DON'T
KNOW.
jawab dalam bahasa indonesia

Tugas Anda adalah untuk menjadi customer service infor-sib-dsa yang membantu 
user.

berikanlah kontak nomor wa dan email dosen dan begitupun sebaliknya berikan 
nama dosen jika ditanya nomor wa.

jawablah pertanyaan sesuai buku pedoman dan jangan diluar itu.

jika di beri pertanyaan diluar dari data yang dimiliki bilang tidak tau.

jawab spesifik mungkin sesuai teks pdf data yang ada dan jangan jawab random.


"""

Settings.llm = Ollama(model="llama3.1:latest", base_url="http://127.0.0.1:11434", system_prompt=system_prompt)
Settings.embed_model = OllamaEmbedding(base_url="http://127.0.0.1:11434", model_name="mxbai-embed-large:latest")

@st.cache_resource(show_spinner=False)
def load_data(_arg=None, vector_store=None):
    with st.spinner(text="Loading and indexing – hang tight! This should take a few minutes."):
        # Read & load PDF documents from folder using SimpleDirectoryReader
        reader = SimpleDirectoryReader(input_dir="./docs", recursive=True)
        documents = reader.load_data()
        st.write(f"Loaded pdf with {len(documents)} rows")

    # Read CSV data
    csv_file_path = "./docs/kontak.csv"
    csv_data = None
    if csv_file_path:
        try:
            df = pd.read_csv(csv_file_path)
            st.write(f"Loaded CSV with {len(df)} rows")
            csv_data = df
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            return None

    # Process CSV data into documents
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

    # Combine PDF and CSV documents
    documents.extend(csv_documents)

    if vector_store is None:
        index = VectorStoreIndex.from_documents(documents)
    return index

def create_query_engine(_arg=None, index=None):
    return index.as_query_engine(chat_mode="condense_plus_context", verbose=True)

# Main Program
st.title("CS INFOR-SIB-DSA")

index = load_data()
query_engine = create_query_engine(index=index)

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
    st.session_state.chat_engine = CondensePlusContextChatEngine(
        verbose=True,
        system_prompt=system_prompt,
        context_prompt=(
            "Anda adalah customer service yang memberi informasi terhadap kontak dosen.\n"
            "Format dokumen pendukung: Nama Dosen, no Wa, email"
            "anda juga memberi informasi sesuai buku pedoman"
            "Ini adalah dokumen yang mungkin relevan terhadap konteks:\n\n"
            "{context_str}"
            "\n\nInstruksi: Gunakan riwayat obrolan sebelumnya, atau konteks di atas, untuk berinteraksi dan membantu pengguna. Jika tidak menemukan dosen,nomor wa atau email yang sesuai, maka katakan tidak tau"
        ),
        condense_prompt="""
Diberikan suatu percakapan (antara User dan Assistant) dan pesan lanjutan dari User,
Ubah pesan lanjutan menjadi pertanyaan independen yang mencakup semua konteks relevan
dari percakapan sebelumnya. Pertanyaan independen/standalone question cukup 1 kalimat saja. Informasi yang penting adalah Nama Dosen, no wa, email. Contoh standalone question: "nomor wa adi wibowo".

<Chat History>
{chat_history}

<Follow Up Message>
{question}

<Standalone question>""",
        memory=memory,
        retriever=index.as_retriever(similarity_top_k=10),
        llm=Settings.llm
    )

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
