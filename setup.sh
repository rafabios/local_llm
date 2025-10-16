#!/bin/bash
#
# Script para configurar o ambiente do "Chat com Documentos" usando Docker.
#

# --- Configurações ---
PROJECT_DIR="meu-chat-de-documentos"

# --- Início do Script ---
echo "🚀 Iniciando a configuração v3 do ambiente 'Chat com Documentos'..."

# 1. Cria a estrutura de diretórios
echo "1. Criando a estrutura de diretórios..."
mkdir -p ${PROJECT_DIR}/documentos
cd ${PROJECT_DIR}
echo "Diretório do projeto '${PROJECT_DIR}' criado."

# 2. Cria o arquivo Dockerfile
echo "2. Gerando o Dockerfile..."
cat <<EOF > Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY app.py .
RUN pip install --no-cache-dir streamlit langchain langchain-community langchain-chroma pypdf sentence-transformers
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
EOF
echo "Dockerfile criado com sucesso."

# 3. Cria o arquivo docker-compose.yml
echo "3. Gerando o docker-compose.yml..."
cat <<EOF > docker-compose.yml
version: '3.8'
services:
  ollama:
    image: ollama/ollama
    container_name: ollama_service
    volumes:
      - ./ollama_data:/root/.ollama
    ports:
      - "11434:11434"
  meu_app:
    build: .
    container_name: app_service
    ports:
      - "8501:8501"
    volumes:
      - ./documentos:/app/documentos
      - ./chroma_db:/app/chroma_db
    depends_on:
      - ollama
EOF
echo "docker-compose.yml criado com sucesso."

# 4. Cria o arquivo da aplicação app.py (VERSÃO CORRIGIDA)
echo "4. Gerando o app.py com a correção do NameError..."
cat <<EOF > app.py
import streamlit as st
import os
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- CONFIGURAÇÕES DA APLICAÇÃO ---
DATA_PATH = "documentos/"
DB_PATH = "chroma_db/"

# --- FUNÇÕES DE BACKEND ---
def processar_documentos():
    with st.spinner('Processando documentos... Isso pode levar um tempo.'):
        loader = PyPDFDirectoryLoader(DATA_PATH)
        documentos = loader.load()
        if not documentos:
            st.warning("Nenhum documento PDF encontrado. Faça o upload de um arquivo.")
            return
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documentos)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=DB_PATH)
        vectorstore.persist()
        st.success(f"Documentos processados! O banco de dados agora contém {len(chunks)} trechos de texto.")
        st.cache_resource.clear()

@st.cache_resource
def carregar_cadeia_qa():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={'k': 3})
    llm = Ollama(model="llama3:8b", base_url="http://ollama:11434")

    template = """
    Você é um assistente de IA especialista em análise de documentos.
    Sua principal função é responder exclusivamente em português do Brasil.
    Use os trechos de contexto a seguir para responder à pergunta no final.
    Se você não sabe a resposta com base no contexto, diga que não encontrou a informação nos documentos. Não invente uma resposta.

    Contexto: {context}

    Pergunta: {question}

    Resposta em Português:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

# --- INTERFACE DO USUÁRIO ---
st.set_page_config(page_title="Chat com Documentos", layout="wide")
st.title("💬 Converse com seus Documentos")

with st.sidebar:
    st.header("Gerenciar Documentos")
    uploaded_files = st.file_uploader("Faça o upload de seus arquivos PDF aqui", type=['pdf'], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with open(os.path.join(DATA_PATH, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success("Arquivo(s) salvo(s) com sucesso!")
    if st.button("Processar Documentos"):
        processar_documentos()
    st.divider()
    st.header("Status do Banco de Dados")
    try:
        db = Chroma(persist_directory=DB_PATH, embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
        total_vectors = db._collection.count()
        st.info(f"**Total de trechos (vetores):** {total_vectors}")
    except Exception:
        st.error("Banco de dados ainda não foi criado.")

st.header("Faça sua pergunta")
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
if prompt := st.chat_input("Qual a sua dúvida sobre os documentos?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Analisando documentos..."):
            try:
                qa_chain = carregar_cadeia_qa()
                resultado = qa_chain.invoke({"query": prompt})
                resposta = resultado["result"]
                st.markdown(resposta)
                with st.expander("Ver fontes da resposta"):
                    st.write("A resposta foi gerada com base nos seguintes trechos:")
                    for doc in resultado["source_documents"]:
                        source_filename = os.path.basename(doc.metadata.get('source', 'N/A'))
                        st.info(f"**Arquivo:** {source_filename} | **Página:** {doc.metadata.get('page', 'N/A')}")
                        st.text(f"...{doc.page_content[:250]}...")
                    st.session_state.messages.append({"role": "assistant", "content": resposta})
            except Exception as e:
                st.error(f"Ocorreu um erro: {e}. Verifique se os documentos já foram processados.")
EOF
echo "app.py (versão corrigida) criado com sucesso."

# 5. Finalização
echo ""
echo "✅ Configuração v3 concluída com sucesso!"
echo ""
echo "--- PRÓXIMOS PASSOS ---"
echo "1. Coloque seus arquivos PDF na pasta: ./${PROJECT_DIR}/documentos/"
echo ""
echo "2. Navegue até o diretório do projeto:"
echo "   cd ${PROJECT_DIR}"
echo ""
echo "3. Inicie a aplicação com o Docker (pode demorar na primeira vez):"
echo "   docker-compose up --build -d"
echo ""
echo "4. Baixe o modelo de linguagem para o Ollama:"
echo "   docker exec -it ollama_service ollama pull llama3:8b"
echo ""
echo "5. Libere a porta no firewall (se necessário):"
echo "   sudo ufw allow 8501/tcp"
echo ""
echo "6. Acesse a aplicação no seu navegador:"
echo "   http://<SEU_IP_DO_SERVIDOR>:8501"
echo "-------------------------"
