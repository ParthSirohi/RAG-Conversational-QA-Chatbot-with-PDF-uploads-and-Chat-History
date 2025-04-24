## RAG Conversational QA Chatbot with Chat history
import streamlit as st
from langchain.chains import create_history_aware_retriever, RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain_core.runnables import RunnableMap, RunnableLambda
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import FAISS 
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

# set up steamlit 
st.title("RAG Conversational QA Chatbot with PDF uploads and Chat History")
st.write("Upload PDFs and chat with their content")

#input the Groq API key
api_key = st.text_input("Enter your Groq API Key", type="password")

#check if groq api key is provided 
if api_key:
    llm=ChatGroq(groq_api_key=api_key, model="Gemma2-9b-It")
    # chat interface
    session_id = st.text_input("Session ID", value="default_session")
    #statefully manage chat history
    
    if 'store' not in st.session_state:
        st.session_state.store = {}
    
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    
    #Process the uploaded files
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temppdf =f"./temp.pdf"
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name
            
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)
    
    # Split and create embeddings for the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(splits, embeddings)
        base_retriever = vectorstore.as_retriever()
    
        contexualie_q__system_prompt = (
            "Given a chat history and the latest user question," 
            "which might reference context in the chat history," 
            "formulate a standalone question which can be understood," \
            "without cha history. DO NOT Answer the question," \
            "just reformulate if needed and otherwise return it as is"
        ) 

        contexualise_q_prompt= ChatPromptTemplate.from_messages([
            ("system", contexualie_q__system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])


        history_aware_retreiver = create_history_aware_retriever(
        llm,base_retriever,
        contexualise_q_prompt
        ) 
        
        ## Answer question Prompt
        system_prompt = (
            "You are an assistant for question-answering tasks." 
            "Use the following pieces of retreived context to answer the question." 
            "If you don't know the answer, just say that you don't know." 
            "Use three sentences or less to answer the question." 
            "\n\n"
            "{context}"    
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
        # rag_chain = ConversationalRetrievalChain(
        #     retreiver=history_aware_retreiver,
        #     combine_docs_chain=question_answer_chain, 
        #     return_source_documents=True
        # ) 
        rag_chain=RunnableMap({
            "context": history_aware_retreiver,
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"]
        }) | question_answer_chain


        def get_session_history(session:str)-> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input", 
            history_messages_key="chat_history", 
            output_messages_key="answer"
        )

        user_input = st.text_input("Enter your question here")
        if user_input:
            session_history = get_session_history(session_id)
            config={"configurable":{"session_id":session_id},
                    "chat_history": session_history.messages}
            
            response = conversational_rag_chain.invoke(
                {"input": user_input}, #pass the chat history to the chain
                config=config #contructs a key "abc123" in store 
            )
            st.write(st.session_state.store)
            st.write("Assistant: ",response)
            st.write("Chat History",session_history.messages)

else:
    st.warning("Please enter your Groq API key to use the chatbot.")


