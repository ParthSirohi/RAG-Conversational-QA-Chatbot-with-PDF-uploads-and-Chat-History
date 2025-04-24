# RAG-Conversational-QA-Chatbot-with-PDF-uploads-and-Chat-History

This application is a Streamlit-based Conversational Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and interact with their content through a conversational interface. It leverages LangChain, HuggingFace Embeddings, FAISS, and Groq's Gemma2-9b-It model to provide context-aware answers, maintaining chat history across sessions.​

Features

    PDF Uploads: Upload multiple PDF files to query their content.

    Conversational Interface: Engage in a chat-like interaction with the uploaded documents.

    Chat History: Maintains conversation history across sessions using session IDs.

    Contextual Understanding: Utilizes chat history to provide context-aware responses.

    RAG Pipeline: Combines document retrieval with generative AI for accurate answers.​
    GitHub

Technologies Used

    Streamlit: For building the interactive web interface.

    LangChain: Framework for developing applications with LLMs.

    HuggingFace Embeddings: For transforming text into vector representations.

    FAISS: Efficient similarity search and clustering of dense vectors.

    Groq's Gemma2-9b-It: Large language model for generating responses.

    PyPDFLoader: To extract text from PDF files.

    dotenv: For managing environment variables.​
 

Installation

    Clone the Repository:

    git clone https://github.com/ParthSirohi/RAG-Conversational-QA-Chatbot-with-PDF-uploads-and-Chat-History.git
    cd RAG-Conversational-QA-Chatbot-with-PDF-uploads-and-Chat-History

    Create a Virtual Environment:

    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate

    Install Dependencies:

    pip install -r requirements.txt

    Set Up Environment Variables:

    Create a .env file in the project root directory and add your API keys:

    GROQ_API_KEY=your_groq_api_key
    HF_TOKEN=your_huggingface_token

Usage

    Run the Application:

    streamlit run app.py

    Interact with the Chatbot:

        Enter your Groq API key when prompted.

        Input a session ID to maintain chat history.

        Upload one or more PDF files.

        Type your questions in the input field to receive answers based on the uploaded documents.​
      

Troubleshooting

    Missing chat_history Variable:

    If you encounter the error:​

KeyError: "Input to ChatPromptTemplate is missing variables {'chat_history'}..."

:contentReference[oaicite:29]{index=29}

:contentReference[oaicite:30]{index=30} :contentReference[oaicite:31]{index=31}&#8203;:contentReference[oaicite:32]{index=32}

- **TypeError: string indices must be integers**:

:contentReference[oaicite:33]{index=33} :contentReference[oaicite:34]{index=34}&#8203;:contentReference[oaicite:35]{index=35}

```python
if isinstance(response, dict):
    st.write("Assistant: ", response.get("answer", "No answer found."))
else:
    st.write("Unexpected response format.")

Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.​
License

This project is licensed under the MIT License.​
Acknowledgements

    LangChain

    HuggingFace

    FAISS

    Streamlit

    Groq​
    GitHub
