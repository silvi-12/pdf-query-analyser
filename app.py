import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from pathlib import Path
from PyPDF2 import PdfReader
from io import BytesIO
from dotenv import load_dotenv
import os

# Import LangChain components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.embeddings.base import Embeddings

# Load environment variables
load_dotenv()

# Set the page configuration as the first Streamlit command
st.set_page_config(page_title="Company’s Annual Report Analysis Platform")

# Custom Embeddings class for Google GenAI
class GoogleGenAIEmbeddings(Embeddings):
    def __init__(self, model: str):
        self.model = model

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            embedding = self._custom_embedding_logic(text)
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text):
        return self._custom_embedding_logic(text)

    def _custom_embedding_logic(self, text):
        return [0.0] * 512  

# Load credentials from the YAML file
def load_credentials():
    with open('./config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)
    return config

# Save updated credentials to the YAML file
def save_credentials(config):
    with open('./config.yaml', 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

# Function to extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_file = BytesIO(pdf.read())
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split the extracted text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Save text chunks as a FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")

# Set up conversational chain for question answering
def get_conversational_chain():
    prompt_template = """
        Answer the question as detailed as possible from the provided context. If the answer is not available, say "answer not available".
        Context:
        {context}?
        Question:
        {question}

        Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Process user question and respond
def user_input(user_question):
    embeddings = GoogleGenAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

# Initialize the authenticator
config = load_credentials()
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# Main function for the Streamlit app
def main():
    # Sidebar login/register
    with st.sidebar:
        st.title("Login/Register")

        # Login form
        name, authentication_status, username = authenticator.login(location='main')

        # Variable to track whether registration is complete
        registration_complete = st.session_state.get("registration_complete", False)

        if not registration_complete and not authentication_status:
            with st.expander("Register a new account"):
                with st.form(key='register_form'):
                    new_username = st.text_input("Enter a new username", key="reg_username")
                    new_password = st.text_input("Enter a new password", type='password', key="reg_password")
                    confirm_password = st.text_input("Confirm your password", type='password', key="reg_confirm_password")
                    email = st.text_input("Enter your email", key="reg_email")
                    name = st.text_input("Enter your full name", key="reg_name")

                    submit_button = st.form_submit_button("Register")

                    if submit_button:
                        if not new_username.strip() or not new_password.strip() or not confirm_password.strip() or not email.strip() or not name.strip():
                            st.warning("Please fill in all the fields!")
                        elif new_username in config['credentials']['usernames']:
                            st.warning("Username already exists. Please choose a different username.")
                        # Check if the email is already registered
                        elif email in [config['credentials']['usernames'][user].get('email') for user in config['credentials']['usernames'] if 'email' in config['credentials']['usernames'][user]]:
                            st.warning("Email is already registered. Please use a different email or log in.")
                        elif new_password != confirm_password:
                            st.warning("Passwords do not match!")
                        else:
                            hashed_password = stauth.Hasher([new_password]).generate()[0]
                            config['credentials']['usernames'][new_username] = {
                                'email': email,
                                'name': name,
                                'password': hashed_password
                            }
                            save_credentials(config)
                            st.success("User registered successfully!")
                            st.session_state["registration_complete"] = True  # Set registration as complete

        # Add a logout button if the user is authenticated
        if authentication_status:
            authenticator.logout("Logout", "sidebar")
            st.sidebar.write(f"Welcome, {name}!")

    # Display main content if authenticated
    if authentication_status:
        st.write(f"Welcome {name}!")
        
        # User input for asking questions about PDFs
        user_question = st.text_input("Ask a question from the PDF files")
        if user_question:
            user_input(user_question)

        # File uploader for PDFs
        pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True, type="pdf")
        if st.button("Submit and Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("PDFs processed and indexed.")
    
    elif authentication_status == False:
        st.error("Incorrect username/password.")
    elif authentication_status == None:
        st.warning("Please log in to access the app.")

if __name__ == "__main__":
    main()
