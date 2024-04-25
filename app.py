import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import NLTKTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
import nltk

loader = PyPDFLoader("C:\\Users\\RATNADEEP\\Downloads\\2404.07143.pdf")
pages = loader.load_and_split()

st.title(" 🔍RAG on “Leave No Context Behind” Paper")


f = open("C:\\Users\\RATNADEEP\\Desktop\\Gemini_api_key.txt")
KEY = f.read()

chat_template = ChatPromptTemplate.from_messages([
    # System Message Prompt Template
    SystemMessage(content="""You are a Helpful AI Bot. 
    You take the context and question from user. Your answer should be based on the specific context."""),
    # Human Message Prompt Template
    HumanMessagePromptTemplate.from_template("""Answer the question based on the given context.
    Context:
    {context}
    
    Question: 
    {question}
    
    Answer: """)
])

chat_model = ChatGoogleGenerativeAI(google_api_key=KEY, 
                                   model="gemini-1.5-pro-latest")

output_parser = StrOutputParser()

embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=KEY, 
                                               model="models/embedding-001")

# Setting a Connection with the ChromaDB
db_connection = Chroma(persist_directory="C:\\Users\\RATNADEEP\\INTERNSHIP DS\\chroma_db_", embedding_function=embedding_model)
# Converting CHROMA db_connection to Retriever Object
retriever = db_connection.as_retriever(search_kwargs={"k": 5})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | chat_template
    | chat_model
    | output_parser
)

user_input = st.text_area("Write your questions here related to this paper")
stream = True
if st.button("Submit"):
    st.header(user_input)
    response = chain.invoke(user_input)
    st.write(response)
    
