import os
import streamlit as st
import boto3
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda
from langchain.embeddings import OpenAIEmbeddings
from pydantic import BaseModel, ValidationError
import re
import pandas as pd

# Load environment variables
load_dotenv()

def load_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY"))
    s3_bucket_name = os.getenv("s3_bucket_name")
    
    if s3_bucket_name:
        local_path = "nhb_vectorstore"
        if not os.path.exists(f"{local_path}/index.faiss") or not os.path.exists(f"{local_path}/index.pkl"):
            s3 = boto3.resource("s3")
            os.makedirs(local_path, exist_ok=True)
            s3.Bucket(s3_bucket_name).download_file("nhb-history-teacher/index.faiss", f"{local_path}/index.faiss")
            s3.Bucket(s3_bucket_name).download_file("nhb-history-teacher/index.pkl", f"{local_path}/index.pkl")
        return FAISS.load_local(local_path, embeddings, allow_dangerous_deserialization=True)
    else:
        return FAISS.load_local("faiss_index_infopedia", embeddings, allow_dangerous_deserialization=True)

class AnswerFormat(BaseModel):
    perspectives: list[str]
    discussion_questions: list[str]

def validate_answer_format(answer: str) -> bool:
    perspectives = re.findall(r"Perspective \d+: (.*?)\n", answer)
    discussion_questions = re.findall(r"\d+\. (.*?)\n", answer.split("Discussion Questions:")[1] if "Discussion Questions:" in answer else "")
    
    try:
        AnswerFormat(perspectives=perspectives, discussion_questions=discussion_questions)
        return True
    except ValidationError:
        return False

def get_custom_css_modifier():
    css_modifier = """
<style>
/* remove Streamlit default menu and footer */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* remove borders from forms */
div[data-testid="stForm"] {
    border: 0 !important;
    padding: 0px;
}
</style>
"""
    return css_modifier

def answer_question_from_vector_store(vector_store, input_question):
    prompt = PromptTemplate.from_template(
        template="""
You are the Heritage Education Research Assistant, an AI-powered tool designed to help educators in Singapore create comprehensive and balanced lesson plans about Singapore's history and culture. Your task is to provide multiple perspectives on historical questions, with a focus on validated sources from the National Heritage Board (NHB) and other reputable institutions.

Generate 3-5 different perspectives on the question, each with a brief summary (2-3 sentences) explaining the reasoning behind that perspective. For each perspective, include a source citation in one of the following formats:

Page Number (if the source is a book or document with specific page references),
Website Link (if the source is a digital resource or website),
Or both if applicable (e.g., a book citation with a page number and a link to the digital source).

Format the answer as follows:

Perspective #: [Answer summary]
Page: [Page Number], Book Title: Sec1 or Sec2
OR
Website Link: [Link to the source]
OR
Page: [Page Number] | Website Link: [Link to the source]

Ensure that the language and content complexity is appropriate for the specified student age group (if provided).

If a specific historical timeframe or theme is specified, tailor your responses to fit within those parameters.

After presenting the perspectives, suggest 2-3 discussion questions that could encourage critical thinking among students about these different viewpoints.

Remember, your goal is to provide educators with balanced, well-sourced information that they can use to create engaging and thought-provoking lessons about Singapore's history and culture. Each citation should be appropriately linked to the perspective it corresponds to, whether it is a page number, website link, or both.

Context: {context}

Question: {question}
        """
    )
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content[:500] for doc in docs)  # Trim content to speed up processing
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # Reduce number of retrieved documents
    retrieved_docs = retriever.invoke(input_question)
    
    formatted_context = format_docs(retrieved_docs)
    
    rag_chain_from_docs = (
        RunnableLambda(lambda x: {"context": x["context"], "question": x["question"]})
        | prompt
        | ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
    )
    
    result = rag_chain_from_docs.invoke({"context": formatted_context, "question": input_question})
    
    if not validate_answer_format(result.content):
        return {"answer": "Validation failed. Please try again.", "context": []}
    
    return {"answer": result.content, "context": retrieved_docs}

# Load FAISS index once and store in session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = load_vectorstore()

# Streamlit UI
st.title("Heritage Education Research Assistant")
st.write("Ask a question related to Singapore's history and culture.")

# Apply custom CSS
st.markdown(get_custom_css_modifier(), unsafe_allow_html=True)

# Initialize session state
if 'response' not in st.session_state:
    st.session_state.response = None
    
user_input = st.text_input("Enter your question:", key="query_input")

if st.button("Get Answer"):
    if user_input:
        st.session_state.response = answer_question_from_vector_store(st.session_state.vectorstore, user_input)
    else:
        st.warning("Please enter a question.")

# Display answer
if st.session_state.response:
    st.subheader("Answer:")
    st.write(st.session_state.response['answer'])
    
    # Display sources in collapsible format
    with st.expander("Referenced Sources"):
        if st.session_state.response['context']:
            sources_data = []
            for doc in st.session_state.response['context']:
                source_info = {"Title": "", "Source": "", "URL": "", "Page Content": doc.page_content[:300] + "..."}
                for key, value in doc.metadata.items():
                    if key in ['title', 'source', 'url']:
                        source_info[key.capitalize()] = value
                sources_data.append(source_info)
            
            df_sources = pd.DataFrame(sources_data)
            
            # Drop columns that are empty or contain only NaN values
            df_sources.dropna(axis=1, how='all', inplace=True)
            
            st.table(df_sources.set_index(pd.Index(["" for _ in range(len(df_sources))])))
