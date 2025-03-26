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
from util.llm_util import evaluate_sources, ANSWER_QUESTION_PROMPT, EVALUATE_SOURCES_PROMPT
from util.utility import check_password, get_custom_css_modifier
from tavily import TavilyClient
from langchain.schema import Document

# Load environment variables
load_dotenv()
st.set_page_config(layout="wide")

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


def filter_docs_by_source(docs, include_infopedia, include_textbooks, include_roots):
    """Filter retrieved documents based on selected sources, allowing partial matches and better debugging."""
    
    filtered_docs = []
    unmatched_sources = set()  # Track sources that don't match any filter

    for doc in docs:
        source = doc.metadata.get("source", "").lower().strip()  # Normalize case & remove whitespace

        if not source:
            unmatched_sources.add("(Missing Source)")
            continue  # Skip documents with no source metadata

        # Apply filter conditions with broader matching
        if (include_infopedia and "infopedia" in source) or \
           (include_textbooks and ("sec1" in source or "sec2" in source or "textbook" in source)) or \
           (include_roots and ("roots website" in source)):  # Fix: Match "Roots Website"
            filtered_docs.append(doc)
        else:
            unmatched_sources.add(source)  # Track sources that didn’t match

    # Debugging: Show unmatched sources
    if not filtered_docs:
        st.warning(f"Warning: No sources matched your filters. Unmatched sources found: {', '.join(unmatched_sources)}")

    return filtered_docs


# def answer_question_from_vector_store(vector_store, input_question, include_infopedia, include_textbooks, include_roots):
#     retriever = vector_store.as_retriever(search_kwargs={"k": 10})
#     retrieved_docs = retriever.invoke(input_question)
#     filtered_docs = filter_docs_by_source(retrieved_docs, include_infopedia, include_textbooks, include_roots)

#     if not filtered_docs:
#         return {"answer": "No relevant sources found based on your filters.", "context": []}

#     #formatted_context = "\n\n".join(doc.page_content[:500] for doc in filtered_docs)
#     formatted_context = "\n\n".join(
#     f"{' | '.join(f'{key}: {value}' for key, value in doc.metadata.items())}\n{doc.page_content[:500]}"
#     for doc in filtered_docs)


#     rag_chain = (
#         RunnableLambda(lambda x: {"context": x["context"], "question": x["question"]})
#         | ANSWER_QUESTION_PROMPT
#         | ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
#     )

#     result = rag_chain.invoke({"context": formatted_context, "question": input_question})
    
#     if not validate_answer_format(result.content):
#         return {"answer": "Validation failed. Please try again.", "context": []}
    
#     return {"answer": result.content, "context": filtered_docs}




# Initialize Tavily Client
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def hybrid_search(vector_store, query, include_infopedia, include_textbooks, include_roots, faiss_top_k=10, tavily_top_k=3):
    """
    Combines vector retrieval from FAISS with real-time search from Tavily.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": faiss_top_k})
    faiss_results = retriever.invoke(query)

    # Filter FAISS results based on user-selected sources
    filtered_faiss_results = filter_docs_by_source(faiss_results, include_infopedia, include_textbooks, include_roots)

    # Tavily Retrieval
    tavily_response = tavily_client.search(query, search_depth="advanced")
    tavily_results = tavily_response.get("results", [])[:tavily_top_k]

    # Convert Tavily results into Document format
    tavily_docs = [
        Document(page_content=entry["content"], metadata={"source": entry["url"], "score": entry["score"]})
        for entry in tavily_results
    ]

    # Combine and Sort Results
    all_results = filtered_faiss_results + tavily_docs
    sorted_results = sorted(all_results, key=lambda x: x.metadata.get("score", 1), reverse=True)

    return sorted_results  # Return sorted documents


def answer_question_hybrid_search(vector_store, input_question, include_infopedia, include_textbooks, include_roots):
    retrieved_docs = hybrid_search(vector_store, input_question, include_infopedia, include_textbooks, include_roots)

    if not retrieved_docs:
        return {"answer": "No relevant sources found based on your filters.", "context": []}

    formatted_context = "\n\n".join(
        f"{' | '.join(f'{key}: {value}' for key, value in doc.metadata.items())}\n{doc.page_content[:500]}"
        for doc in retrieved_docs
    )

    rag_chain = (
        RunnableLambda(lambda x: {"context": x["context"], "question": x["question"]})
        | ANSWER_QUESTION_PROMPT
        | ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
    )

    result = rag_chain.invoke({"context": formatted_context, "question": input_question})

    if not validate_answer_format(result.content):
        return {"answer": "Validation failed. Please try again.", "context": []}

    return {"answer": result.content, "context": retrieved_docs}


def render_ui():
    st.title("Heritage Education Sources") 
    st.write("Ask a question or topic related to Singapore's history and culture and we will retrieve the relevant sources. Do not ask questions outside of this topic!")
    st.write("Filter the sources to narrow down the sources you want to retrieve from.")
    
    st.markdown(get_custom_css_modifier(), unsafe_allow_html=True)
    
# Move filters to the main page
    st.header("Filter Sources")
    col1, col2, col3 = st.columns(3)  # Arrange filters in three columns for better layout

    with col1:
        include_infopedia = st.checkbox("Infopedia", value=True)
    with col2:
        include_textbooks = st.checkbox("Textbooks", value=True)
    with col3:
        include_roots = st.checkbox("Roots Articles", value=True)
    
    user_input = st.text_input("Enter your question:", key="query_input")
    
    # if st.button("Get Answer"):
    #     if user_input:
    #         st.session_state.response = answer_question_from_vector_store(
    #             st.session_state.vectorstore, user_input, include_infopedia, include_textbooks, include_roots
    #         )
    #     else:
    #         st.warning("Please enter a question.")


    # Update UI to Use Hybrid Search
    if st.button("Get Answer"):
        if user_input:
            st.session_state.response = answer_question_hybrid_search(
                st.session_state.vectorstore, user_input, include_infopedia, include_textbooks, include_roots
            )
        else:
            st.warning("Please enter a question.")
    
    if 'response' in st.session_state and st.session_state.response:
        st.subheader("Answer:")
        st.write(st.session_state.response['answer'])

        with st.expander("Referenced Sources"):
            if st.session_state.response['context']:
                sources_data = []
                
                for doc in st.session_state.response['context']:
                    source_info = {
                        "Title": doc.metadata.get("title", "Unknown"),
                        "Source": doc.metadata.get("source", "Unknown"),
                        "Page": doc.metadata.get("page", "N/A"),  # Include Page Number
                        "Page Content": doc.page_content[:300] + "...",
                        "URL": doc.metadata.get("url","N/A")
                    }

                    sources_data.append(source_info)

                df_sources = pd.DataFrame(sources_data)

                # Drop the "URL" column since it’s empty
                st.table(df_sources)
                
                st.subheader("Source Evaluation")
                evaluation = evaluate_sources(df_sources.reset_index(drop=True))
                st.write(evaluation)

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = load_vectorstore()

render_ui()
