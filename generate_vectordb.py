# include infopedia
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
import os
import pickle
import pandas as pd
from langchain.schema.runnable import RunnableLambda
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file

load_dotenv()


# Function to sanitize metadata (ensures valid data types for FAISS)
def sanitize_metadata(metadata):
    return {k: (v if isinstance(v, (str, int, float, bool)) else "Unknown") for k, v in metadata.items()}

# Step 1: Load Articles from CSV
csv_file = "data/roots_sg_articles_cleaned.csv"
df = pd.read_csv(csv_file)

# Fill NaN values and ensure text is string type
df['text'] = df['text'].fillna('missing content').astype(str)

# Convert CSV data into LangChain Document objects
articles = []
for _, row in df.iterrows():
    articles.append(Document(
        page_content=row['text'],
        metadata=sanitize_metadata({
            'title': row['title'],
            'source': row['source'],
            'url': row['url']
        })
    ))

# Step 2: Load Processed PDF Documents from Pickle File
with open('data/textbooks.pkl', 'rb') as f:
    pdf_documents = pickle.load(f)

# Ensure PDF documents have sanitized metadata
pdf_documents = [
    Document(page_content=doc.page_content, metadata=sanitize_metadata(doc.metadata))
    for doc in pdf_documents
]

# Step 3: Load Infopedia Articles from Pickle File
with open("data/infopedia.pickle", "rb") as f:
    infopedia_data = pickle.load(f)

# Convert Infopedia data into LangChain Document objects
infopedia_articles = []
for title, details in infopedia_data.items():
    infopedia_articles.append(Document(
        page_content=details["content"],
        metadata=sanitize_metadata({
            'title': title,
            'source': details.get('source', 'Unknown'),
            'url': details.get('url', 'No URL'),
            'last_update_date': details.get('last_update_date', 'Unknown')
        })
    ))

# Combine all document sources (CSV, PDF, Infopedia)
all_documents = articles + pdf_documents + infopedia_articles

# Step 4: Chunk the Documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=100
)
chunks = []

for doc in all_documents:
    splits = text_splitter.split_text(doc.page_content)
    for split in splits:
        chunks.append({
            "text": split,
            "metadata": doc.metadata
        })

# Step 5: Generate Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY"))

# Step 6: Create FAISS Vector Store
faiss_index = FAISS.from_texts(
    [chunk["text"] for chunk in chunks], 
    embeddings, 
    metadatas=[chunk["metadata"] for chunk in chunks]
)

# Step 7: Save the FAISS Vector Store
faiss_index.save_local("faiss_index_infopedia")

print(f"✅ Vector database created successfully! Total documents stored: {len(chunks)}")