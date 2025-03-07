import pickle

# Load the processed PDF documents
with open('data/textbooks.pkl', 'rb') as f:
    pdf_documents = pickle.load(f)

# Print metadata of the first document to check
print(pdf_documents[0].metadata)
