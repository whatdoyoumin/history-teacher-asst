import pickle

# Load the textbooks.pkl file
file_path = "./data/textbooks.pkl"  # Make sure this file is in the same directory

with open(file_path, "rb") as f:
    textbooks_data = pickle.load(f)

# Check the structure of the data
print(f"Type of object loaded: {type(textbooks_data)}")

# If it's a list, print details about the first few items
if isinstance(textbooks_data, list):
    print(f"Number of documents: {len(textbooks_data)}")
    print(f"Type of first document: {type(textbooks_data[0])}")

    # If they are LangChain Document objects, print metadata and sample content
    if hasattr(textbooks_data[0], "page_content") and hasattr(textbooks_data[0], "metadata"):
        for i, doc in enumerate(textbooks_data[:3]):  # Inspect the first 3 documents
            print(f"\n--- Document {i+1} ---")
            print(f"Metadata: {doc.metadata}")
            print(f"Sample Content: {doc.page_content[:500]}...")  # Show first 500 chars

# If it's a dictionary, print its keys
elif isinstance(textbooks_data, dict):
    print(f"Dictionary Keys: {list(textbooks_data.keys())}")

    # Inspect the first few entries
    for key, value in list(textbooks_data.items())[:3]:
        print(f"\n--- Key: {key} ---")
        print(f"Value Type: {type(value)}")
        print(f"Sample Content: {value[:500] if isinstance(value, str) else value}")

else:
    print("Unexpected format. Please provide the output here for further analysis.")
