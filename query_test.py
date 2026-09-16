import pathlib
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

store = Chroma(
    collection_name="criteria",
    persist_directory="chroma_db",
    embedding_function=embeddings,
)

#print (store._collection.count())

# results = store.similarity_search("how should I open", k=3, filter={"category": "undergrad_essay"})
# for doc in results:
#     print(doc.metadata)
#     print(doc.page_content[:500])
#     print("---")

query = "what makes a thesis statement arguable" #returns scores of ~0.7-0.8
query = "what makes a strong undergraduate essay" #returns scores of ~0.8-1.0
# query = "how do I bake sourdough bread" #returns scores of ~1.7
for doc, score in store.similarity_search_with_score(query, k=3):
    print(round(score, 3), doc.metadata["source"])
    print(doc.page_content[:500])
    print("---")