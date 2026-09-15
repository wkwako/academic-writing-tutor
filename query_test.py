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

results = store.similarity_search("what makes a thesis statement arguable", k=3)
for doc in results:
    print(doc.metadata)
    print(doc.page_content[:200])
    print("---")

# store.similarity_search("how should I open", k=3, filter={"category": "grad_statement"})

# query = "what makes a thesis statement arguable"
# for doc, score in store.similarity_search_with_score(query, k=3):
#     print(round(score, 3), doc.metadata["source"])
#     print(doc.page_content[:200])
#     print("---")