import pathlib
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def index_corpus():
    #load
    path = pathlib.Path.cwd() / "corpus"

    docs = []
    for folder in path.iterdir():
        if not folder.is_dir():
            continue

        files = folder.glob("*.txt")

        for file in files:
            text = file.read_text(encoding="utf-8")
            docs.append(Document(page_content=text, metadata={"category": folder.name, "source": file.name}))

    #chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    #embed and store
    shutil.rmtree("chroma_db", ignore_errors=True)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embedded = Chroma.from_documents(chunks, embedding=embeddings, collection_name="criteria", persist_directory="chroma_db")

    return embedded._collection.count()
    

#print (index_corpus())