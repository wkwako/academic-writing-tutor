import os
import pathlib
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


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

    return docs

    #chunk
    #embed and store

docs = index_corpus()

print (docs[0].metadata)