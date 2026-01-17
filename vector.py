from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("./realistic_restaurant_reviews.csv")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for i, row in  df.iterrows():
        doc = Document(
            page_content=row["Title"] + " " + row["Review"],
            metadata={
                "rating": row["Rating"],
                "date": row["Date"],
            },
            id = str(i)
        )
        documents.append(doc)
        ids.append(str(i))

vectorestore = Chroma(
    persist_directory=db_location,
    collection_name="restaurant_reviews",
    embedding_function=embeddings,
)

if add_documents:
    vectorestore.add_documents(documents, ids=ids)

retriever = vectorestore.as_retriever(search_kwargs={"k": 5})
