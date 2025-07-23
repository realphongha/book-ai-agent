import os
import yaml
import argparse
import logging
logging.basicConfig(level=logging.INFO)
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from ebooklib import epub
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_milvus import Milvus
from langchain.schema import Document


def ingest(path, cfg):
    # Load document
    if path.endswith('.pdf'):
        docs = PyPDFLoader(path).load()
    elif path.endswith('.txt'):
        docs = TextLoader(path).load()
    elif path.endswith('.epub'):
        book = epub.read_epub(path)
        text = ''.join([item.get_body_content().decode() for item in book.get_items() if item.get_type() == epub.EpubHtml])
        docs = [Document(page_content=text)]
    else:
        raise ValueError("Unsupported format")
    book_title = path.split(os.path.sep)[-1]

    # Split into chunks with metadata
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        chunk.metadata['book_title'] = book_title

    # Embed and store in Milvus
    embs = OllamaEmbeddings(model=cfg['embedding']['model'])
    Milvus.from_documents(
        chunks,
        embedding=embs,
        collection_name="books",
        connection_args={
            "host": cfg['milvus']['host'],
            "port": cfg['milvus']['port'],
        },
        partition_key_field="book_title",
    )
    logging.info(f"Ingested {len(chunks)} chunks into 'books' tagged"
                 f" with book_title = {book_title}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ingest books into Milvus')
    parser.add_argument('path', type=str, help='Path to the book file')
    args = parser.parse_args()
    cfg = yaml.safe_load(open('configs.yaml', 'r'))
    ingest(args.path, cfg)

