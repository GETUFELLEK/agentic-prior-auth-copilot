import os, glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

paths = glob.glob("policies/*.pdf")
print(f"Found {len(paths)} PDF(s): {paths}")
if not paths:
    raise SystemExit("No PDFs in ./policies/")

emb = OpenAIEmbeddings(model="text-embedding-3-small")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

docs = []
for path in paths:
    pages = PyPDFLoader(path).load()
    print(f"  {os.path.basename(path)}: {len(pages)} page(s)")
    chunks = splitter.split_documents(pages)
    for c in chunks:
        c.metadata["policy_file"] = os.path.basename(path)
    docs.extend(chunks)

print(f"Total chunks: {len(docs)}")
if not docs:
    raise SystemExit("0 chunks — PDFs may be scanned images needing OCR.")

store = FAISS.from_documents(docs, emb)
store.save_local("faiss_index")
print(f"Indexed {len(docs)} chunks locally to ./faiss_index/")