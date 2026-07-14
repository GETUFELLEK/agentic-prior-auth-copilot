import glob, os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
# from sentence_transformers import CrossEncoder

def _build_chunks():
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = []
    for path in glob.glob("policies/*.pdf"):
        for c in splitter.split_documents(PyPDFLoader(path).load()):
            c.metadata["policy_file"] = os.path.basename(path)
            docs.append(c)
    return docs

emb = OpenAIEmbeddings(model="text-embedding-3-small")
_dense = FAISS.load_local("faiss_index", emb,
                          allow_dangerous_deserialization=True).as_retriever(search_kwargs={"k": 8})
_bm25 = BM25Retriever.from_documents(_build_chunks()); _bm25.k = 8
# _reranker = CrossEncoder("BAAI/bge-reranker-base")   # downloads ~1GB on first run

def hybrid_retrieve(query, top_k=5):
    pool = _dense.invoke(query) + _bm25.invoke(query)
    seen, uniq = set(), []
    for d in pool:
        key = d.page_content[:200]
        if key not in seen:
            seen.add(key); uniq.append(d)
    return uniq[:top_k]