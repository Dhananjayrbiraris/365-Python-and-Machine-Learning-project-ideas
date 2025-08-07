import os
import uuid
import chromadb
import pandas as pd
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions
from PyPDF2 import PdfReader
from groq import Groq
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
import docx
import openpyxl
from typing import Union, Dict, List, Any, Optional
import tempfile
import re

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize clients and models
groq_client = Groq(api_key=GROQ_API_KEY)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ChromaDB setup
chroma_client = chromadb.PersistentClient(path="db")
collection = chroma_client.get_or_create_collection(
    name="documents",
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
)

# In-memory storage
chat_history: Dict[str, List[tuple]] = {}
csv_tables: Dict[str, pd.DataFrame] = {}
bm25_corpus: List[str] = []
bm25_index: Optional[BM25Okapi] = None

def clear_vector_db():
    """Clear all vector database data while preserving the collection structure"""
    global bm25_corpus, csv_tables, collection
    
    try:
        # Clear ChromaDB collection data
        collection.delete()
        
        # Clear in-memory data stores
        bm25_corpus = []
        csv_tables = {}
        
        # Reinitialize the collection
        collection = chroma_client.get_or_create_collection(
            name="documents",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        )
        return True
    except Exception as e:
        print(f"Error clearing vector DB: {str(e)}")
        return False

def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    text = re.sub(r'\s+', ' ', text).strip()
    return ''.join(char for char in text if char.isprintable())

def read_document(file_path: str) -> str:
    """Read various file types and return cleaned text content"""
    text = ""
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext == ".pdf":
            reader = PdfReader(file_path)
            text_pages = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_pages.append(clean_text(page_text))
            text = "\n".join(text_pages)
        
        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = clean_text(f.read())
        
        elif ext == ".csv":
            df = pd.read_csv(file_path)
            csv_tables[os.path.basename(file_path)] = df
            text = df.to_string(index=False)
        
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(file_path)
            csv_tables[os.path.basename(file_path)] = df
            text = df.to_string(index=False)
        
        elif ext == ".docx":
            doc = docx.Document(file_path)
            text = clean_text("\n".join(para.text for para in doc.paragraphs))
        
        elif ext == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            text = clean_text(json.dumps(data, indent=2))
        
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
    
    except Exception as e:
        raise ValueError(f"Error reading {file_path}: {str(e)}")
    
    return text

def process_data(data: Union[str, Dict, List, pd.DataFrame], file_path: Optional[str] = None) -> str:
    """Process direct data input and return text content"""
    try:
        if isinstance(data, pd.DataFrame):
            csv_tables["direct_input.csv"] = data
            return data.to_string(index=False)
        
        elif isinstance(data, (dict, list)):
            if file_path and os.path.splitext(file_path)[1].lower() in (".csv", ".xlsx", ".xls"):
                df = pd.DataFrame(data)
                csv_tables[os.path.basename(file_path)] = df
                return df.to_string(index=False)
            else:
                return clean_text(json.dumps(data, indent=2))
        
        elif isinstance(data, str):
            return clean_text(data)
        
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    except Exception as e:
        raise ValueError(f"Error processing data: {str(e)}")

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into chunks with optional overlap"""
    words = text.split()
    chunks = []
    
    if len(words) <= chunk_size:
        return [" ".join(words)]
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
    
    return chunks

def add_to_vector_db(file_path: Optional[str] = None, data: Optional[Union[str, Dict, List]] = None) -> None:
    """Add document chunks to ChromaDB and BM25 from either file or direct data"""
    global bm25_corpus, bm25_index
    
    try:
        if file_path:
            text = read_document(file_path)
        elif data is not None:
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tmp:
                processed_text = process_data(data)
                tmp.write(processed_text)
                tmp_path = tmp.name
            text = read_document(tmp_path)
            os.unlink(tmp_path)
        else:
            raise ValueError("Either file_path or data must be provided")
                    
        if not text:
            raise ValueError("No content found to process")
        
        chunks = chunk_text(text)
        if not chunks:
            return
        
        # Add to ChromaDB
        embeddings = embedder.encode(chunks).tolist()
        doc_id = os.path.basename(file_path) if file_path else f"direct_{uuid.uuid4().hex[:8]}"
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        collection.add(documents=chunks, embeddings=embeddings, ids=ids)
        
        # Add to BM25
        bm25_corpus.extend(chunks)
        tokenized_corpus = [doc.split(" ") for doc in bm25_corpus]
        bm25_index = BM25Okapi(tokenized_corpus)
    
    except Exception as e:
        raise ValueError(f"Error adding to vector database: {str(e)}")

def hybrid_search(query: str, top_k: int = 6) -> List[str]:
    """Combine semantic and keyword search"""
    # Semantic search
    query_embedding = embedder.encode([query]).tolist()
    semantic_results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    semantic_docs = semantic_results["documents"][0] if semantic_results["documents"] else []
    
    # BM25 keyword search
    keyword_docs = []
    if bm25_index:
        bm25_scores = bm25_index.get_scores(query.split())
        if len(bm25_scores) > 0:
            bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k]
            keyword_docs = [bm25_corpus[i] for i in bm25_top_indices if i < len(bm25_corpus)]
    
    # Merge & deduplicate while keeping order
    all_docs = []
    seen = set()
    for doc in semantic_docs + keyword_docs:
        if doc not in seen:
            seen.add(doc)
            all_docs.append(doc)
    
    return all_docs[:top_k]

def rerank_with_groq(query: str, docs: List[str]) -> List[str]:
    """Use Groq LLM to re-rank retrieved documents by relevance"""
    if not docs or len(docs) == 1:
        return docs
    
    try:
        ranking_prompt = (
            f"Rank these document chunks by relevance to the question.\n"
            f"Question: {query}\n\n"
            f"Chunks:\n"
            f"{chr(10).join([f'{i+1}. {doc[:200]}...' for i, doc in enumerate(docs)])}\n\n"
            f"Return ONLY a comma-separated list of the top {min(len(docs), 5)} most relevant chunk numbers (1-{len(docs)})."
        )
        
        completion = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": ranking_prompt}],
            temperature=0,
            max_tokens=200
        )
        
        ranked_str = completion.choices[0].message.content
        ranked_indices = []
        
        for match in re.finditer(r'\d+', ranked_str):
            num = int(match.group())
            if 1 <= num <= len(docs):
                ranked_indices.append(num - 1)
        
        if not ranked_indices:
            return docs
        
        reranked = []
        remaining = set(range(len(docs)))
        
        for idx in ranked_indices:
            if idx in remaining:
                reranked.append(docs[idx])
                remaining.remove(idx)
        
        for i in range(len(docs)):
            if i in remaining:
                reranked.append(docs[i])
        
        return reranked[:len(docs)]
    
    except Exception:
        return docs

def search_documents(query: str, top_k: int = 6) -> List[str]:
    """Hybrid search + LLM re-ranking"""
    candidate_docs = hybrid_search(query, top_k=top_k)
    return rerank_with_groq(query, candidate_docs)

def is_table_question(question: str) -> bool:
    """Detect if question is about tabular data"""
    keywords = [
        "table", "row", "column", "cell", "excel", "csv",
        "average", "sum", "total", "max", "min", "count",
        "median", "mean", "statistic", "calculate", "number"
    ]
    question_lower = question.lower()
    return any(word in question_lower for word in keywords)

def answer_csv_question(question: str) -> Optional[str]:
    """Answer questions about tabular data"""
    for filename, df in csv_tables.items():
        try:
            for col in df.columns:
                col_lower = col.lower()
                if col_lower in question.lower():
                    if "average" in question.lower() or "mean" in question.lower():
                        if pd.api.types.is_numeric_dtype(df[col]):
                            return f"The average {col} is {df[col].mean():.2f}"
                    
                    elif "sum" in question.lower() or "total" in question.lower():
                        if pd.api.types.is_numeric_dtype(df[col]):
                            return f"The total {col} is {df[col].sum():.2f}"
                    
                    elif "maximum" in question.lower() or "max" in question.lower():
                        if pd.api.types.is_numeric_dtype(df[col]):
                            return f"The maximum {col} is {df[col].max():.2f}"
                        else:
                            return f"The latest {col} is {df[col].iloc[-1]}"
                    
                    elif "minimum" in question.lower() or "min" in question.lower():
                        if pd.api.types.is_numeric_dtype(df[col]):
                            return f"The minimum {col} is {df[col].min():.2f}"
                        else:
                            return f"The first {col} is {df[col].iloc[0]}"
                    
                    elif "count" in question.lower():
                        return f"There are {df[col].count()} entries for {col}"
            
            if "how many rows" in question.lower():
                return f"There are {len(df)} rows in the table"
            
            elif "how many columns" in question.lower():
                return f"There are {len(df.columns)} columns in the table"
            
            elif "show first" in question.lower() or "show sample" in question.lower():
                return f"First few rows:\n{df.head().to_string(index=False)}"
        
        except Exception:
            continue
    
    return None

def ask_groq(session_id: str, context: str, question: str) -> str:
    """Ask Groq model with context and chat history"""
    if is_table_question(question):
        csv_answer = answer_csv_question(question)
        if csv_answer:
            return csv_answer
    
    history_text = ""
    if session_id in chat_history:
        for q, a in chat_history[session_id][-5:]:
            history_text += f"Q: {q}\nA: {a}\n\n"
    
    prompt = f"""You are an AI assistant that answers questions based on the provided context.
You must follow these rules:
1. Only use information from the CONTEXT below
2. if conversation starts with Hi, Hello, or Hey, respond then only with "Hello! How can I assist you today?"
3. if conversation starts with What, Where, When, Who, or Why, respond with a detailed answer
3. If the answer isn't in the CONTEXT, say "I don't know"
4. For table/data questions, use precise numbers when available
5. Be concise but helpful

CONTEXT:
{context}

CHAT HISTORY:
{history_text}

QUESTION:
{question}

ANSWER:"""
    
    try:
        completion = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024
        )
        
        answer = completion.choices[0].message.content
        
        if session_id not in chat_history:
            chat_history[session_id] = []
        chat_history[session_id].append((question, answer))
        
        return answer
    
    except Exception as e:
        return f"Error generating answer: {str(e)}"