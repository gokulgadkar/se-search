import os
import faiss
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai


app = FastAPI()

# CORS stuff
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)


load_dotenv()
genai.configure(api_key=os.getenv("API_KEY"))

# 768 for faiss, got value from gemini embedding
dimension = 768
faiss_index = faiss.IndexFlatL2(dimension)

# paths
INDEXED_CSV_PATH = "indexed.csv"
EMBEDDINGS_FILE_PATH = "embeddings.npy"

link_data = {}
embeddings = []
keywords = []
trie = None  # Initialize trie

# generate embedding 
def get_genai_embedding(text):
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_document",
        title="Embedding of single string"
    )
    return np.array(result['embedding'], dtype=np.float32)

# data from csv and make FAISS
def load_data():
    global embeddings, keywords, link_data
    if os.path.exists(INDEXED_CSV_PATH):
        df = pd.read_csv(INDEXED_CSV_PATH)
        for _, row in df.iterrows():
            keyword, link, embedding_str = row['keyword'], row['link'], row['embedding']
            embedding = np.fromstring(embedding_str, sep=" ")
            embeddings.append(embedding)
            keywords.append(keyword)
            link_data.setdefault(keyword, []).append(link)
        faiss_index.add(np.array(embeddings))

# saving newly embedded
def save_new_embeddings():
    global embeddings, keywords, link_data
    np.save(EMBEDDINGS_FILE_PATH, np.array(embeddings))
    indexed_data = [[keyword, link, " ".join(map(str, embeddings[keywords.index(keyword)]))]
                    for keyword, links in link_data.items() for link in links]
    pd.DataFrame(indexed_data, columns=['keyword', 'link', 'embedding']).to_csv(INDEXED_CSV_PATH, index=False)

# 
def process_keyword(keyword, link):
    if keyword in keywords:
        if link not in link_data[keyword]:
            link_data[keyword].append(link)
        return
    embedding = get_genai_embedding(keyword)
    embeddings.append(embedding)
    keywords.append(keyword)
    link_data[keyword] = [link]
    faiss_index.add(np.array([embedding]))
    save_new_embeddings()

# LOading key to link
@app.on_event("startup")
async def load_and_process_data():
    load_data()
    df = pd.read_csv("index.csv")
    for _, row in df.iterrows():
        process_keyword(row['keyword'], row['link'])

# searching using vector store, by embedding current phrase and comparing with embeded phrases
@app.get("/search")
async def search(query: str = Query(...)):
    query_embedding = get_genai_embedding(query).reshape(1, -1)
    distances, indices = faiss_index.search(query_embedding, k=3)
    matched_links = [link_data[keywords[idx]] for idx in indices[0] if idx < len(keywords)]
    unique_links = list(dict.fromkeys([link for sublist in matched_links for link in sublist]))
    return {"query": query, "results": unique_links}

# Trie for autosuggest
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            node = node.children.setdefault(char, TrieNode())
        node.is_end_of_word = True

    def search_prefix(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        suggestions = []
        self._collect_words(node, prefix, suggestions)
        return suggestions[:3]

    def _collect_words(self, node, prefix, suggestions):
        if node.is_end_of_word:
            suggestions.append(prefix)
        for char, child_node in node.children.items():
            self._collect_words(child_node, prefix + char, suggestions)

# make the Trie
def load_keywords_from_csv(file_path):
    global trie
    trie = Trie()
    df = pd.read_csv(file_path)
    for _, row in df.iterrows():
        for word in row['keyword'].split():
            trie.insert(word)

load_keywords_from_csv("index.csv")

# autosuggest 
@app.get("/autosuggest")
async def autosuggest(word: str = Query(...)):
    suggestions = trie.search_prefix(word)
    return {"query": word, "suggestions": suggestions}
