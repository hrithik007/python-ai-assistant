from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import faiss
import numpy as np

app = FastAPI()

client = OpenAI(api_key="open ai key")

# OpenAI embedding vector has 1536 numbers 
dimension = 1536

# FAISS index that uses Euclidean distance (L2 norm) for similarity search.
index = faiss.IndexFlatL2(dimension)

#list to keep track of original text chunks. , will be used later to retrieve the text based on index search results
documents: list[str] = []

class UploadRequest(BaseModel):
    text: str

class AskRequest(BaseModel):
    question: str

@app.post("/upload")
async def upload(req: UploadRequest):
    chunks = [req.text]

    for chunk in chunks:
        embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )
     # Embedding is a list of 1536 float values that represent the meaning of the text.
        vector = embedding.data[0].embedding

        # Convert Python list -> NumPy array -> float32 type.
        # bacause FAISS only works with float32 arrays.
        vector_np = np.array([vector]).astype("float32")

        # FAISS stores this vector in its internal data structure.
        index.add(vector_np)

        #Store original text alongside the vector (so we can retrieve later).
        documents.append(chunk)

    return {"message": "Document uploaded and stored!"}

@app.post("/ask")
async def ask(req: AskRequest):
    q_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=req.question
    )
    q_vector = q_embedding.data[0].embedding
    q_vector_np = np.array([q_vector]).astype("float32")

    k = 3
    distances, labels = index.search(q_vector_np, k)
    # FAISS computes distances between question vector and stored vectors,
    # returns top k closest matches where we put k = 3.

    #Build context from retrieved documents for the LLM , we just pass the related data to LLM
    context = ""
    for idx in labels[0]:
        if idx >= 0:
            context += documents[idx] + "\n"

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Use provided context to answer."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {req.question}"}
            #Sends request to OpenAI API, GPT generates text response.
        ]
    )

    return {"answer": completion.choices[0].message.content}

@app.get("/")
def home():
    return {"message": "AI Knowledge Assistant running! Use /upload and /ask endpoints."}
