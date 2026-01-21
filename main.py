

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from openai import OpenAI
from dotenv import load_dotenv

# 1. SETUP
load_dotenv()
app = FastAPI()

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to Services
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"status": "Juris AI is running"}

@app.post("/ask")
def ask_juris(request: QueryRequest):
    user_query = request.query
    print(f"🧠 Processing: {user_query}")

    # STEP 1: EMBED THE USER'S QUESTION
    # We convert "What are student rights?" into numbers
    query_vector = openai.embeddings.create(
        input=[user_query],
        model="text-embedding-3-small"
    ).data[0].embedding

    # STEP 2: SEARCH THE DATABASE (Semantic Search)
    # We lowered the threshold to 0.1 to allow broader natural language matches
    response = supabase.rpc("match_cases", {
        "query_embedding": query_vector,
        "match_threshold": 0.1, 
        "match_count": 5
    }).execute()
    
    matches = response.data

    # STEP 3: PREPARE CONTEXT FOR AI
    # If no cases found, just tell the AI to rely on general knowledge (or say sorry)
    context_text = ""
    if matches:
        for case in matches:
            context_text += f"CASE: {case['title']} ({case['year']})\nSUMMARY: {case['summary']}\nHOLDING: {case['holding']}\n---\n"
    else:
        context_text = "No specific case law found in the provided database."

    # STEP 4: GENERATE ANSWER (The "Chat" Part)
    # We give the AI the user's question AND the cases we found.
    system_prompt = """You are a legal research assistant. 
    1. Answer the user's question using ONLY the provided case law context.
    2. If the context contains relevant cases, cite them by name.
    3. If the context is empty or irrelevant, politely say you don't have that information in your database.
    4. Keep the answer concise and professional."""

    completion = openai.chat.completions.create(
        model="gpt-4o-mini", # Fast and smart
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_text}\n\nUser Question: {user_query}"}
        ]
    )
    
    ai_answer = completion.choices[0].message.content

    # Return both the AI's written answer AND the raw case data for cards
    return {
        "answer": ai_answer,
        "sources": matches
    }