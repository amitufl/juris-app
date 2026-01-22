

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class QueryRequest(BaseModel):
    query: str

@app.get("/trending")
def get_trending():
    """Fetches 5 random cases from your actual DB to populate the sidebar"""
    try:
        # We fetch a few cases. Using a randomizer or just the latest ones.
        response = supabase.table("cases").select("title, year, summary").limit(5).execute()
        return response.data
    except Exception as e:
        print(f"Error fetching trending: {e}")
        return []

@app.post("/ask")
def ask_juris(request: QueryRequest):
    user_query = request.query
    
    # 1. Embed Query
    query_vector = openai.embeddings.create(
        input=[user_query],
        model="text-embedding-3-small"
    ).data[0].embedding

    # 2. Search DB (Threshold 0.25 is the sweet spot)
    response = supabase.rpc("match_cases", {
        "query_embedding": query_vector,
        "match_threshold": 0.25, 
        "match_count": 5
    }).execute()
    
    matches = response.data

    # 3. Build Context
    context_text = ""
    if matches:
        for case in matches:
            context_text += f"CASE: {case['title']} ({case['year']})\nSUMMARY: {case['summary']}\nHOLDING: {case['holding']}\n---\n"
    else:
        context_text = "No specific case law found in the provided database."

    # 4. Strict Formatting Prompt
    system_prompt = """
    You are a legal expert. Format your answer exactly like this:
    
    **Case Description**
    [Brief history of what happened]

    **The Two Sides**
    * **Plaintiff:** [Their key argument]
    * **Defendant:** [Their key argument]

    **The Decision**
    [Who won and the legal reasoning]

    **Significance**
    [Why this matters today]

    If the user asks a general question (not about one specific case), summarize the legal principles found in the context cases.
    """

    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_text}\n\nUser Question: {user_query}"}
        ]
    )
    
    return {
        "answer": completion.choices[0].message.content,
        "sources": matches
    }