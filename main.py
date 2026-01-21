

# 1. Add this import at the top
from fastapi.middleware.cors import CORSMiddleware
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client
from openai import OpenAI
from dotenv import load_dotenv

# 1. Load Secrets
load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 2. Initialize the App
app = FastAPI()

# 2. Add this block right here
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for development only)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],
)

# --- Data Models (What the frontend sends us) ---
class SearchRequest(BaseModel):
    query: str

class ChatRequest(BaseModel):
    case_id: int
    question: str

# --- Helper Function: Get Embedding ---
def get_embedding(text):
    text = text.replace("\n", " ")
    return openai.embeddings.create(input=[text], model="text-embedding-3-small").data[0].embedding

# --- ENDPOINT 1: SEARCH ---
@app.post("/search")
async def search_cases(request: SearchRequest):
    # 1. Convert user query to numbers (vector)
    query_vector = get_embedding(request.query)
    
    # 2. Ask Supabase to find similar cases
    # We call the 'match_cases' function we created in SQL earlier
    response = supabase.rpc("match_cases", {
        "query_embedding": query_vector,
        "match_threshold": 0.5, # Adjust this to be stricter/looser
        "match_count": 5
    }).execute()
    
    return response.data

# --- ENDPOINT 2: CHAT WITH CASE ---
@app.post("/chat")
async def chat_with_case(request: ChatRequest):
    # 1. Fetch the case data from Supabase
    response = supabase.table("cases").select("*").eq("id", request.case_id).execute()
    
    if not response.data:
        raise HTTPException(status_code=404, detail="Case not found")
        
    case_data = response.data[0]
    
    # 2. Construct the prompt for GPT-4
    system_prompt = f"""
    You are a legal assistant. You are analyzing the case '{case_data['title']}'.
    
    Here is the Summary of the case:
    {case_data['summary']}
    
    Here is the Holding (Decision):
    {case_data['holding']}
    
    Answer the user's question based ONLY on this information. 
    If the answer is not in the text, say you don't know.
    """
    
    # 3. Call OpenAI
    completion = openai.chat.completions.create(
        model="gpt-4o",  # or gpt-3.5-turbo
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.question}
        ]
    )
    
    return {"answer": completion.choices[0].message.content}

# --- ROOT CHECK ---
@app.get("/")
def read_root():
    return {"status": "Juris API is running"}