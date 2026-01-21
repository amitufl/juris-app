import os
import time
import requests
from tqdm import tqdm  # The progress bar library
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI

# 1. SETUP
load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# CONFIGURATION
# 1960 is roughly where modern civil rights cases begin. 
# 1789 is the actual start of the court (takes much longer).
START_YEAR = 1960
END_YEAR = 2023

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json'
}

def get_embedding(text):
    text = text.replace("\n", " ")
    return openai.embeddings.create(input=[text], model="text-embedding-3-small").data[0].embedding

def clean_html(raw_html):
    if not raw_html: return None
    import re
    return re.sub('<.*?>', '', raw_html).strip()

def seed_history():
    print(f"📚 Starting Historical Archive ({START_YEAR}-{END_YEAR})...")
    
    for term in range(START_YEAR, END_YEAR + 1):
        print(f"\n🗓️  Processing Term: {term}")
        
        # 1. Get the list of ALL cases for this year
        list_url = f"https://api.oyez.org/cases?filter=term:{term}&per_page=100"
        try:
            resp = requests.get(list_url, headers=HEADERS)
            if resp.status_code != 200:
                print(f"   ⚠️ Could not fetch term {term}. Skipping.")
                continue
                
            cases_list = resp.json()
            print(f"   Found {len(cases_list)} cases.")
            
            # 2. Iterate through cases with a Progress Bar
            for case_meta in tqdm(cases_list, desc=f"   Downloading {term}"):
                case_url = case_meta['href']
                
                # CHECK: Do we already have this case?
                # We check locally to save time/money on embeddings
                existing = supabase.table("cases").select("id").eq("case_id", case_url).execute()
                if existing.data:
                    continue # Skip if already saved
                
                # FETCH DETAILS
                try:
                    detail_resp = requests.get(case_url, headers=HEADERS)
                    full_data = detail_resp.json()
                    
                    # Handle the list quirk
                    if isinstance(full_data, list): 
                        if not full_data: continue
                        full_data = full_data[0]
                        
                    # EXTRACT
                    title = full_data['name']
                    summary = clean_html(full_data.get('facts_of_the_case', ''))
                    holding = clean_html(full_data.get('conclusion', ''))
                    
                    if summary:
                        # EMBED & SAVE
                        vector = get_embedding(summary)
                        
                        record = {
                            "case_id": case_url,
                            "title": title,
                            "year": int(full_data['term']),
                            "summary": summary,
                            "holding": holding,
                            "embedding": vector
                        }
                        
                        supabase.table("cases").upsert(record, on_conflict="case_id").execute()
                    
                    # Be polite to the API
                    time.sleep(0.5) 
                    
                except Exception as e:
                    # Log error but keep moving
                    # print(f"Error on {case_url}: {e}")
                    pass
                    
        except Exception as e:
            print(f"   ❌ Critical Error on term {term}: {e}")

    print("\n🎉 Archive Complete. Your database is now a powerhouse.")

if __name__ == "__main__":
    seed_history()