import json
import os
import requests
from typing import List, Dict
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# 读取API密钥
def get_api_key():
    """获取DeepSeek API密钥"""
    # 首先尝试从环境变量获取
    api_key = os.environ.get('DEEPSEEK_API_KEY')
    if api_key:
        return api_key

    # 如果环境变量没有，尝试从.env文件获取
    env_file = str(PROJECT_ROOT / 'embedding-level/.env')
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                if line.strip().startswith('DEEPSEEK_API_KEY='):
                    return line.strip().split('=', 1)[1].strip()
    
    raise ValueError("Please set DEEPSEEK_API_KEY environment variable or configure it in embedding-level/.env file")

def extract_keywords_from_query(query: str, api_key: str) -> List[str]:
    """Use DeepSeek API to extract 2-3 key English keywords from the query"""
    url = "https://api.deepseek.com/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 使用英文提示词
    prompt = f"""Extract 2-3 most important English keywords from the following query. Return only the keywords separated by commas, without any other text.

Query: {query}

Keywords:"""
    
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 50
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        keywords_text = result['choices'][0]['message']['content'].strip()
        
        # Clean and split keywords
        keywords = [kw.strip() for kw in keywords_text.split(',') if kw.strip()]
        
        # Ensure 2-3 keywords
        if len(keywords) > 3:
            keywords = keywords[:3]
        elif len(keywords) == 0:
            # Fallback: use main words from query
            words = query.replace('?', '').replace('.', '').split()
            keywords = words[:2] if len(words) >= 2 else words + ['general']
            
        return keywords
        
    except Exception as e:
        print(f"API call failed: {e}")
        # Return fallback keywords
        words = query.replace('?', '').replace('.', '').split()
        return words[:2] if len(words) >= 2 else words + ['general']

def process_batch(batch_queries: List[str], api_key: str, start_index: int) -> List[Dict]:
    """Process a batch of queries"""
    results = []
    for i, query in enumerate(batch_queries):
        print(f"Processing query {start_index + i + 1}/50...")
        keywords = extract_keywords_from_query(query, api_key)
        result = {
            "query": query,
            "keywords": keywords
        }
        results.append(result)
        time.sleep(0.1)  # Small delay to avoid rate limiting
    return results

def main():
    # Read safe prompts file
    safe_prompts_file = str(PROJECT_ROOT / "data/01_safe.json")
    output_file = str(PROJECT_ROOT / "data/adv.jsonl")
    
    with open(safe_prompts_file, 'r', encoding='utf-8') as f:
        safe_prompts = json.load(f)
    
    # Extract first 50 prompts
    selected_prompts = safe_prompts[:50]
    
    # Get API key
    api_key = get_api_key()
    
    # Process in parallel batches of 10
    batch_size = 10
    all_results = []
    
    with ThreadPoolExecutor(max_workers=5) as executor:  # Use 5 threads for parallel processing
        futures = []
        for i in range(0, len(selected_prompts), batch_size):
            batch = selected_prompts[i:i + batch_size]
            future = executor.submit(process_batch, batch, api_key, i)
            futures.append(future)
        
        # Collect results
        for future in as_completed(futures):
            batch_results = future.result()
            all_results.extend(batch_results)
    
    # Sort results by original order (since parallel processing may change order)
    all_results.sort(key=lambda x: selected_prompts.index(x["query"]))
    
    # Save results to JSONL file
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"Completed! Results saved to {output_file}")

if __name__ == "__main__":
    main()