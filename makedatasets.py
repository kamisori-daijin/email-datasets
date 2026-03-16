import json
import os # 追加
import random
from tqdm import tqdm
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# 1. モデルのロード
model, tokenizer = load("mlx-community/gemma-3-4b-it-4bit-DWQ")

# 2. リスト設定（内容は変更なし）
roles = [
    "CEO", "CTO", "Founder", "Junior Developer", "Senior Engineer", 
    "Product Manager", "Sales Executive", "Customer Success Manager", 
    "HR Manager", "Legal Counsel", "Marketing Lead", "PR Manager", 
    "Security Researcher", "Intern", "Account Manager",
    "Disgruntled Employee", "Head of Ethics", "Whistleblower", 
    "Data Scientist", "System Architect"
]

targets = [
    "New Client", "Dissatisfied User", "Recruiter", "Internal Team", 
    "Technical Director", "Potential Partner", "Board of Directors", 
    "Angel Investor", "Vendor", "Job Applicant", "Industry Peer", 
    "CEO of a Partner Company", "Regulatory Authority", 
    "Angry Shareholder", "Competitor", "Law Enforcement"
]

tones = [
    "extremely formal and cautious", "friendly and casual", "direct and concise", 
    "visionary and inspiring", "apologetic and humble", "firm and authoritative", 
    "urgent and high-pressure", "highly technical and detailed", "warm and empathetic", 
    "strictly professional", "passive-aggressive", "impatient and blunt", 
    "overly enthusiastic", "dry and academic", "sarcastic but professional"
]

scenarios = [
    "requesting a deadline extension", "pitching a new AI integration", 
    "scheduling a follow-up interview", "negotiating a partnership deal", 
    "apologizing for a server outage", "reporting a critical security vulnerability", 
    "announcing a company pivot", "requesting a budget increase", 
    "declining a feature request", "inviting to a beta testing program", 
    "discussing quarterly performance", "giving feedback on a prototype", 
    "asking for a referral", "notifying about a price change", 
    "resolving a billing dispute", "proposing a joint webinar", 
    "requesting a testimonial", "explaining a delay in shipping", 
    "terminating a contract", "welcoming a new high-profile hire",
    "responding to an AI-generated deepfake scandal", 
    "reporting a toxic workplace incident", "proposing a 4-day work week trial", 
    "correcting a hallucination in a previous report", "requesting a mental health day"
]

contexts = [
    "on a Friday evening", "during a major industry crisis", 
    "after a successful funding round", "following a negative viral tweet", 
    "right before a long holiday", "in the middle of a rebranding phase", 
    "while the system is partially down", "after winning a major award",
    "during a 24-hour hackathon", "minutes before the stock market opens", 
    "while traveling in a low-connectivity area", 
    "after a massive data leak was leaked to the press"
]

def get_prompt():
    r, t, tone, s, c = random.choice(roles), random.choice(targets), random.choice(tones), random.choice(scenarios), random.choice(contexts)
    instruction = f"Write a {tone} business email(under 200 words) from a {r} to a {t} regarding {s}, specifically {c}."
    full_prompt = (
        f"Instruction: {instruction}\n"
        "Constraint: Output ONLY a raw JSON object with 'subject' and 'body'. Use [Name] placeholders.\n"
        "JSON: {"
    )
    return instruction, full_prompt

# 3. 生成ループの設定と【再開機能】
OUTPUT_FILE = "email_dataset_100k_v2.jsonl"
num_samples = 100000


# --- 再開ロジック開始 ---
start_id = 0
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:  # readlines()を使わず、1行ずつ回す（超軽量）
            try:
                last_line = json.loads(line)
                start_id = last_line["id"] + 1
            except Exception:
                pass
print(f"Resuming from ID: {start_id}")
# --- 再開ロジック終了 ---

temperature = 0.8
sampler = make_sampler(temp=temperature)

# "a" モードで追記していく
# --- 生成ループ部分 (修正版) ---
with open(OUTPUT_FILE, "a", encoding="utf-8", buffering=1) as f:
    for i in tqdm(range(start_id, num_samples)):
        # 1. プロンプトの取得
        instruction, prompt = get_prompt()
        
        # 2. 生成 (1件ずつ確実に)
        response = generate(
            model, 
            tokenizer, 
            prompt=prompt, 
            max_tokens=500, 
            sampler=sampler,
            verbose=False
        )
        
        try:
                    # 1. レスポンスから最初の { と 最後の } を探す
            raw_res = response.strip()
                    
            # 命令で最初に "{" を出させているので、基本は先頭にあるはずですが
            # 念のため最初と最後のカッコの位置を特定
            start_idx = raw_res.find("{")
            end_idx = raw_res.rfind("}")
                    
            if start_idx != -1 and end_idx != -1:
                # { から } までを抽出
                clean_output_str = "{" + raw_res[start_idx+1 : end_idx] + "}"
                email_json = json.loads(clean_output_str) 
            else:
                # カッコが見つからない場合、プロンプトの続きとして結合を試みる
                # （Instructionで "JSON: {" と言っているので、続きだけ返ってくる場合）
                clean_output_str = "{" + raw_res.split("}")[0] + "}"
                email_json = json.loads(clean_output_str) 
            
            # 4. データの整理
            data = {
                "id": i,
                "instruction": instruction,
                "output": email_json
            }
            
            # 5. 書き込み
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            
            # 10件ごとにディスクに書き出し
            if i % 10 == 0:
                f.flush()
            os.fsync(f.fileno())    
        except Exception:
            # 解析に失敗した場合はそのIDを飛ばして次へ
            tqdm.write(f"ID {i} failed | Response: {response[:30]}...")
            print(f"\n--- DEBUG (ID {i}) ---")
            print(f"RAW RESPONSE: |{response}|")
            print("----------------------")
            continue