import json
import os 
import random
from tqdm import tqdm
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# 1. Loading the model
model, tokenizer = load("mlx-community/gemma-3-4b-it-4bit-DWQ")


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

# 3. Setting up the generation loop
OUTPUT_FILE = "email_dataset_100k_v2.jsonl"
num_samples = 100000


# --- Restart logic started ---
start_id = 0
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:  
            try:
                last_line = json.loads(line)
                start_id = last_line["id"] + 1
            except Exception:
                pass
print(f"Resuming from ID: {start_id}")
# --- End of restart logic ---

temperature = 0.8
sampler = make_sampler(temp=temperature)

# Adding in "a" mode
with open(OUTPUT_FILE, "a", encoding="utf-8", buffering=1) as f:
    for i in tqdm(range(start_id, num_samples)):
        # 1. Obtaining the prompt
        instruction, prompt = get_prompt()
        
        # 2. Generation 
        response = generate(
            model, 
            tokenizer, 
            prompt=prompt, 
            max_tokens=500, 
            sampler=sampler,
            verbose=False
        )
        
        try:
            # 1. Find the first { and last } in the response.
            raw_res = response.strip()
                    
           
            start_idx = raw_res.find("{")
            end_idx = raw_res.rfind("}")
                    
            if start_idx != -1 and end_idx != -1:
               
                clean_output_str = "{" + raw_res[start_idx+1 : end_idx] + "}"
                email_json = json.loads(clean_output_str) 
            else:
                # If parentheses are not found, attempt to concatenate them as a continuation of the prompt.
            
                clean_output_str = "{" + raw_res.split("}")[0] + "}"
                email_json = json.loads(clean_output_str) 
            
           # 4. Data organization
            data = {
                "id": i,
                "instruction": instruction,
                "output": email_json
            }
            
            # 5. Writing
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            
            # Write to disk every 10 items
            if i % 10 == 0:
                f.flush()
            os.fsync(f.fileno())    
        except Exception:
            # If the analysis fails, skip that ID and move on to the next step.
            tqdm.write(f"ID {i} failed | Response: {response[:30]}...")
            print(f"\n--- DEBUG (ID {i}) ---")
            print(f"RAW RESPONSE: |{response}|")
            print("----------------------")
            continue