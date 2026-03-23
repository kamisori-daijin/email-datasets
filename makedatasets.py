import json
import os 
import random
import re  
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
    "Client", "AngryUser", "Recruiter", "Team", "TechDir",
    "Partner", "Board", "Investor", "Vendor", "Applicant",
    "Peer", "PartnerCEO", "Authority", "Shareholder", "Competitor",
    "Police"
]

tones = [
    "Formal", "Casual", "Direct", "Visionary", "Humble", "Firm", "Urgent",
    "Technical", "Empathetic", "Professional", "Passive-Aggressive", "Blunt",
    "Enthusiastic", "Academic", "Sarcastic"
]

scenarios = [
    "ExtendingDeadline", "AIPitch", "Interview", "Negotiation", "OutageApology", 
    "SecurityLeak", "Pivot", "BudgetIncrease", "FeatureRefusal", "BetaInvite", 
    "Performance", "PrototypeFeedback", "Referral", "PriceChange", "BillingDispute", 
    "Webinar", "Testimonial", "ShippingDelay", "Termination", "NewHire", "DeepfakeScandal",
    "ToxicIncident", "WorkWeekTrial", "FixingHallucination", "MentalHealthDay"
]

contexts = [
    "FridayNight", "IndustryCrisis", "AfterFunding", "ViralTweet", 
    "Pre-Holiday", "Rebranding", "SystemDown", "AfterAward", "Hackathon", 
    "MarketOpening", "LowSignal", "DataLeak"
]


def get_prompt():
    r, t, tone, s, c = random.choice(roles), random.choice(targets), random.choice(tones), random.choice(scenarios), random.choice(contexts)
    instruction = f"Write {tone} email from {r} to {t} about {s} ({c}). Max 120 words."
    
   
    full_prompt = (
        f"Instruction: {instruction}\n\n"
        "Constraint: You MUST output in the following format exactly. \n"
        "The <think> section must be a concise bulleted list (max 3 items, 5 words each).\n\n"
        "Format Example:\n"
        "<think>\n"
        "- Goal: [Objective]\n"
        "- Reason: [Brief context]\n"
        "- Tone: [Specific style]\n"
        "</think>\n"
        "<generate>\n"
        "Subject: [Subject]\n\n"
        "[Body]\n"
        "</generate>"
    )
    return instruction, full_prompt

# setup
OUTPUT_FILE = "trm_cot_dataset_100k_v2.jsonl"
num_samples = 100000

# restart logic
start_id = 0
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:  
            try:
                last_line = json.loads(line)
                start_id = last_line["id"] + 1
            except: pass
print(f"Resuming from ID: {start_id}")

temperature = 0.8
sampler = make_sampler(temp=temperature)

# generation loop
with open(OUTPUT_FILE, "a", encoding="utf-8", buffering=1) as f:
    for i in tqdm(range(start_id, num_samples)):
        instruction, prompt = get_prompt()
        
        response = generate(
            model, 
            tokenizer, 
            prompt=prompt, 
            max_tokens=300, 
            sampler=sampler,
            verbose=False
        )
        
        # --- cleanup  ---
        # 1. delete <expand> Tag
        cleaned_res = re.sub(r'<expand>.*?</expand>', '', response, flags=re.DOTALL)
        
       
        parts = re.findall(r'(<(think|generate)>.*?</\2>)', cleaned_res, flags=re.DOTALL)
        
        
        raw_res = "\n".join([p[0] for p in parts]).strip()
   
        
 
        if "<think>" in raw_res and "<generate>" in raw_res:
            full_text = f"<user>\n{instruction}\n{raw_res}"
            
            if not full_text.endswith("</s>"):
                full_text += "</s>"
            
            data = {
                "id": i,
                "instruction": instruction,
                "text": full_text  
            }
            
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            
            if i % 10 == 0:
                f.flush()
                os.fsync(f.fileno())
        else:
            tqdm.write(f"ID {i} failed: Tag mismatch or cleaning error.")
            
            continue