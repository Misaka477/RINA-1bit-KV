"""
KVR risk scenario tests: 6 diverse prompts where context > window.
Compares KVRGenerator vs native, saves full texts to JSON.
"""
import os, sys, json, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from transformers import AutoModelForCausalLM, AutoTokenizer
from modules import KVRGenerator

MODEL = "D:/Software_Development/Project/models/Llama-3.2-1B"
DEV = None

# ── Scenario prompts (each > 64 tokens, non-repetitive) ──

SCENARIOS = {
    "chain_instruction": {
        "prompt": (
            "I will give you a three-step task. Follow each step carefully.\n"
            "Step 1: Read the following list of cities and their famous landmarks:\n"
            "Paris - Eiffel Tower, London - Big Ben, Tokyo - Shibuya Crossing, "
            "New York - Statue of Liberty, Rome - Colosseum, Sydney - Opera House, "
            "Mumbai - Gateway of India, Cairo - Pyramids of Giza, Rio - Christ the Redeemer, "
            "Berlin - Brandenburg Gate, Moscow - Saint Basil's Cathedral, "
            "Beijing - Forbidden City, Istanbul - Hagia Sophia.\n"
            "Step 2: Identify which city has the Eiffel Tower, which has the Colosseum, "
            "and which has the Statue of Liberty. Write them in order.\n"
            "Step 3: Using only the cities from Step 2, write a short paragraph about "
            "visiting each of them. The paragraph must start with:\n"
            "I would love to visit "
        ),
        "check": ["Paris", "Rome", "New York"],
        "window": 64,
        "desc": "Long chain instruction - retrieval must find all 3 cities"
    },

    "multi_turn_dialogue": {
        "prompt": (
            "The following is a conversation between Alice and Bob about travel plans.\n\n"
            "Alice: Hi Bob! I'm planning a trip to Japan next month. Any recommendations?\n"
            "Bob: Absolutely! Tokyo is amazing. You should visit during cherry blossom season.\n"
            "Alice: That sounds lovely. How long should I stay?\n"
            "Bob: At least 10 days. Spend 3 days in Tokyo, 2 in Kyoto, and the rest exploring.\n"
            "Alice: What about the food? I'm vegetarian.\n"
            "Bob: Japan has great vegetarian options. Look for shojin ryori - it's Buddhist cuisine.\n"
            "Alice: Oh that's perfect! Also, I need to know about transportation.\n"
            "Bob: Get a JR Pass before you go. It covers most trains including the Shinkansen.\n"
            "Alice: The Shinkansen - that's the bullet train right?\n"
            "Bob: Yes! It's incredibly efficient. Book your seats in advance for popular routes.\n"
            "Alice: What about accommodation? Hostels or hotels?\n"
            "Bob: Try a ryokan - traditional Japanese inn. They're expensive but worth it.\n"
            "Alice: How much should I budget for 10 days?\n"
            "Bob: Around 2000-3000 USD including flights from the US.\n"
            "Alice: Thanks Bob! This is really helpful.\n"
            "Bob: No problem! Have a great trip.\n\n"
            "Now Alice asks: So what did Bob say about the JR Pass? "
        ),
        "check": ["JR Pass"],
        "window": 64,
        "desc": "Multi-turn dialogue - must retrieve JR Pass detail from earlier in conversation"
    },

    "json_format": {
        "prompt": (
            "I need you to generate a JSON object with specific fields.\n"
            "The JSON schema is:\n"
            "{\n"
            '  "user": {"type": "string", "description": "Full name of the user"},\n'
            '  "age": {"type": "integer", "description": "Age in years"},\n'
            '  "email": {"type": "string", "description": "Email address"},\n'
            '  "preferences": {\n'
            '    "theme": {"type": "string", "enum": ["dark", "light"]},\n'
            '    "notifications": {"type": "boolean"},\n'
            '    "language": {"type": "string", "default": "en"},\n'
            '    "items_per_page": {"type": "integer", "minimum": 10, "maximum": 100}\n'
            "  },\n"
            '  "tags": {"type": "array", "items": {"type": "string"}},\n'
            '  "created_at": {"type": "string", "format": "date-time"}\n'
            "}\n\n"
            "Now generate a valid JSON object using this schema for a user named "
            "Alex Johnson who is 28 years old, prefers dark theme, wants notifications on, "
            "speaks French, wants 50 items per page, and has tags: developer, open-source.\n\n"
            "The JSON must be valid. Output: "
        ),
        "check": ["{", "\"language\": \"fr", "50"],
        "window": 64,
        "desc": "JSON format generation - retrieval must keep schema constraints"
    },

    "long_coref": {
        "prompt": (
            "Let me define some terms first.\n"
            "Term ALPHA: A process by which data is transformed from raw format to structured format "
            "using supervised learning models. It typically requires labeled training data and "
            "produces a confidence score between 0 and 1.\n"
            "Term BETA: A caching mechanism that stores frequently accessed results in memory "
            "to reduce latency for subsequent requests. Uses LRU eviction policy by default.\n"
            "Term GAMMA: A security protocol that encrypts data in transit using AES-256 "
            "and authenticates both parties before establishing a connection.\n"
            "Term DELTA: A load balancing algorithm that distributes incoming requests across "
            "multiple servers based on current CPU utilization and response time.\n"
            "Term EPSILON: A data compression technique that reduces file size by removing "
            "redundant information while preserving semantic meaning. Achieves 10x compression.\n"
            "Term ZETA: A monitoring system that tracks application performance metrics "
            "including request latency, error rates, and throughput.\n"
            "Term ETA: A testing framework that supports unit, integration, and end-to-end tests "
            "with automatic parallel execution and coverage reporting.\n"
            "Term THETA: A deployment pipeline that automates building, testing, and releasing "
            "applications across development, staging, and production environments.\n"
            "Term IOTA: A database indexing strategy that uses B-trees for range queries "
            "and hash indexes for equality lookups.\n"
            "Term KAPPA: A messaging queue system that guarantees at-least-once delivery "
            "and supports publish-subscribe and point-to-point patterns.\n\n"
            "Now answer: Which term defined above describes a caching mechanism, "
            "and what eviction policy does it use? "
        ),
        "check": ["BETA", "LRU"],
        "window": 64,
        "desc": "Long-range coreference - must retrieve BETA term definition from 500+ tok away"
    },

    "dense_facts": {
        "prompt": (
            "Here are important historical events and their years:\n"
            "1492: Columbus reaches the Americas.\n"
            "1776: American Declaration of Independence.\n"
            "1789: French Revolution begins.\n"
            "1815: Battle of Waterloo.\n"
            "1865: US Civil War ends.\n"
            "1914: World War I begins.\n"
            "1917: Russian Revolution.\n"
            "1918: World War I ends.\n"
            "1929: Stock Market Crash.\n"
            "1939: World War II begins.\n"
            "1945: World War II ends.\n"
            "1947: Indian Independence.\n"
            "1957: Sputnik launched.\n"
            "1963: JFK assassination.\n"
            "1969: Moon landing.\n"
            "1989: Fall of Berlin Wall.\n"
            "1991: Soviet Union collapses.\n"
            "2001: 9/11 attacks.\n"
            "2008: Global financial crisis.\n"
            "2020: COVID-19 pandemic.\n\n"
            "Question: In which year did World War II end, and which year did the "
            "French Revolution begin? "
        ),
        "check": ["1945", "1789"],
        "window": 64,
        "desc": "Dense fact list - retrieval must pick correct year-event pairs from many similar entries"
    },

    "hallucination_sensitive": {
        "prompt": (
            "Here are some verified facts for your reference:\n"
            "The Amazon rainforest produces approximately 6% of the world's oxygen. "
            "The Great Wall of China is not visible from space with the naked eye. "
            "Mount Everest is 8848 meters tall and located in the Himalayas. "
            "The speed of light in a vacuum is approximately 299,792,458 meters per second. "
            "Octopuses have three hearts and blue blood. "
            "Bananas are berries botanically, while strawberries are not. "
            "A day on Venus is longer than a year on Venus. "
            "Honey never spoils - archaeologists found 3000-year-old honey in Egyptian tombs. "
            "Cleopatra lived closer in time to the Moon landing than to the construction of the Great Pyramid. "
            "Wombat poop is cube-shaped to prevent it from rolling away. "
            "The shortest war in history lasted only 38 minutes between Britain and Zanzibar in 1896. "
            "There are more trees on Earth than stars in the Milky Way galaxy. "
            "The Eiffel Tower can be 15 cm taller during summer due to thermal expansion. "
            "Some penguins propose to their mates with pebbles. "
            "The longest recorded flight of a chicken is 13 seconds. "
            "A jiffy is an actual unit of time: 1/100th of a second. "
            "The word 'nerd' was first coined by Dr. Seuss in his 1950 book 'If I Ran the Zoo'. "
            "The total weight of all ants on Earth is roughly equal to the total weight of all humans. "
            "Octopuses have nine brains and can change color in milliseconds. "
            "The human nose can detect over 1 trillion distinct scents.\n\n"
            "Based ONLY on the facts above: Is the Great Wall visible from space? "
            "Answer exactly as stated in the facts: "
        ),
        "check": ["not visible"],
        "window": 64,
        "desc": "Hallucination sensitive - must retrieve exact fact without adding common misconceptions"
    },
}


@torch.no_grad()
def main():
    global DEV
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval(); DEV = model.device

    all_results = {}

    for name, sc in SCENARIOS.items():
        prompt = sc["prompt"]
        ws = sc["window"]
        checks = sc["check"]
        desc = sc["desc"]
        max_new = 40

        inp = tok(prompt, return_tensors="pt", truncation=True, max_length=4096).to(DEV)
        ctx = inp["input_ids"].shape[1]
        if ctx <= ws:
            prompt += "\n" + ("More context. " * 20)
            inp = tok(prompt, return_tensors="pt", truncation=True, max_length=4096).to(DEV)
            ctx = inp["input_ids"].shape[1]

        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"  {desc}")
        print(f"  ctx={ctx} tok, window={ws}, ret_active={ctx > ws}")
        print(f"{'='*60}")

        # Native
        native_out = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
            pad_token_id=tok.eos_token_id, return_dict_in_generate=True, output_logits=True)
        native_new = tok.decode(native_out.sequences[0, ctx:ctx+max_new].cpu(), skip_special_tokens=True)
        native_full = tok.decode(native_out.sequences[0].cpu(), skip_special_tokens=True)
        native_check = all(c.lower() in native_new.lower() for c in (checks if isinstance(checks, list) else [checks]))

        # KVR
        gen = KVRGenerator(model, window_size=ws, top_k=128, device=DEV)
        gen.prefill(inp["input_ids"])

        gen_ids = []
        for step in range(max_new):
            tid = None if step > 0 else inp["input_ids"][0, -1]
            nid = gen.step(token_id=tid, temperature=1.0, top_k=1)
            gen_ids.append(nid.item())

        kvr_new = tok.decode(gen_ids, skip_special_tokens=True)
        kvr_full = tok.decode(torch.cat([inp["input_ids"].cpu(), torch.tensor([gen_ids])], dim=1)[0], skip_special_tokens=True)

        kvr_check = all(c.lower() in kvr_new.lower() for c in (checks if isinstance(checks, list) else [checks]))

        all_results[name] = {
            "desc": desc,
            "context_tokens": ctx,
            "window_size": ws,
            "retrieval_active": ctx > ws,
            "native": {"new": native_new, "full": native_full, "check_pass": native_check},
            "kvr": {"new": kvr_new, "full": kvr_full, "check_pass": kvr_check},
            "checks": checks,
        }

        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Scenario':<25s} {'Native':>8s} {'KVR':>8s} {'ctx/win':>10s}")
    print("-" * 55)
    for name, r in sorted(all_results.items()):
        n = "PASS" if r["native"]["check_pass"] else "FAIL"
        k = "PASS" if r["kvr"]["check_pass"] else "FAIL"
        print(f"{name:<25s} {n:>8s} {k:>8s} {r['context_tokens']}/{r['window_size']:>4d}")

    json.dump(all_results, open("eval_kvr_scenarios.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nSaved eval_kvr_scenarios.json")


if __name__ == "__main__":
    main()
