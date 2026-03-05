import json
import os
import ollama
from datetime import datetime

MEMORY_FILE = "memory.json"

def load_memory():
    """Load memory from disk"""
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r') as f:
            return json.load(f)
    return {
        "user": {},
        "facts": [],
        "preferences": [],
        "conversations": [],
        "created_at": datetime.now().isoformat()
    }

def save_memory(memory):
    """Save memory to disk"""
    with open(MEMORY_FILE, 'w') as f:
        json.dump(memory, f, indent=2)

def is_duplicate(new_item, existing_list):
    """Check if item is already in list (case insensitive, fuzzy)"""
    new_lower = new_item.lower().strip()
    for existing in existing_list:
        existing_lower = existing.lower().strip()
        # Check exact match
        if new_lower == existing_lower:
            return True
        # Check if one contains the other
        if new_lower in existing_lower or existing_lower in new_lower:
            return True
    return False

def extract_memories(conversation, memory):
    """Use AI to extract important facts from conversation"""

    conversation_text = "\n".join([
        f"{msg['role'].upper()}: {msg['content']}"
        for msg in conversation
    ])

    response = ollama.chat(
        model='samrat-ai',
        messages=[
            {
                'role': 'system',
                'content': """You are a memory extraction system.
Extract ONLY new important facts not already known.
Return ONLY a valid JSON object:
{
  "user_facts": ["fact1", "fact2"],
  "preferences": ["pref1", "pref2"],
  "important_info": ["info1", "info2"]
}
Rules:
- Keep facts short and clear
- No duplicate meanings
- Only truly important information
- Empty lists if nothing new
Return ONLY JSON, no other text."""
            },
            {
                'role': 'user',
                'content': f"Already known facts: {memory['facts']}\n\nAlready known preferences: {memory['preferences']}\n\nExtract NEW memories from:\n\n{conversation_text}"
            }
        ]
    )

    try:
        text = response['message']['content'].strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        extracted = json.loads(text)

        # Add only non-duplicate facts
        for fact in extracted.get('user_facts', []):
            if not is_duplicate(fact, memory['facts']):
                memory['facts'].append(fact)

        for pref in extracted.get('preferences', []):
            if not is_duplicate(pref, memory['preferences']):
                memory['preferences'].append(pref)

        for info in extracted.get('important_info', []):
            if not is_duplicate(info, memory['facts']):
                memory['facts'].append(info)

    except:
        pass

    return memory

def build_memory_prompt(memory):
    """Build memory context for system prompt"""
    
    if not memory['facts'] and not memory['preferences']:
        return ""
    
    prompt = "\n## What You Remember About Samrat\n"
    
    if memory['facts']:
        prompt += "\n### Known Facts:\n"
        for fact in memory['facts'][-20:]:  # Last 20 facts
            prompt += f"- {fact}\n"
    
    if memory['preferences']:
        prompt += "\n### Preferences:\n"
        for pref in memory['preferences'][-10:]:  # Last 10 preferences
            prompt += f"- {pref}\n"
    
    prompt += "\nUse this information naturally in conversation without explicitly saying 'I remember that...'\n"
    
    return prompt

def add_conversation_summary(memory, conversation):
    """Save a summary of the conversation"""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "message_count": len(conversation),
        "preview": conversation[0]['content'][:100] if conversation else ""
    }
    memory['conversations'].append(summary)
    
    # Keep only last 50 conversation summaries
    if len(memory['conversations']) > 50:
        memory['conversations'] = memory['conversations'][-50:]
    
    return memory

def clear_memory():
    """Wipe all memory"""
    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)
    print("🗑️ Memory cleared!")

def show_memory():
    """Display current memory"""
    memory = load_memory()
    print("\n🧠 Samrat AI Memory")
    print("=" * 40)
    
    if memory['facts']:
        print("\n📌 Known Facts:")
        for fact in memory['facts']:
            print(f"  • {fact}")
    else:
        print("\n📌 No facts stored yet")
        
    if memory['preferences']:
        print("\n❤️ Preferences:")
        for pref in memory['preferences']:
            print(f"  • {pref}")
    else:
        print("\n❤️ No preferences stored yet")
    
    print(f"\n💬 Total conversations: {len(memory['conversations'])}")
    print(f"📅 Memory created: {memory.get('created_at', 'Unknown')}")
    print("=" * 40)
