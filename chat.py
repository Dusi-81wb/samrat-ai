import ollama
import warnings
warnings.filterwarnings("ignore")
from memory import (
    load_memory, save_memory, extract_memories,
    build_memory_prompt, add_conversation_summary, show_memory
)

SYSTEM_BASE = """You are Samrat AI — an advanced personal AI assistant 
built and owned by Samrat. You run 100% locally with complete privacy.
You are helpful, intelligent, and remember important things about Samrat.
Reason step by step for complex problems."""

def chat():
    # Load persistent memory
    memory = load_memory()
    memory_context = build_memory_prompt(memory)
    
    # Build system prompt with memory
    system_prompt = SYSTEM_BASE + memory_context
    
    messages = [{'role': 'system', 'content': system_prompt}]
    conversation = []
    
    print("🧠 Samrat AI — with Persistent Memory")
    print("Commands: 'quit', 'memory', 'clear memory'")
    print("-" * 40)
    
    # Greet based on memory
    if memory['facts']:
        print("✅ Memory loaded — Samrat AI remembers you!\n")
    else:
        print("👋 No memory yet — start chatting!\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
            
        if user_input.lower() == 'quit':
            # Extract and save memories before quitting
            print("\n💾 Saving memories...")
            memory = extract_memories(conversation, memory)
            memory = add_conversation_summary(memory, conversation)
            save_memory(memory)
            print("✅ Memory saved! Goodbye!")
            break
        
        if user_input.lower() == 'memory':
            show_memory()
            continue
            
        if user_input.lower() == 'clear memory':
            from memory import clear_memory
            clear_memory()
            memory = load_memory()
            print("✅ Memory cleared!")
            continue
        
        # Add to conversation
        messages.append({'role': 'user', 'content': user_input})
        conversation.append({'role': 'user', 'content': user_input})
        
        # Get response
        response = ollama.chat(model='samrat-ai', messages=messages)
        reply = response['message']['content']
        
        messages.append({'role': 'assistant', 'content': reply})
        conversation.append({'role': 'assistant', 'content': reply})
        
        print(f"\nSamrat AI: {reply}\n")
        print("-" * 40)
        
        # Auto save memory every 10 messages
        if len(conversation) % 10 == 0:
            print("💾 Auto-saving memory...")
            memory = extract_memories(conversation, memory)
            save_memory(memory)

if __name__ == "__main__":
    chat()
