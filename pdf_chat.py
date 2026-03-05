import ollama
from pypdf import PdfReader

def read_pdf(file_path):
    """Extract all text from a PDF file"""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def chat_with_pdf(pdf_path):
    print(f"📄 Loading PDF: {pdf_path}")
    
    # Read the PDF
    pdf_text = read_pdf(pdf_path)
    print(f"✅ PDF loaded — {len(pdf_text)} characters extracted")
    print("-" * 30)
    
    # Build system prompt with PDF content
    system_prompt = f"""You are a helpful assistant. 
Answer questions based on this document:

{pdf_text}

If the answer is not in the document, say 'I could not find that in the document.'
"""
    
    messages = [{'role': 'system', 'content': system_prompt}]
    
    print("🤖 PDF Chatbot Ready — Ask anything about your document!")
    print("Type 'quit' to exit")
    print("-" * 30)
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
            
        if user_input.strip() == '':
            continue
        
        messages.append({'role': 'user', 'content': user_input})
        
        response = ollama.chat(
            model='qwen2.5:3b-instruct',
            messages=messages
        )
        
        reply = response['message']['content']
        messages.append({'role': 'assistant', 'content': reply})
        
        print(f"\nAI: {reply}\n")
        print("-" * 30)

import sys

if len(sys.argv) < 2:
    print("❌ Please provide a PDF path")
    print("Usage: python pdf_chat.py /path/to/your/file.pdf")
else:
    chat_with_pdf(sys.argv[1])
