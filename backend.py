import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from langdetect import detect
import sqlite3
from huggingface_hub import InferenceClient
import logging

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize database
def init_db():
    conn = sqlite3.connect('qa_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS qa_logs
                 (timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  question TEXT,
                  answer TEXT,
                  language TEXT,
                  model TEXT)''')
    conn.commit()
    conn.close()

init_db()

# Initialize Hugging Face client
client = InferenceClient()

def process_document(file_path, question, model="llama"):
    """Process document and generate answer using specified model"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Detect language
        language = detect(content)
        
        # Prepare prompt
        prompt = f"""Context: {content}\n\nQuestion: {question}\n\nAnswer:"""
        
        # Get model response
        if model.lower() == "llama":
            model_id = "meta-llama/Llama-2-70b-chat-hf"
        else:  # DeepSeek
            model_id = "deepseek-ai/deepseek-coder-33b-instruct"
            
        response = client.text_generation(
            prompt,
            model=model_id,
            max_new_tokens=512,
            temperature=0.7
        )
        
        # Log to database
        conn = sqlite3.connect('qa_history.db')
        c = conn.cursor()
        c.execute("INSERT INTO qa_logs (question, answer, language, model) VALUES (?, ?, ?, ?)",
                 (question, response, language, model))
        conn.commit()
        conn.close()
        
        return {"answer": response, "language": language}
    
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return {"error": str(e)}

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({"message": "File uploaded successfully",
                       "filepath": filepath})

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    if not all(k in data for k in ['filepath', 'question', 'model']):
        return jsonify({"error": "Missing required parameters"}), 400
    
    result = process_document(
        data['filepath'],
        data['question'],
        data.get('model', 'llama')
    )
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)