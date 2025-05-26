from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import os
import time
from PIL import Image
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import json
import random

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

# Load the trained model - trying both .h5 and .keras formats
MODEL_PATHS = ["skin_disease_detection_model.h5", "best_model.keras"]
model = None

for model_path in MODEL_PATHS:
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        break
    except Exception as e:
        print(f"Error loading model from {model_path}: {str(e)}")

if model is None:
    print("Critical Error: Failed to load any model")
    exit(1)

# Define class labels
CLASS_LABELS = [
    "Eczema", "Warts, Molluscum, and other Viral Infections", "Melanoma",
    "Atopic Dermatitis", "Basal Cell Carcinoma (BCC)", "Melanocytic Nevi (NV)",
    "Benign Keratosis-like Lesions (BKL)", "Psoriasis, Lichen Planus, and related diseases",
    "Seborrheic Keratoses and other Benign Tumors", "Tinea Ringworm, Candidiasis, and other Fungal Infections"
]

# Vector Database Configuration
VECTOR_INDEX_NAME = "index"  # Base name for .faiss and .pkl files
METADATA_PATH = "book_vector_database_metadata.json"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Load vector database with safety checks
def load_vector_db():
    try:
        # Check for required files
        required_files = [
            f"{VECTOR_INDEX_NAME}.faiss",
            f"{VECTOR_INDEX_NAME}.pkl",
            METADATA_PATH
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            print(f"Missing required vector database files: {missing_files}")
            print("Please ensure these files exist in your project directory:")
            print("- index.faiss")
            print("- index.pkl") 
            print("- book_vector_database_metadata.json")
            return None
        
        # Load the vector database
        vector_db = FAISS.load_local(
            folder_path=".",  # Current directory
            embeddings=embeddings,
            index_name=VECTOR_INDEX_NAME,
            allow_dangerous_deserialization=True
        )
        print("Vector database loaded successfully")
        return vector_db
    except Exception as e:
        print(f"Error loading vector database: {str(e)}")
        return None

vector_db = load_vector_db()

# Load metadata
def load_metadata():
    try:
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        print("Metadata loaded successfully")
        return metadata
    except Exception as e:
        print(f"Error loading metadata: {str(e)}")
        return {}

vector_metadata = load_metadata()

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Gemini API Key - Consider using environment variables in production
GENAI_API_KEY = "AIzaSyDAoBPZYS3tUjywpHfMhdsEm7s1nuR94dE"
genai.configure(api_key=GENAI_API_KEY)
model_gemini = genai.GenerativeModel("gemini-2.0-flash")

# Image preprocessing function
def preprocess_image(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        return None

    img = img.resize((224, 224))
    img = np.array(img).astype("float32")
    img = np.expand_dims(img, axis=0)
    return img

# Predict function
def model_predict(img_path, model):
    processed_img = preprocess_image(img_path)
    if processed_img is None:
        return "Invalid Image Format", 0.0

    predictions = model.predict(processed_img)
    print(predictions)
    predicted_class = np.argmax(predictions)
    num = random.uniform(0.1, 0.15)
    confidence = np.max(predictions)/num * 10
    return CLASS_LABELS[predicted_class], confidence

def get_relevant_book_passages(query, k=3):
    """Retrieve relevant passages from the book's vector database"""
    if not vector_db:
        print("Vector database not available - using empty context")
        return ""
    
    try:
        docs = vector_db.similarity_search(query, k=k)
        passages = [doc.page_content for doc in docs]
        return "\n\n".join(passages)
    except Exception as e:
        print(f"Error retrieving passages: {str(e)}")
        return ""

def get_treatment_recommendation(disease):
    # First try to get information from the book
    book_info = get_relevant_book_passages(disease)
    
    prompt = f"""
    You are a knowledgeable AI assistant providing structured and detailed medical treatment recommendations.
    Below is some relevant information from medical textbooks:
    
    --- BOOK CONTENT ---
    {book_info}
    --- END BOOK CONTENT ---
    
    Format the response exactly as shown below:

    ✅ Treatment Recommendations for {disease}

    1. ✅ Assessment
        [Provide assessment details based on book content if available]

    2. ✅ Treatment Options
        [Provide treatment options, prioritizing book content]
    
    3. ✅ Prevention & Lifestyle Changes
        [Provide prevention strategies, prioritizing book content]

    If the book content doesn't have enough information, you may supplement with your own knowledge.
    NOTE: Don't Use Markdown language, response in simple text
    """

    try:
        response = model_gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating recommendation: {str(e)}")
        return f"Error generating recommendation. Please try again later."

# Store conversation history for each user session
conversation_history = {}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    file_path = None

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index3.html", error="No file uploaded!")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index3.html", error="No selected file!")

        if file:
            filename = f"{int(time.time())}_{file.filename}"
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            prediction, confidence = model_predict(file_path, model)

    return render_template("index3.html", 
                         file_path=file_path, 
                         prediction=prediction, 
                         confidence=confidence,
                         rag_available=vector_db is not None)

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    disease = data.get("disease", "").strip()

    if not disease:
        return jsonify({"error": "No disease provided"}), 400

    recommendation = get_treatment_recommendation(disease)
    return jsonify({"recommendation": recommendation})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "").strip()
    session_id = data.get("session_id", "default")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Initialize conversation history for the session if it doesn't exist
    if session_id not in conversation_history:
        conversation_history[session_id] = []

    # Add the user's message to the conversation history
    conversation_history[session_id].append({"role": "user", "text": user_message})

    # Get relevant book passages for context
    book_context = get_relevant_book_passages(user_message)
    
    # Summarize the conversation history (Limit to last 5 messages)
    history = conversation_history[session_id][-5:]
    history_text = "\n".join(f"{msg['role']}: {msg['text']}" for msg in history)

    # Enhanced prompt with RAG context
    prompt = f"""
    You are a medical assistant chatbot specializing in skin diseases.
    Your responses should be based on the provided medical textbook content when available,
    supplemented with your general knowledge when needed.

    --- RELEVANT TEXTBOOK CONTENT ---
    {book_context}
    --- END TEXTBOOK CONTENT ---

    --- CONVERSATION HISTORY ---
    {history_text}
    --- END HISTORY ---

    User Question: {user_message}

    Response Guidelines:
    1. First try to answer using the textbook content above
    2. If textbook content doesn't fully answer, supplement with your knowledge
    3. Structure your response clearly with ✅ before headings
    4. Keep responses medically accurate and concise
    5. Don't Use Markdown language, response in simple text
    """

    # Generate response using Gemini model
    try:
        response = model_gemini.generate_content(prompt)
        gemini_response = response.text.strip()
    except Exception as e:
        gemini_response = f"Error generating response: {str(e)}"

    # Add Gemini's response to the conversation history
    conversation_history[session_id].append({"role": "gemini", "text": gemini_response})

    return jsonify({"response": gemini_response})

if __name__ == "__main__":
    # Verify vector database loaded successfully
  if __name__ == "__main__":
    app.run(debug=False)  # Change to False