from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import os
import time
from PIL import Image
import google.generativeai as genai

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "best_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels
CLASS_LABELS = [
    "Eczema", "Warts, Molluscum, and other Viral Infections", "Melanoma",
    "Atopic Dermatitis", "Basal Cell Carcinoma (BCC)", "Melanocytic Nevi (NV)",
    "Benign Keratosis-like Lesions (BKL)", "Psoriasis, Lichen Planus, and related diseases",
    "Seborrheic Keratoses and other Benign Tumors", "Tinea Ringworm, Candidiasis, and other Fungal Infections"
]

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Gemini API Key
GENAI_API_KEY = "Your_gemini_api_key"
genai.configure(api_key=GENAI_API_KEY)
model_gemini = genai.GenerativeModel("gemini-2.0-flash")

# Image preprocessing function
def preprocess_image(img_path):
    try:
        img = Image.open(img_path).convert("RGB")  # Convert to RGB
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    img = img.resize((224, 224))  # Resize for model input
    img = np.array(img).astype("float32") / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)
    return img

# Predict function
def model_predict(img_path, model):
    processed_img = preprocess_image(img_path)
    if processed_img is None:
        return "Invalid Image Format", 0.0

    predictions = model.predict(processed_img)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    return CLASS_LABELS[predicted_class], confidence

# Get AI-based treatment recommendation
def get_treatment_recommendation(disease):
    prompt = f"""
    You are a knowledgeable AI assistant providing structured and detailed medical treatment recommendations. 
    Format the response exactly as shown below:

    ✅ Treatment Recommendations for {disease}

    1. ✅ Assessment
        Conduct a thorough physical examination to determine the extent and severity of the condition.
        Obtain a detailed patient history, including symptoms, duration, triggers, lifestyle factors, and previous treatments.
        Consider laboratory tests or imaging if needed to confirm the diagnosis.
        Evaluate for any underlying conditions that may contribute to the disease.

    2. ✅ Treatment Options
    Medications:
        Topical treatments: (e.g., corticosteroids, salicylic acid, imiquimod) to reduce inflammation and control symptoms.
        Oral medications: (e.g., antihistamines for itching, antiviral drugs like acyclovir for viral infections).
        Immunosuppressants: (e.g., methotrexate, cyclosporine) for severe cases where topical treatments are insufficient.
        Antibiotics: If secondary bacterial infection is present.

    Therapies:
        Cryotherapy: (freezing) commonly used for warts, molluscum, and viral skin infections.
        Phototherapy: UV light therapy for chronic skin conditions like eczema and psoriasis.
        Laser therapy: Effective for stubborn or resistant lesions.
        Curettage: (scraping) for molluscum contagiosum or keratotic lesions.

    Procedures:
        Surgical excision: For large, persistent, or recurrent growths.
        Electrocautery: Using heat to remove warts and lesions.
        Chemical peels: For skin resurfacing in cases of scarring or discoloration.

    3. ✅ Prevention & Lifestyle Changes
        Maintain proper skin hygiene with mild, fragrance-free soaps and moisturizers.
        Avoid known irritants, allergens, and triggers that can worsen symptoms.
        Strengthen the immune system through a balanced diet, regular exercise, and adequate sleep.
        Use protective measures (e.g., wearing gloves, sunscreen, or protective clothing).
        Educate patients about the contagious nature of certain infections and proper precautions.
        Vaccination may help prevent certain viral infections (e.g., HPV vaccine for warts).

    Now, generate a structured response for {disease} following this exact format. Ensure the response is **well-detailed**, providing **more treatment options** and **clear alignment**. Avoid using large text sizes and unnecessary spacing.
    """

    response = model_gemini.generate_content(prompt)
    return response.text

# Store conversation history for each user session
conversation_history = {}

# Function to summarize conversation history
def summarize_conversation(history):
    summary = "Conversation Summary:\n"
    for msg in history:
        summary += f"{msg['role']}: {msg['text']}\n"
    return summary

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    file_path = None

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index2.html", error="No file uploaded!")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index2.html", error="No selected file!")

        if file:
            filename = f"{int(time.time())}_{file.filename}"
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            prediction, confidence = model_predict(file_path, model)

    return render_template("index2.html", file_path=file_path, prediction=prediction, confidence=confidence)

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
    session_id = data.get("session_id", "default")  # Use a session ID to track conversations

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Initialize conversation history for the session if it doesn't exist
    if session_id not in conversation_history:
        conversation_history[session_id] = []

    # Add the user's message to the conversation history
    conversation_history[session_id].append({"role": "user", "text": user_message})

    # Summarize the conversation history (Limit to last 5 messages)
    history = conversation_history[session_id][-5:]
    history_text = "\n".join(f"{msg['role']}: {msg['text']}" for msg in history)

    # Structured prompt to guide Gemini
    prompt = f"""
    You are a medical assistant chatbot specializing in skin diseases.
    Your task is to provide clear, medically accurate, and structured responses.
    Use ✅ before any heading.
    Do not use symbols (**) before and after headings in the generated text.
    Avoid using bullet points (*) or any unnecessary formatting.

    If the user asks about a specific disease, provide:
    1. ✅ Symptoms
    2. ✅ Diagnosis methods
    3. ✅ Treatment recommendations (both medical and home remedies)
    
    If the user asks general questions, respond concisely and accurately.

    --- Conversation History ---
    {history_text}

    User: {user_message}
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
    app.run(debug=True)