import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import urllib.request

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



MODEL_PATH = 'model4.h5'

MODEL_URL = 'https://docs.google.com/uc?export=download&id=1WuoWc8B919dENaY1Ynu0cUogi4cmaSuY'

if not os.path.exists(MODEL_PATH):
    print("Downloading AI model matrix assets from Google Drive...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete!")
    except Exception as e:
        print(f"Error downloading model: {e}")

model = load_model(MODEL_PATH)

EMOTION_CLASSES = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'
}

MOOD_INSIGHTS = {
    'Angry': "High voltage detected. Take a gentle breath. Let's step away for a moment and reset your posture. 🌿",
    'Disgust': "Something isn't sitting quite right with you. Take a glass of water and re-center your thoughts. ✨",
    'Fear': "Elevated tension detected. Try a slow, 4-second box breath. You have complete control over this moment. 🧘",
    'Happy': "Your climate is completely radiant! Keep sharing this optimal energy parameter. ☀️",
    'Neutral': "Calm focus calibrated. Your mind is in a balanced, grounded state. Perfect for deep concentration. 💻",
    'Sad': "A lower emotional velocity detected. It is completely okay to feel down. Give yourself space to rest. ☕",
    'Surprise': "A sudden spike in awareness! Did something unexpected grab your interest or spark your focus? 😮"
}

def preprocess_image(image_path):
    img = load_img(image_path, color_mode="grayscale", target_size=(48, 48))
    img_array = img_to_array(img)
    normalized_array = img_array.astype('float32') / 255.0
    return np.expand_dims(normalized_array, axis=0)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/checker')
def checker():
    return render_template('check.html')

@app.route('/game')
def game():
    return render_template('game.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No image file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Empty filename string'}), 400

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        try:
            processed_img = preprocess_image(file_path)
            predictions = model.predict(processed_img)[0]
            max_index = int(np.argmax(predictions))
            predicted_emotion = EMOTION_CLASSES[max_index]
            
            os.remove(file_path)
            return jsonify({
                'success': True,
                'prediction_index': max_index,
                'emotion': predicted_emotion,
                'insight': MOOD_INSIGHTS[predicted_emotion],
                'all_scores': {EMOTION_CLASSES[i]: float(predictions[i] * 100) for i in range(7)}
            })
        except Exception as e:
            if os.path.exists(file_path): os.remove(file_path)
            return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
    
