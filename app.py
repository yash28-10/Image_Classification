import os
from flask import Flask, render_template, request
from ultralytics import YOLO

# Initialize Flask App
app = Flask(__name__)

# Configure Upload Folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained YOLO11 classification model
# Make sure 'best.pt' is in the same folder as this script!
model = YOLO('best.pt')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    image_path = None

    if request.method == 'POST':
        # Check if an image was uploaded
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No file selected")

        if file:
            # Save the uploaded image
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # Update path for HTML rendering
            image_path = filepath

            # Run YOLO11 Prediction
            results = model(filepath)
            
            # Extract the top predicted class and confidence score
            top1_index = results[0].probs.top1
            prediction = results[0].names[top1_index].capitalize()
            confidence = round(float(results[0].probs.top1conf) * 100, 2)

    return render_template('index.html', prediction=prediction, confidence=confidence, image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)