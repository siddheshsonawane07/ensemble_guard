from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os

# Import the modified `ensemblePredict` file
from ensemblePredict import predict_single_pe

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'exe'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Check for file in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        # Check allowed file type 
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Call ensemblePredict.py with the file path and get the prediction
            result = predict_single_pe(file_path)

            
            # Return the prediction as JSON
            return jsonify(result)

        else:
            return jsonify({'error': 'Invalid file type'})

    except Exception as e:
        # Return error message in JSON format
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
