from flask import Flask, render_template, request
import numpy as np
import joblib
import cv2
from skimage.feature import hog
import os
import gdown

# Model file download
MODEL_FILE = "parkinson_knn_model.pkl"
FILE_ID = "1--rgHv5E4qMLeK_0kCODr2jhlCHeDo9b"
GDRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_FILE):
    print("Downloading model file from Google Drive...")
    gdown.download(GDRIVE_URL, MODEL_FILE, quiet=False)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load models only when this route is called
    svm_model = joblib.load('parkinsons_svm_model.pkl')
    dct_model = joblib.load('parkinson_dct_model.pkl')
    knn_image_model = joblib.load('parkinson_knn_model.pkl')

    result_svm = result_dct = result_knn_image = None

    # --- SVM input ---
    svm_features = [
        'RMS', 'MAX_BETWEEN_ET_HT', 'MIN_BETWEEN_ET_HT', 'STD_DEVIATION_ET_HT',
        'MRT', 'MAX_HT', 'MIN_HT', 'STD_HT', 'CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT'
    ]
    svm_values = [float(request.form.get(f, 0)) for f in svm_features]
    result_svm = svm_model.predict([svm_values])[0]

    # --- DCT input ---
    dct_features = [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
        'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
        'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
        'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
        'spread1', 'spread2', 'D2', 'PPE'
    ]
    dct_values = [float(request.form.get(f, 0)) for f in dct_features]
    result_dct = dct_model.predict([dct_values])[0]

    # --- KNN Image input ---
    image = request.files['image']
    file_bytes = np.frombuffer(image.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    image_resized = cv2.resize(image, (80, 80))
    gray_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY) if len(image_resized.shape) == 3 else image_resized
    features = hog(gray_image).reshape(1, -1)
    result_knn_image = knn_image_model.predict(features)[0]

    # Final Decision
    combined_score = 0.3 * result_svm + 0.3 * result_dct + 0.4 * result_knn_image
    final_output = 1 if combined_score >= 0.6 else 0

    return render_template(
        'result.html',
        final_output=final_output,
        result_dct=result_dct,
        result_svm=result_svm,
        result_knn_image=result_knn_image,
        show_knn_form=(final_output == 1)
    )

@app.route('/predict-knn-form', methods=['POST'])
def predict_knn_form():
    knn_form_model = joblib.load('knn_model.pkl')  # Load only when needed

    knn_form_features = [
        'Speech', 'Facial Expression', 'Rest Tremor (Face/Lips)', 'Rest Tremor (Right Hand)',
        'Rest Tremor (Left Hand)', 'Rest Tremor (Right Leg)', 'Rest Tremor (Left Leg)',
        'Action/Postural Tremor (Right Hand)', 'Action/Postural Tremor (Left Hand)', 'Rigidity (Neck)',
        'Rigidity (Right Arm)', 'Rigidity (Left Arm)', 'Rigidity (Right Leg)', 'Rigidity (Left Leg)',
        'Finger Tapping (Right)', 'Finger Tapping (Left)', 'Hand Movements (Right)', 'Hand Movements (Left)',
        'Pronation-Supination (Right)', 'Pronation-Supination (Left)', 'Toe Tapping (Right)',
        'Toe Tapping (Left)', 'Leg Agility (Right)', 'Leg Agility (Left)', 'Arising from Chair',
        'Posture', 'Gait', 'Postural Stability', 'Body Bradykinesia'
    ]
    values = [float(request.form.get(f, 0)) for f in knn_form_features]
    total_sum = sum(values)
    result = knn_form_model.predict(np.array([[total_sum]]))[0]

    return render_template('knn_result.html', result=result)

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
