from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import os
import numpy as np
import joblib
import pandas as pd
import cv2
import requests
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

app = Flask(__name__)

# Configure the upload folder
UPLOAD_FOLDER = './static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the upload folder exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Get the user's location based on their IP address
def get_user_location():
    try:
        response = requests.get('https://ipinfo.io/json')  # Get location data
        data = response.json()

        if response.status_code == 200:
            location = data['loc'].split(',')  # 'loc' contains 'latitude,longitude'
            print(f"City: {data['city']}, Country: {data['country']}")
            print(f"Latitude: {location[0]}, Longitude: {location[1]}")
            return float(location[0]), float(location[1])  # Returns (latitude, longitude)
        else:
            print("Failed to retrieve location.")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

# Function to get the nearest hospital using Overpass API
def get_nearest_hospital(lat, lon):
    try:
        # Use Overpass API to find nearby hospitals
        overpass_url = "http://overpass-api.de/api/interpreter"
        query = f"""
        [out:json];
        node["amenity"="hospital"](around:5000,{lat},{lon});
        out body;
        """
        response = requests.get(overpass_url, params={'data': query})
        data = response.json()

        # Parse results
        if 'elements' in data and len(data['elements']) > 0:
            hospital = data['elements'][0]  # Get the first hospital (you can modify this if you want more results)
            hospital_name = hospital['tags'].get('name', 'Unknown')
            hospital_lat = hospital['lat']
            hospital_lon = hospital['lon']
            hospital_address = f"Latitude: {hospital_lat}, Longitude: {hospital_lon}"

            return {
                'name': hospital_name,
                'address': hospital_address,
                'latitude': hospital_lat,
                'longitude': hospital_lon
            }
        else:
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about.html")
def about():
    return render_template("about.html")

@app.route("/contact.html")
def contact():
    return render_template("contact.html")

@app.route("/heart.html")
def heart():
    return render_template("heart.html")

@app.route("/lung.html")
def lung():
    return render_template("lung.html")

@app.route("/parkinson.html")
def parkinson():
    return render_template("parkinson.html")

@app.route("/kidney.html")
def kidney():
    return render_template("kidney.html")



@app.route("/kidney", methods=["POST"])
def result_kidney():
    # Check if the request contains a file
    if 'ct-scan' not in request.files:
        return "No file uploaded", 400

    file = request.files['ct-scan']

    # Validate that a file is selected
    if file.filename == '':
        return "No file selected", 400

    # Save the file securely
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    model = load_model("my_tf_model.keras")
    pic = []
    img = cv2.imread(str(filepath))
    img = cv2.resize(img, (28, 28))
    pic.append(img)
    pic1 = np.array(pic)
    a = model.predict(pic1)
    if a.argmax() == 0:
        d="Cyst"
    elif a.argmax() == 1:
        d="Normal"
        return render_template("result.html", file_url=filepath, result=f"The kidney is {d}.")
    elif a.argmax() == 2:
        d="Stone"
    else:
        d= "Tumor"

    # Get the user's location (latitude, longitude)
    location = get_user_location()

    if location:
        lat, lon = location
        nearest_hospital = get_nearest_hospital(lat, lon)
        if nearest_hospital:
            hospital_info = f"The nearest hospital is {nearest_hospital['name']}. Location: {nearest_hospital['address']}"
        else:
            hospital_info = "No nearby hospitals found."
    else:
        hospital_info = "Unable to determine your location."

    return render_template("result.html", file_url=filepath, result=f"The detected disease is {d}. {hospital_info}")


@app.route("/lung", methods=["POST"])
def result_lung():
    # Check if the request contains a file
    if "chest-xray" not in request.files:
        return "No file uploaded", 400

    file = request.files["chest-xray"]

    # Validate that a file is selected
    if file.filename == '':
        return "No file selected", 400

    # Save the file securely
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Load the saved model
    model = tf.keras.models.load_model('my_model.keras')

    # Define image size and class names as per your model setup
    img_size = (224, 224)  # Adjust if needed, based on your model's input size
    classes = ['Lung squamous_cell_carcinoma', 'Lung_adenocarcinoma',
               'Lung_benign_tissue']  # Adjust class names accordingly

    # Load and preprocess the image
    img = image.load_img(filepath, target_size=img_size)  # Resize image to match model input
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Normalize the image (if your model was trained with normalization)
    img_array = img_array / 255.0  # Normalize if model was trained with this step

    # Predict the class of the image
    predictions = model.predict(img_array)

    # Get the predicted class index and corresponding label
    predicted_class_index = np.argmax(predictions, axis=1)
    predicted_class = classes[predicted_class_index[0]]

    return render_template("result.html", file_url=filepath, result=f'The image is classified as: {predicted_class}')



@app.route("/heart", methods=["POST"])
def result_heart():

    form_data = request.form.to_dict()
    values_list = [float(value) for value in form_data.values()]
    data = np.array(values_list)
    data = data.reshape(-1, 1)
    clf = joblib.load(r"C:\Users\PC\OneDrive\Desktop\new_\naiveBayesModel.pkl")
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    input_data = pd.DataFrame(data.T, columns=columns)
    preds = clf.predict(input_data)
    if preds[0] == 1:
        return render_template("result.html", form_data=form_data, result="You have issues related to your cardiovascular system. Contact the nearest hospital as soon as possible.")
    else:
        return render_template("result.html", form_data=form_data, result="Your cardiovascular system is healthy.")


@app.route("/parkinson", methods=["POST"])
def result_parkinson():
    form_data = request.form.to_dict()
    values_list = [float(value) for value in form_data.values()]
    values_array = np.array(values_list)
    return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True)
