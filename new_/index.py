from flask import Flask, render_template, request
import os
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__)

# Configure the upload folder
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the upload folder exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
    return render_template("result.html")

@app.route("/lung",methods=["POST"])
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
    return render_template("home.html")





@app.route("/heart", methods=["POST"])
def result_heart():

    form_data = request.form.to_dict()
    values_list = [float(value) for value in form_data.values()]
    data = np.array(values_list)
    data =data.reshape(-1,1)
    clf = joblib.load(r"C:\Users\PC\OneDrive\Desktop\new_\naiveBayesModel.pkl")
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',
               'thal']
    input_data = pd.DataFrame(data.T, columns=columns)
    preds = clf.predict(input_data)
    if preds[0]==1:
        return render_template("result.html",form_data=form_data,result="You have issues related to your cardiovascular system.")
    else:
        return render_template("result.html", form_data=form_data,
                               result="Your cardiovascular system if healthy.")
@app.route("/parkinson",methods=["POST"])
def result_parkinson():

    form_data = request.form.to_dict()
    values_list = [float(value) for value in form_data.values()]
    values_array = np.array(values_list)
    return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True)
