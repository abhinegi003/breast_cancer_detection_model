from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("cancer_model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""

    if request.method == "POST":
        values = [float(x) for x in request.form.values()]
        final = np.array(values).reshape(1, -1)
        result = model.predict(final)

        prediction = "Malignant" if result[0] == 1 else "Benign"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
