from flask import Flask, render_template, request
from model import modelTrain, predictor

app = Flask(__name__)

model, selected_columns, gender_encoder = modelTrain()

@app.route("/", methods=["GET", "POST"])
def home():
    ans = None
    if request.method == "POST":
        try:
            input_data = [
                int(request.form['age']),
                request.form['gender'],
                float(request.form['weight']),
                float(request.form['hemoglobin']),
                int(request.form['num_donations']),
                int(request.form['months_last'])
            ]
            result = predictor(input_data, model, selected_columns, gender_encoder)
            ans = "Eligible to Donate Blood ✅" if result == 1 else "Not Eligible to Donate Blood ❌"
        except Exception as e:
            ans = f"Error: {e}"
    return render_template("index.html", ans=ans)

if __name__ == "__main__":
    app.run(debug=True)