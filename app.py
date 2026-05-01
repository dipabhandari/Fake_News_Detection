from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']

    if not news.strip():
        return render_template("index.html", result="Please enter text")

    data = vectorizer.transform([news])

    prediction = model.predict(data)[0]

    prob = model.predict_proba(data)[0]

    # Check class order
    # print(model.classes_)

    fake_prob = round(prob[0] * 100, 2)
    real_prob = round(prob[1] * 100, 2)

    return render_template(
        "index.html",
        result=prediction,
        real_prob=real_prob,
        fake_prob=fake_prob
    )

if __name__ == "__main__":
    app.run(debug=True)