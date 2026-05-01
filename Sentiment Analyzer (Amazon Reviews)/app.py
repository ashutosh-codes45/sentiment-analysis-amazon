from flask import Flask, render_template, request 
import os 
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stops = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# function to cleaning (preprocessing) of the Review Text 

def clean(text) : 
    text = text.lower()
    text=re.sub('[^a-z ]','',text)
    words = [w for w in text.split() if w not in stops]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

if os.path.exists('models/logistic_model.pkl') :
    model = joblib.load('models/logistic_model.pkl')
else :
    raise Exception("Model not found. Run train_model.py first")

if os.path.exists('models/tfidf_vectorizer.pkl'):
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    print(type(vectorizer))
else:
    raise Exception("tfidf_vectorizer.pkl not found")

app = Flask(__name__, template_folder='templates')

@app.route('/') 
def index() :
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict() :
    review_txt = request.form.get('review')
    review_txt = clean(review_txt)
    feature = vectorizer.transform([review_txt])

    prediction = model.predict(feature)[0]
    result = "Positive 😊" if prediction==1 else "Negative 😞"

    return render_template('index.html', prediction=result)

if __name__ == "__main__" :
  app.run(debug=True, use_reloader=False)
