from preprocess import tf_idf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
import joblib 

X, y, tfidf = tf_idf()

def logistic_model() :
  # Model Training (Logistic Model for Predicting -> +ve or Negative Review) :-
  X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

  model_logisticRegression = LogisticRegression()
  model_logisticRegression.fit(X_train, y_train)
  y_pred = model_logisticRegression.predict(X_test)
  print(y_test[:20].values)
  print(y_pred[:20])

  # Model Evaluation :-

  print(accuracy_score(y_test, y_pred))
  print(confusion_matrix(y_test, y_pred))
  print(classification_report(y_test, y_pred))

  joblib.dump(model_logisticRegression, 'models/logistic_model.pkl')
  joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')

  return model_logisticRegression

logistic_model()



