import joblib
import sys

def load_model():
    model = joblib.load('spam_detector_model.pkl')
    vectorizer = joblib.load('spam_detector_vectorizer.pkl')
    print("Model and vectorizer loaded.")  # Debug statement
    return model, vectorizer

def predict(text):
    model, vectorizer = load_model()
    print(f"Input text: {text}")  # Debug statement
    text_transformed = vectorizer.transform([text])
    print(f"Transformed text: {text_transformed}")  # Debug statement
    prediction = model.predict(text_transformed)  # Fixed method call
    return "Spam" if prediction[0] == 1 else "Not Spam"

if __name__ == "__main__":
    text = sys.argv[1]
    prediction = predict(text)
    print(f"The Email is: {prediction}")
