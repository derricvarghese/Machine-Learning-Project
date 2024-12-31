import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


def load_data():
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    data = pd.read_csv(url, sep='\t', header=None, names=['label', 'text'])
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})
    return data

def train_model(data):
    x = data['text']
    y = data['label']
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = MultinomialNB()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return accuracy, report


if __name__ == "__main__":
    data = load_data()
    print(data)
