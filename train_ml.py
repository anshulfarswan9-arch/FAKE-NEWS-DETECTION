import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from src.preprocess import load_and_prepare

def train_and_evaluate(path='data/fake_and_real_news.csv'):
    df = load_and_prepare(path)
    X, y = df['clean'], df['label_num']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    vectorizer = TfidfVectorizer(max_df=0.8, min_df=5, ngram_range=(1,2))
    models = {
        'logreg': LogisticRegression(max_iter=1000),
        'nb': MultinomialNB(),
        'svm': LinearSVC(),
        'rf': RandomForestClassifier(n_estimators=200, random_state=42)
    }
    for name, clf in models.items():
        print(f"Training {name}...")
        pipe = Pipeline([('tfidf', vectorizer), ('clf', clf)])
        pipe.fit(Xtr, ytr)
        preds = pipe.predict(Xte)
        acc = accuracy_score(yte, preds)
        print(f"{name} Accuracy: {acc:.4f}")
        print(classification_report(yte, preds))
        joblib.dump(pipe, f'model_{name}.joblib')
    print("✅ Models saved successfully.")

if __name__ == '__main__':
    train_and_evaluate()
