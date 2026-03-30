from data_loader import load_medhallu
from preprocess import clean_text
from feature_extraction import extract_features
from model import train_model
from evaluate import evaluate_model
import joblib

from sklearn.model_selection import train_test_split

df = load_medhallu()

df['text'] = df['text'].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

X_train_vec, X_test_vec, vectorizer = extract_features(X_train, X_test)

model = train_model(X_train_vec, y_train)
joblib.dump(model, "hallucination_model.pkl")

evaluate_model(model, X_test_vec, y_test)