from data_loader import load_medhallu
from preprocess import clean_text
from feature_extraction import extract_features
from model import train_models

from sklearn.model_selection import train_test_split

def run_training():
    df = load_medhallu()

    print("Dataset size:", len(df))

    df['text'] = df['text'].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )

    X_train_vec, X_test_vec, vectorizer = extract_features(X_train, X_test)

    models = train_models(X_train_vec, y_train)

    return models, X_test_vec, y_test