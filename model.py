from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

def train_models(X_train, y_train):

    models = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    models["Logistic Regression"] = lr

    # Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    models["Naive Bayes"] = nb

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf

    return models