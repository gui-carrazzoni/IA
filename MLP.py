from sklearn.neural_network import MLPClassifier as SklearnMLPClassifier
import numpy as np

class MLPClassifier:
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', random_state=42, max_iter=300):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.random_state = random_state
        self.max_iter = max_iter
        self.model = SklearnMLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            random_state=self.random_state,
            max_iter=self.max_iter
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
