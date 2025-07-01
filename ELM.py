import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

class ELMClassifier:
    def __init__(self, hidden_size=100, activation='sigmoid', random_state=42):
        self.hidden_size = hidden_size
        self.activation = activation
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.W = None
        self.b = None
        self.beta = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        classes = np.unique(y)
        self.classes_ = classes
        y_onehot = np.zeros((len(y), len(classes)))
        for i, c in enumerate(classes):
            y_onehot[y == c, i] = 1
        np.random.seed(self.random_state)
        self.W = np.random.randn(X_scaled.shape[1], self.hidden_size) * 0.1
        self.b = np.random.randn(self.hidden_size) * 0.1
        H = X_scaled @ self.W + self.b
        if self.activation == 'sigmoid':
            H = self.sigmoid(H)
        elif self.activation == 'relu':
            H = self.relu(H)
        self.beta = np.linalg.pinv(H) @ y_onehot
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        H = X_scaled @ self.W + self.b
        if self.activation == 'sigmoid':
            H = self.sigmoid(H)
        elif self.activation == 'relu':
            H = self.relu(H)
        output = H @ self.beta
        return self.classes_[np.argmax(output, axis=1)]

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        H = X_scaled @ self.W + self.b
        if self.activation == 'sigmoid':
            H = self.sigmoid(H)
        elif self.activation == 'relu':
            H = self.relu(H)
        output = H @ self.beta
        exp_output = np.exp(output - np.max(output, axis=1, keepdims=True))
        return exp_output / np.sum(exp_output, axis=1, keepdims=True)

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
        
    def get_params(self, deep=True):
        return {
            'hidden_size': self.hidden_size,
            'activation': self.activation,
            'random_state': self.random_state
        }
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

def train_elm_model(X_train, y_train, X_test, y_test):
    modelo = ELMClassifier(hidden_size=100, activation='sigmoid', random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(modelo, X_train, y_train, cv=3)
    relatorio = classification_report(y_test, y_pred, output_dict=True)
    return {
        'modelo': modelo,
        'acuracia': acc,
        'cv_media': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'precision_macro': relatorio['macro avg']['precision'],
        'recall_macro': relatorio['macro avg']['recall'],
        'f1_macro': relatorio['macro avg']['f1-score'],
        'y_pred': y_pred
    }