import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class KNN:
    """
    Implementação do algoritmo K-Nearest Neighbors (KNN)
    """
    
    def __init__(self, k=3, distance_metric='euclidean'):
        """
        Inicializa o classificador KNN
        
        Parâmetros:
        k (int): Número de vizinhos mais próximos
        distance_metric (str): Métrica de distância ('euclidean', 'manhattan', 'minkowski')
        """
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y):
        """
        Treina o modelo KNN (apenas armazena os dados de treinamento)
        
        Parâmetros:
        X (array): Features de treinamento
        y (array): Labels de treinamento
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
    
    def calculate_distance(self, x1, x2):
        """
        Calcula a distância entre dois pontos
        
        Parâmetros:
        x1, x2 (array): Pontos para calcular a distância
        
        Retorna:
        float: Distância calculada
        """
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.distance_metric == 'minkowski':
            return np.sum(np.abs(x1 - x2) ** 3) ** (1/3)
        else:
            raise ValueError(f"Métrica de distância '{self.distance_metric}' não suportada")
    
    def predict_single(self, x):
        """
        Faz predição para um único ponto
        
        Parâmetros:
        x (array): Ponto para predição
        
        Retorna:
        int: Classe predita
        """
        # Calcula distâncias para todos os pontos de treinamento
        distances = []
        for i in range(len(self.X_train)):
            dist = self.calculate_distance(x, self.X_train[i])
            distances.append((dist, self.y_train[i]))
        
        # Ordena por distância e pega os k vizinhos mais próximos
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:self.k]
        
        # Extrai as classes dos k vizinhos mais próximos
        k_nearest_labels = [label for _, label in k_nearest]
        
        # Retorna a classe mais frequente (voto majoritário)
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
    def predict(self, X):
        """
        Faz predições para múltiplos pontos
        
        Parâmetros:
        X (array): Features para predição
        
        Retorna:
        array: Classes preditas
        """
        X = np.array(X)
        predictions = []
        
        for x in X:
            pred = self.predict_single(x)
            predictions.append(pred)
            
        return np.array(predictions)
    
    def score(self, X, y):
        """
        Calcula a acurácia do modelo
        
        Parâmetros:
        X (array): Features de teste
        y (array): Labels verdadeiros
        
        Retorna:
        float: Acurácia do modelo
        """
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

def carregar_dados_wine():
    """
    Carrega e prepara o dataset de vinhos
    """
    try:
        # Tenta carregar o dataset
        df = pd.read_csv('data/Wine dataset.csv')
        
        # Separa features e target
        X = df.drop('class', axis=1).values
        y = df['class'].values
        
        print(f"Dataset carregado com sucesso!")
        print(f"Shape dos dados: {X.shape}")
        print(f"Número de classes: {len(np.unique(y))}")
        
        return X, y
        
    except FileNotFoundError:
        print("Arquivo 'data/Wine dataset.csv' não encontrado!")
        print("Criando dados sintéticos para demonstração...")
        
        # Cria dados sintéticos para demonstração
        np.random.seed(42)
        n_samples = 200
        n_features = 13
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 3, n_samples)
        
        return X, y

def avaliar_modelo(y_true, y_pred, classes=None):
    """
    Avalia o modelo e mostra métricas
    """
    print("\n=== AVALIAÇÃO DO MODELO ===")
    print(f"Acurácia: {accuracy_score(y_true, y_pred):.4f}")
    
    print("\nRelatório de Classificação:")
    print(classification_report(y_true, y_pred, target_names=classes))
    
    # Matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Matriz de Confusão')
    plt.ylabel('Valor Real')
    plt.xlabel('Valor Predito')
    plt.tight_layout()
    plt.savefig('Results/matriz_confusao_knn.png', dpi=300, bbox_inches='tight')
    plt.show()

def encontrar_melhor_k(X_train, X_val, y_train, y_val, k_range=range(1, 21)):
    """
    Encontra o melhor valor de k usando validação
    """
    print("\n=== ENCONTRANDO O MELHOR K ===")
    accuracies = []
    
    for k in k_range:
        knn = KNN(k=k)
        knn.fit(X_train, y_train)
        accuracy = knn.score(X_val, y_val)
        accuracies.append(accuracy)
        print(f"k={k:2d}: Acurácia = {accuracy:.4f}")
    
    best_k = k_range[np.argmax(accuracies)]
    best_accuracy = max(accuracies)
    
    print(f"\nMelhor k: {best_k} com acurácia: {best_accuracy:.4f}")
    
    # Plot do gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, accuracies, 'bo-', linewidth=2, markersize=8)
    plt.axvline(x=best_k, color='red', linestyle='--', label=f'Melhor k = {best_k}')
    plt.xlabel('Valor de k')
    plt.ylabel('Acurácia')
    plt.title('Acurácia vs Valor de k')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Results/melhor_k_knn.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return best_k

def main():
    """
    Função principal para demonstrar o uso do KNN
    """
    print("=== IMPLEMENTAÇÃO DO ALGORITMO KNN ===")
    
    # Carrega os dados
    X, y = carregar_dados_wine()
    
    # Divide os dados em treino, validação e teste
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    
    print(f"\nDivisão dos dados:")
    print(f"Treino: {X_train.shape[0]} amostras")
    print(f"Validação: {X_val.shape[0]} amostras")
    print(f"Teste: {X_test.shape[0]} amostras")
    
    # Normaliza os dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Encontra o melhor k
    best_k = encontrar_melhor_k(X_train_scaled, X_val_scaled, y_train, y_val)
    
    # Treina o modelo final com o melhor k
    print(f"\n=== TREINANDO MODELO FINAL (k={best_k}) ===")
    knn_final = KNN(k=best_k)
    knn_final.fit(X_train_scaled, y_train)
    
    # Faz predições no conjunto de teste
    y_pred = knn_final.predict(X_test_scaled)
    
    # Avalia o modelo
    classes = [f'Classe {i}' for i in np.unique(y)]
    avaliar_modelo(y_test, y_pred, classes)
    
    # Demonstra predições individuais
    print("\n=== EXEMPLOS DE PREDIÇÕES ===")
    for i in range(min(5, len(X_test))):
        pred = knn_final.predict_single(X_test_scaled[i])
        print(f"Amostra {i+1}: Real = {y_test[i]}, Predito = {pred}")
    
    print(f"\n=== RESUMO ===")
    print(f"Algoritmo: K-Nearest Neighbors")
    print(f"Melhor k encontrado: {best_k}")
    print(f"Acurácia final: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Resultados salvos em: Results/")

if __name__ == "__main__":
    main()
