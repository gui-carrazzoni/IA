import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import warnings
import os
warnings.filterwarnings('ignore')

def main():
    print("PRÉ-PROCESSAMENTO DO DATASET DE VINHO")
    print("=" * 50)
    
    # 1. Carregar dados
    print("\nCarregando dados...")
    dados = pd.read_csv('data/Wine dataset.csv')
    print(f"Dados carregados: {dados.shape[0]} amostras, {dados.shape[1]} variáveis")
    
    # 2. Análise inicial
    print(f"\nColunas: {list(dados.columns)}")
    print(f"Valores nulos: {dados.isnull().sum().sum()}")
    
    # 3. Análise da variável alvo
    print("\nAnálise da variável alvo:")
    contagem_classes = dados['class'].value_counts()
    print(contagem_classes)
    
    # Verificar desbalanceamento
    total = len(dados)
    proporcoes = (contagem_classes / total * 100).round(2)
    print(f"\nProporção de cada classe:")
    for classe, prop in proporcoes.items():
        print(f"   Classe {classe}: {prop}%")
    
    razao_desbalanceamento = proporcoes.max() / proporcoes.min()
    print(f"\nRazão de desbalanceamento: {razao_desbalanceamento:.2f}")
    
    if razao_desbalanceamento > 1.5:
        print("Dataset apresenta desbalanceamento!")
    else:
        print("Dataset relativamente balanceado!")
    
    # 4. Pré-processamento
    print("\nPré-processamento:")
    
    # Separar features e target
    X = dados.drop('class', axis=1)
    y = dados['class']
    
    # Tratar outliers com clipping
    print("   Tratando outliers...")
    for coluna in X.columns:
        Q1 = X[coluna].quantile(0.25)
        Q3 = X[coluna].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        X[coluna] = X[coluna].clip(limite_inferior, limite_superior)
    
    # Normalização
    print("   Aplicando normalização...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Dividir em treino e teste
    print("   Dividindo em treino e teste...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Treino: {X_train.shape[0]} amostras")
    print(f"   Teste: {X_test.shape[0]} amostras")
    
    # 5. Aplicar SMOTE para balanceamento
    print("\nAplicando SMOTE para balanceamento...")
    
    # Verificar distribuição antes
    print(f"   Distribuição antes do SMOTE:")
    contagem_antes = pd.Series(y_train).value_counts()
    for classe, contagem in contagem_antes.items():
        print(f"      Classe {classe}: {contagem} amostras")
    
    # Aplicar SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Verificar distribuição depois
    print(f"   Distribuição após SMOTE:")
    contagem_depois = pd.Series(y_train_balanced).value_counts()
    for classe, contagem in contagem_depois.items():
        print(f"      Classe {classe}: {contagem} amostras")
    
    # 6. Testar modelo com dados originais vs balanceados
    print("\nTestando modelos...")
    
    # Modelo com dados originais
    print("   Testando com dados originais:")
    modelo_original = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo_original.fit(X_train, y_train)
    y_pred_original = modelo_original.predict(X_test)
    acuracia_original = accuracy_score(y_test, y_pred_original)
    print(f"      Acurácia: {acuracia_original:.4f}")
    
    # Modelo com dados balanceados
    print("   Testando com dados balanceados (SMOTE):")
    modelo_balanced = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo_balanced.fit(X_train_balanced, y_train_balanced)
    y_pred_balanced = modelo_balanced.predict(X_test)
    acuracia_balanced = accuracy_score(y_test, y_pred_balanced)
    print(f"      Acurácia: {acuracia_balanced:.4f}")
    
    # 7. Relatório detalhado
    print("\nRelatório de classificação (dados balanceados):")
    print(classification_report(y_test, y_pred_balanced))
    
    # 8. Salvar dados processados
    print("\nSalvando dados processados...")
    
    dados_processados = {
        'X_train_original': X_train,
        'X_train_balanced': X_train_balanced,
        'X_test': X_test,
        'y_train_original': y_train,
        'y_train_balanced': y_train_balanced,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': X.columns.tolist()
    }
    
    import pickle
    os.makedirs('data/processado', exist_ok=True)
    with open('data/processado/dados_processados.pkl', 'wb') as f:
        pickle.dump(dados_processados, f)
    
    print("Dados salvos em 'data/processado/dados_processados.pkl'")
    
    # 9. Criar visualização
    print("\nCriando visualização...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Distribuição antes e depois do SMOTE
    axes[0].bar(['Classe 1', 'Classe 2', 'Classe 3'], contagem_antes.values, 
                color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.7, label='Antes')
    axes[0].set_title('Distribuição das Classes - Antes do SMOTE', fontweight='bold')
    axes[0].set_ylabel('Número de Amostras')
    
    axes[1].bar(['Classe 1', 'Classe 2', 'Classe 3'], contagem_depois.values,
                color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.7, label='Depois')
    axes[1].set_title('Distribuição das Classes - Após SMOTE', fontweight='bold')
    axes[1].set_ylabel('Número de Amostras')
    
    plt.tight_layout()
    plt.savefig('balanceamento_smote.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualização salva como 'balanceamento_smote.png'")
    
    # 10. Resumo final
    print("Dados carregados e analisados")
    print("Outliers tratados")
    print("Normalização aplicada")
    print("SMOTE aplicado para balanceamento")

    
    return dados_processados

if __name__ == "__main__":
    dados_processados = main() 