import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split

def main():
    print("\nGRADIENT BOOSTING")
    with open('data/processado/dados_processados.pkl', 'rb') as f:
        dados = pickle.load(f)
    X_train = dados['X_train_balanced']
    y_train = dados['y_train_balanced']
    X_test = dados['X_test']
    y_test = dados['y_test']
    scaler = dados['scaler']
    feature_names = dados['feature_names']

    modelo = GradientBoostingClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {acc:.4f}\n")
    print(classification_report(y_test, y_pred))

def train_gb_model(X_train, y_train, X_test, y_test):
    """
    Treina um Gradient Boosting com busca de hiperparâmetros
    """
    print("   🔍 Buscando melhor configuração do Gradient Boosting...")
    
    # Dividir dados de treino em treino e validação para busca de hiperparâmetros
    X_train_search, X_val, y_train_search, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Parâmetros para busca
    n_estimators_list = [50, 100, 150, 200]
    learning_rates = [0.01, 0.1, 0.2]
    max_depths = [3, 5, 7, None]
    
    best_accuracy = 0.0
    best_model = None
    best_params = {}
    
    # Grid search para encontrar melhor configuração
    for n_est in n_estimators_list:
        for lr in learning_rates:
            for md in max_depths:
                modelo = GradientBoostingClassifier(
                    n_estimators=n_est,
                    learning_rate=lr,
                    max_depth=md,
                    random_state=42
                )
                
                # Treinar modelo
                modelo.fit(X_train_search, y_train_search)
                
                # Avaliar na validação
                y_val_pred = modelo.predict(X_val)
                acc_val = accuracy_score(y_val, y_val_pred)
                
                # Atualizar melhor modelo
                if acc_val > best_accuracy:
                    best_accuracy = acc_val
                    best_model = modelo
                    best_params = {
                        'n_estimators': n_est,
                        'learning_rate': lr,
                        'max_depth': md
                    }
    
    print(f"   Melhor configuração: {best_params['n_estimators']} estimadores, "
          f"lr={best_params['learning_rate']}, max_depth={best_params['max_depth']}")
    
    # Treinar melhor modelo com todos os dados de treino
    best_model.fit(X_train, y_train)
    
    # Fazer predições no conjunto de teste
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)
    
    # Calcular métricas
    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=3)
    relatorio = classification_report(y_test, y_pred, output_dict=True)
    
    # Visualizações
    visualizar_predicoes_gb(y_test, y_pred, y_pred_proba, best_params, X_train, best_model)
    
    return {
        'modelo': best_model,
        'acuracia': acc,
        'cv_media': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'precision_macro': relatorio['macro avg']['precision'],
        'recall_macro': relatorio['macro avg']['recall'],
        'f1_macro': relatorio['macro avg']['f1-score'],
        'y_pred': y_pred,
        'melhor_params': best_params
    }

def visualizar_predicoes_gb(y_test, y_pred, y_pred_proba, best_params, X_train, modelo):
    """
    Cria visualizações das predições do Gradient Boosting
    """
    print("   Criando visualizações das predições...")
    
    # 1. Matriz de Confusão
    plt.figure(figsize=(16, 4))
    
    plt.subplot(1, 4, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Classe 1', 'Classe 2', 'Classe 3'],
                yticklabels=['Classe 1', 'Classe 2', 'Classe 3'])
    plt.title('Matriz de Confusão - Gradient Boosting')
    plt.ylabel('Valor Real')
    plt.xlabel('Valor Predito')
    
    # 2. Probabilidades por Classe
    plt.subplot(1, 4, 2)
    classes = ['Classe 1', 'Classe 2', 'Classe 3']
    prob_means = np.mean(y_pred_proba, axis=0)
    prob_stds = np.std(y_pred_proba, axis=0)
    
    bars = plt.bar(classes, prob_means, yerr=prob_stds, capsize=5, 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.7)
    plt.title('Probabilidades Médias por Classe')
    plt.ylabel('Probabilidade')
    plt.ylim(0, 1)
    
    # Adicionar valores nas barras
    for bar, mean, std in zip(bars, prob_means, prob_stds):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{mean:.3f}\n±{std:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Importância das Features
    plt.subplot(1, 4, 3)
    feature_importance = modelo.feature_importances_
    feature_names = X_train.columns
    
    # Ordenar features por importância
    indices = np.argsort(feature_importance)[::-1]
    top_features = 10  # Mostrar top 10 features
    
    plt.barh(range(top_features), feature_importance[indices[:top_features]], 
             color='#4ECDC4', alpha=0.7)
    plt.yticks(range(top_features), [feature_names[i] for i in indices[:top_features]])
    plt.title('Top 10 Features Mais Importantes')
    plt.xlabel('Importância')
    
    # 4. Distribuição de Erros
    plt.subplot(1, 4, 4)
    erros = y_test != y_pred
    acertos = y_test == y_pred
    
    plt.pie([sum(acertos), sum(erros)], labels=['Acertos', 'Erros'], 
            autopct='%1.1f%%', colors=['#4ECDC4', '#FF6B6B'])
    plt.title('Distribuição de Acertos vs Erros')
    
    plt.tight_layout()
    plt.savefig('gb_predicoes.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("   ✅ Visualizações salvas como 'gb_predicoes.png'")
    
    # 5. Análise detalhada
    print(f"   📋 Análise Detalhada:")
    print(f"      Melhor configuração: {best_params}")
    print(f"      Acurácia: {accuracy_score(y_test, y_pred):.4f}")
    print(f"      Total de erros: {sum(erros)}")
    print(f"      Taxa de erro: {sum(erros)/len(y_test)*100:.2f}%")
    
    # Mostrar top 5 features mais importantes
    print(f"      Top 5 features mais importantes:")
    for i in range(5):
        print(f"         {feature_names[indices[i]]}: {feature_importance[indices[i]]:.4f}")
    
    # Mostrar alguns exemplos de erros
    if sum(erros) > 0:
        indices_erros = np.where(erros)[0]
        print(f"      Exemplos de erros (primeiros 3):")
        for i in indices_erros[:3]:
            print(f"         Real: {y_test.iloc[i]}, Predito: {y_pred[i]}, "
                  f"Confiança: {max(y_pred_proba[i]):.3f}")

if __name__ == "__main__":
    main() 