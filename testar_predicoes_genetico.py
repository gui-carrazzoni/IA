import pickle
import numpy as np
import os
from algoritmo_genetico import AlgoritmoGenetico, avaliar_melhor_solucao

def testar_predicoes_genetico():
    print("Predição com modelo genético otimizado (Random Forest)")
    modelo_path = 'data/processado/modelo_genetico_otimizado.pkl'
    if not os.path.exists(modelo_path):
        print(f"Modelo otimizado não encontrado. Treinando agora...")
        with open('data/processado/dados_processados.pkl', 'rb') as f:
            dados = pickle.load(f)
        if 'X_train_balanced' in dados:
            dados_para_ga = dados
        else:
            dados_para_ga = dados['dados_processados']
            dados_para_ga = {
                'X_train_balanced': dados_para_ga['X_train'],
                'y_train_balanced': dados_para_ga['y_train'],
                'X_test': dados_para_ga['X_test'],
                'y_test': dados_para_ga['y_test'],
                'feature_names': dados_para_ga['feature_names']
            }
        ga = AlgoritmoGenetico(
            dados=dados_para_ga,
            tamanho_populacao=15,
            taxa_mutacao=0.15,
            taxa_crossover=0.8,
            max_geracoes=10,
            elitismo=0.2
        )
        melhor_solucao = ga.executar()
        ga.visualizar_evolucao()
        modelo_info = avaliar_melhor_solucao(melhor_solucao, dados_para_ga)
        print("Modelo otimizado treinado e salvo.")
    else:
        with open(modelo_path, 'rb') as f:
            modelo_info = pickle.load(f)
        print("Modelo otimizado carregado.")
    modelo = modelo_info['modelo']
    features_mask = modelo_info['features_mask']
    features_selecionadas = modelo_info['features_selecionadas']
    # Carrega dados de teste
    with open('data/processado/dados_processados.pkl', 'rb') as f:
        dados = pickle.load(f)
    X_test = dados['X_test']
    y_test = dados['y_test']
    idx_test = X_test.index.values if hasattr(X_test, 'index') else np.arange(len(X_test))
    # Seleciona features otimizadas pelo genético
    if hasattr(X_test, 'loc'):
        X_test_sel = X_test.loc[:, features_selecionadas].values
    else:
        X_test_sel = X_test[:, features_mask]
    y_pred = modelo.predict(X_test_sel)
    acertos = np.sum(y_pred == (y_test.values if hasattr(y_test, 'values') else y_test))
    print(f"Acurácia: {acertos}/{len(X_test_sel)} ({acertos/len(X_test_sel)*100:.1f}%)")
    erros = []
    for i in range(len(X_test_sel)):
        real = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
        predita = y_pred[i]
        if real != predita:
            erros.append((idx_test[i], real, predita))
    if erros:
        print("Erros de predição:")
        for linha, real, predita in erros:
            print(f"Linha {linha}: Real = {real}, Predita = {predita}")
    else:
        print("Todas as predições estão corretas.")


def main():
    testar_predicoes_genetico()

if __name__ == "__main__":
    main() 