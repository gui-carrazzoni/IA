#!/usr/bin/env python3
"""
Script unificado para testar predições com KNN, Gradient Boosting, ELM ou Genético
Permite escolher o modelo de forma interativa
"""
from predicoes_simples_knn import testar_predicoes_knn
from testar_predicoes_gb import testar_predicoes_gb
from testar_predicoes_elm import testar_predicoes_elm
from testar_predicoes_mlp import testar_predicoes_mlp
from testar_predicoes_genetico import testar_predicoes_genetico
import pickle
import numpy as np


def testar_predicoes_genetico():
    print("=== PREDIÇÕES GENÉTICAS OTIMIZADO (RF) ===")
    
    # Carrega modelo Random Forest otimizado pelo genético
    with open('data/processado/modelo_genetico_otimizado.pkl', 'rb') as f:
        modelo_info = pickle.load(f)
    
    modelo = modelo_info['modelo']
    features_mask = modelo_info['features_mask']
    feature_names = modelo_info['feature_names']
    features_selecionadas = modelo_info['features_selecionadas']
    hiperparametros = modelo_info['hiperparametros']
    
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
        # Caso seja numpy array
        X_test_sel = X_test[:, features_mask]
    
    print(f"Dados de teste: {X_test_sel.shape[0]} amostras, {X_test_sel.shape[1]} features selecionadas")
    print(f"\n🔧 Hiperparâmetros otimizados pelo genético:")
    for param, value in hiperparametros.items():
        print(f"   {param}: {value}")
    
    print(f"\n📊 Features selecionadas ({len(features_selecionadas)}/{len(feature_names)}):")
    for i, feature in enumerate(features_selecionadas):
        print(f"   {i+1}. {feature}")
    
    print(f"\n{'Linha':<6} {'Real':<6} {'Predita':<8} {'Acerto':<8} {'Features'}")
    print("=" * 100)
    
    # Fazer predições
    y_pred = modelo.predict(X_test_sel)
    y_pred_proba = modelo.predict_proba(X_test_sel)
    
    acertos = 0
    erros = []
    for i in range(len(X_test_sel)):
        real = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
        predita = y_pred[i]
        acertou = real == predita
        if acertou:
            acertos += 1
        else:
            erros.append((idx_test[i], real, predita))
        linha_idx = idx_test[i]
        # Mostra apenas as features selecionadas pelo genético
        features_str = "[" + ", ".join([f"{x:.1f}" for x in X_test_sel[i]]) + "]"
        print(f"{linha_idx:<6} {real:<6} {predita:<8} {'✓' if acertou else '✗':<8} {features_str}")
    print("=" * 100)
    print(f"Acurácia: {acertos}/{len(X_test_sel)} ({acertos/len(X_test_sel)*100:.1f}%)")
    
    # Mostrar informações sobre o modelo otimizado
    print(f"\n🧬 Informações do Modelo Otimizado:")
    print(f"   Algoritmo: Random Forest")
    print(f"   Features originais: {len(feature_names)}")
    print(f"   Features selecionadas: {len(features_selecionadas)}")
    print(f"   Redução: {((len(feature_names) - len(features_selecionadas)) / len(feature_names) * 100):.1f}%")
    print(f"   Performance salva: {modelo_info['performance']['acuracia']:.4f} acurácia")
    
    if erros:
        print(f"\n{'='*50}")
        print("LINHAS PREDITAS INCORRETAMENTE:")
        print(f"{'='*50}")
        for linha, real, predita in erros:
            print(f"Linha {linha}: Real = {real}, Predita = {predita}")
        print(f"{'='*50}")
    else:
        print(f"\n{'='*50}")
        print("TODAS AS PREDIÇÕES ESTÃO CORRETAS! 🎉")
        print(f"{'='*50}")


def main():
    print("🤖 TESTADOR DE PREDIÇÕES - KNN vs Gradient Boosting vs ELM vs MLP vs Genético")
    print("=" * 50)
    print("Escolha o modelo:")
    print("1. KNN (K-Nearest Neighbors)")
    print("2. GB (Gradient Boosting)")
    print("3. ELM (Extreme Learning Machine)")
    print("4. MLP (Multi-Layer Perceptron)")
    print("5. Genético (Random Forest otimizado)")
    print("6. Predições do Genético (detalhado)")
    print("7. Sair")
    
    while True:
        try:
            escolha = input("\nDigite sua escolha (1, 2, 3, 4, 5, 6 ou 7): ").strip()
            
            if escolha == '1':
                testar_predicoes_knn()
                break
            elif escolha == '2':
                testar_predicoes_gb()
                break
            elif escolha == '3':
                testar_predicoes_elm()
                break
            elif escolha == '4':
                testar_predicoes_mlp()
                break
            elif escolha == '5':
                testar_predicoes_genetico()
                break
            elif escolha == '6':
                testar_predicoes_genetico()
                break
            elif escolha == '7':
                print("Saindo...")
                break
            else:
                print("❌ Escolha inválida! Digite 1, 2, 3, 4, 5, 6 ou 7.")
        except KeyboardInterrupt:
            print("\n\nSaindo...")
            break
        except Exception as e:
            print(f"❌ Erro: {e}")

if __name__ == "__main__":
    main()
