#!/usr/bin/env python3
"""
Script para testar predições com Gradient Boosting
Mostra: índice da linha original, classe real, classe predita, acerto e features
Destaca erros no final
"""
import pickle
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import os

def testar_predicoes_gb():
    print("=== PREDIÇÕES GRADIENT BOOSTING ===")
    modelo_path = 'data/processado/gb_modelo.pkl'
    # Carrega dados
    with open('data/processado/dados_processados.pkl', 'rb') as f:
        dados = pickle.load(f)
    X_train_df = dados['X_train_balanced']
    y_train = dados['y_train_balanced'].values
    X_test_df = dados['X_test']
    y_test = dados['y_test'].values
    
    X_train = X_train_df.values
    X_test = X_test_df.values
    idx_test = X_test_df.index.values
    
    print(f"Dados: {X_train.shape[0]} treino, {X_test.shape[0]} teste")
    print(f"Split: Treino = {X_train.shape[0]} amostras, Teste = {X_test.shape[0]} amostras")
    print(f"Proporção: {X_test.shape[0]/(X_train.shape[0]+X_test.shape[0])*100:.1f}% para teste")
    print()
    
    # Normaliza
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if os.path.exists(modelo_path):
        with open(modelo_path, 'rb') as f:
            modelo = pickle.load(f)
        print('Modelo Gradient Boosting carregado!')
    else:
        print("🌳 Usando Gradient Boosting")
        modelo = GradientBoostingClassifier(n_estimators=100, random_state=42)
        modelo.fit(X_train_scaled, y_train)
        os.makedirs('data/processado', exist_ok=True)
        with open(modelo_path, 'wb') as f:
            pickle.dump(modelo, f)
        print('Modelo Gradient Boosting treinado e salvo!')
    
    print(f"{'Linha':<6} {'Real':<6} {'Predita':<8} {'Acerto':<8} {'Features'}")
    print("=" * 100)
    
    acertos = 0
    erros = []
    for i in range(len(X_test)):
        real = y_test[i]
        predita = modelo.predict([X_test_scaled[i]])[0]
        acertou = real == predita
        if acertou:
            acertos += 1
        else:
            erros.append((idx_test[i], real, predita))
        linha_idx = idx_test[i]
        features_str = "[" + ", ".join([f"{x:.1f}" for x in X_test[i]]) + "]"
        print(f"{linha_idx:<6} {real:<6} {predita:<8} {'✓' if acertou else '✗':<8} {features_str}")
    print("=" * 100)
    print(f"Acurácia: {acertos}/{len(X_test)} ({acertos/len(X_test)*100:.1f}%)")
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
    testar_predicoes_gb()

if __name__ == "__main__":
    main() 