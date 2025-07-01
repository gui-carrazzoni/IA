#!/usr/bin/env python3
"""
Script para testar predições KNN com novos splits aleatórios a cada execução.
Mostra: índice da linha original, classe real, classe predita e as features.
"""

import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from KNN import KNN

def testar_predicoes_knn():
    print("=== PREDIÇÕES KNN - NOVOS SPLITS ALEATÓRIOS ===")

    # Carrega dados originais
    with open('data/processado/dados_processados.pkl', 'rb') as f:
        dados = pickle.load(f)

    X_df = dados['X_train_original']
    y = dados['y_train_original'].values

    # Split aleatório a cada execução
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, stratify=y
    )

    X_train = X_train_df.values
    X_test = X_test_df.values
    idx_test = X_test_df.index.values

    print(f"Dados: {X_train.shape[0]} treino, {X_test.shape[0]} teste")
    print(f"Split: Treino = {X_train.shape[0]} amostras, Teste = {X_test.shape[0]} amostras")
    print(f"Proporção: {X_test.shape[0]/(X_train.shape[0]+X_test.shape[0])*100:.1f}% para teste")
    print()

    # Normaliza e treina
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("🔍 Usando K-Nearest Neighbors (KNN)")
    knn = KNN(k=5)
    knn.fit(X_train_scaled, y_train)

    print(f"{'Linha':<6} {'Real':<6} {'Predita':<8} {'Acerto':<8} {'Features'}")
    print("=" * 100)

    acertos = 0
    erros = []
    
    for i in range(len(X_test)):
        real = y_test[i]
        predita = knn.predict_single(X_test_scaled[i])
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
    
    # Destaque das linhas erradas
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
    testar_predicoes_knn()

if __name__ == "__main__":
    main() 