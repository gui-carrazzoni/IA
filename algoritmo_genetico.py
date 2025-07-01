import pickle
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score
import random
import copy
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class AlgoritmoGenetico:
    def __init__(self, dados, tamanho_populacao=50, taxa_mutacao=0.1, taxa_crossover=0.8, 
                 max_geracoes=100, elitismo=0.1):
        
        # Argumentos
        self.dados = dados
        self.tamanho_populacao = tamanho_populacao
        self.taxa_mutacao = taxa_mutacao
        self.taxa_crossover = taxa_crossover
        self.max_geracoes = max_geracoes
        self.elitismo = elitismo
        
        # Dados de treino
        self.X_train = dados['X_train_balanced']
        self.y_train = dados['y_train_balanced']
        self.feature_names = dados['feature_names']
        
        # Histórico de fitness
        self.historico_fitness = []
        self.melhor_fitness_geracao = []
        
    def criar_individuo(self) -> Dict:
        """Cria um indivíduo (solução) aleatória"""
        individuo = {
            # Hiperparâmetros do Random Forest (valores menores para velocidade)
            'n_estimators': random.choice([25, 50, 75, 100]),
            'max_depth': random.choice([None, 5, 10, 15, 20]),
            'min_samples_split': random.choice([2, 3, 4, 5]),
            'min_samples_leaf': random.choice([1, 2, 3]),
            'criterion': random.choice(['gini', 'entropy']),
            
            # Seleção de features (vetor binário)
            'features_selecionadas': [random.choice([0, 1]) for _ in range(len(self.feature_names))]
        }
        return individuo
    
    def criar_populacao_inicial(self) -> List[Dict]:

        return [self.criar_individuo() for _ in range(self.tamanho_populacao)]
    
    def fitness(self, individuo: Dict) -> float:

        # Selecionar features
        features_mask = np.array(individuo['features_selecionadas'])
        if sum(features_mask) == 0:  # Se nenhuma feature foi selecionada
            return 0.0
        
        # Corrigir seleção de colunas para DataFrame pandas
        colunas_selecionadas = [col for col, use in zip(self.feature_names, features_mask) if use]
        X_selecionado = self.X_train.loc[:, colunas_selecionadas]
        
        # Criar modelo com hiperparâmetros do indivíduo
        modelo = RandomForestClassifier(
            n_estimators=individuo['n_estimators'],
            max_depth=individuo['max_depth'],
            min_samples_split=individuo['min_samples_split'],
            min_samples_leaf=individuo['min_samples_leaf'],
            criterion=individuo['criterion'],
            random_state=42,
            n_jobs=1  # Usar apenas 1 processo
        )
        
        # Avaliar com cross-validation
        scores = cross_val_score(modelo, X_selecionado, self.y_train, cv=3, scoring='f1_macro', n_jobs=1)
        fitness_score = scores.mean()
        
        # Penalizar por usar muitas features (parcimônia)
        num_features = sum(features_mask)
        penalizacao = 0.01 * (num_features / len(self.feature_names))
        
        return max(0.0, fitness_score - penalizacao)
    
    def selecao_torneio(self, populacao: List[Dict], k=3) -> Dict:
        """Seleção por torneio"""
        torneio = random.sample(populacao, k)
        return max(torneio, key=self.fitness)
    
    def crossover(self, pai1: Dict, pai2: Dict) -> Tuple[Dict, Dict]:
        """Operador de crossover (cruzamento)"""
        if random.random() > self.taxa_crossover:
            return pai1, pai2
        
        filho1 = copy.deepcopy(pai1)
        filho2 = copy.deepcopy(pai2)
        
        # Crossover para hiperparâmetros
        if random.random() < 0.5:
            filho1['n_estimators'], filho2['n_estimators'] = filho2['n_estimators'], filho1['n_estimators']
        if random.random() < 0.5:
            filho1['max_depth'], filho2['max_depth'] = filho2['max_depth'], filho1['max_depth']
        if random.random() < 0.5:
            filho1['min_samples_split'], filho2['min_samples_split'] = filho2['min_samples_split'], filho1['min_samples_split']
        if random.random() < 0.5:
            filho1['min_samples_leaf'], filho2['min_samples_leaf'] = filho2['min_samples_leaf'], filho1['min_samples_leaf']
        if random.random() < 0.5:
            filho1['criterion'], filho2['criterion'] = filho2['criterion'], filho1['criterion']
        
        # Crossover para seleção de features (ponto único)
        ponto_corte = random.randint(1, len(self.feature_names) - 1)
        filho1['features_selecionadas'] = pai1['features_selecionadas'][:ponto_corte] + pai2['features_selecionadas'][ponto_corte:]
        filho2['features_selecionadas'] = pai2['features_selecionadas'][:ponto_corte] + pai1['features_selecionadas'][ponto_corte:]
        
        return filho1, filho2
    
    def mutacao(self, individuo: Dict):
        """Operador de mutação"""
        if random.random() < self.taxa_mutacao:
            # Mutação de hiperparâmetros
            if random.random() < 0.3:
                individuo['n_estimators'] = random.choice([25, 50, 75, 100])
            if random.random() < 0.3:
                individuo['max_depth'] = random.choice([None, 5, 10, 15, 20])
            if random.random() < 0.3:
                individuo['min_samples_split'] = random.choice([2, 3, 4, 5])
            if random.random() < 0.3:
                individuo['min_samples_leaf'] = random.choice([1, 2, 3])
            if random.random() < 0.3:
                individuo['criterion'] = random.choice(['gini', 'entropy'])
            
            # Mutação de features (bit flip)
            for i in range(len(individuo['features_selecionadas'])):
                if random.random() < 0.1:  # 10% de chance de flip por bit
                    individuo['features_selecionadas'][i] = 1 - individuo['features_selecionadas'][i]
    
    def executar(self) -> Dict:
        """Executa o algoritmo genético"""
        print("🧬 INICIANDO ALGORITMO GENÉTICO")
        print("=" * 50)
        
        # Criar população inicial
        populacao = self.criar_populacao_inicial()
        
        # Avaliar população inicial
        fitness_populacao = [self.fitness(ind) for ind in populacao]
        self.historico_fitness.append(np.mean(fitness_populacao))
        
        melhor_individuo = populacao[np.argmax(fitness_populacao)]
        melhor_fitness = max(fitness_populacao)
        self.melhor_fitness_geracao.append(melhor_fitness)
        
        print(f"Geração 0 - Melhor Fitness: {melhor_fitness:.4f}")
        
        # Loop principal
        for geracao in range(1, self.max_geracoes + 1):
            nova_populacao = []
            
            # Elitismo: manter os melhores indivíduos
            num_elite = int(self.elitismo * self.tamanho_populacao)
            indices_ordenados = np.argsort(fitness_populacao)[::-1]
            elite = [populacao[i] for i in indices_ordenados[:num_elite]]
            nova_populacao.extend(elite)
            
            # Gerar resto da população
            while len(nova_populacao) < self.tamanho_populacao:
                # Seleção
                pai1 = self.selecao_torneio(populacao)
                pai2 = self.selecao_torneio(populacao)
                
                # Crossover
                filho1, filho2 = self.crossover(pai1, pai2)
                
                # Mutação
                self.mutacao(filho1)
                self.mutacao(filho2)
                
                nova_populacao.extend([filho1, filho2])
            
            # Ajustar tamanho da população
            nova_populacao = nova_populacao[:self.tamanho_populacao]
            
            # Atualizar população
            populacao = nova_populacao
            
            # Avaliar nova população
            fitness_populacao = [self.fitness(ind) for ind in populacao]
            self.historico_fitness.append(np.mean(fitness_populacao))
            
            # Atualizar melhor indivíduo
            melhor_ind_atual = populacao[np.argmax(fitness_populacao)]
            melhor_fitness_atual = max(fitness_populacao)
            
            if melhor_fitness_atual > melhor_fitness:
                melhor_individuo = copy.deepcopy(melhor_ind_atual)
                melhor_fitness = melhor_fitness_atual
            
            self.melhor_fitness_geracao.append(melhor_fitness)
            
            # Mostrar progresso
            if geracao % 10 == 0:
                print(f"Geração {geracao} - Melhor Fitness: {melhor_fitness:.4f} - Média: {np.mean(fitness_populacao):.4f}")
        
        print(f"Melhor Fitness Final: {melhor_fitness:.4f}")
        
        return melhor_individuo
    
    def visualizar_evolucao(self):
        """Visualiza a evolução do algoritmo genético"""
        plt.figure(figsize=(15, 5))
        
        # Gráfico de fitness
        plt.subplot(1, 3, 1)
        plt.plot(self.historico_fitness, label='Fitness Médio', alpha=0.7)
        plt.plot(self.melhor_fitness_geracao, label='Melhor Fitness', linewidth=2)
        plt.xlabel('Geração')
        plt.ylabel('Fitness')
        plt.title('Evolução do Fitness')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Gráfico de melhoria
        plt.subplot(1, 3, 2)
        melhorias = np.diff(self.melhor_fitness_geracao)
        plt.plot(melhorias, color='green', alpha=0.7)
        plt.xlabel('Geração')
        plt.ylabel('Melhoria no Fitness')
        plt.title('Melhoria por Geração')
        plt.grid(True, alpha=0.3)
        
        # Gráfico de convergência
        plt.subplot(1, 3, 3)
        plt.plot(self.historico_fitness, label='Média', alpha=0.7)
        plt.fill_between(range(len(self.historico_fitness)), 
                        [f - 0.01 for f in self.historico_fitness],
                        [f + 0.01 for f in self.historico_fitness], alpha=0.3)
        plt.xlabel('Geração')
        plt.ylabel('Fitness')
        plt.title('Convergência da População')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('evolucao_genetico.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualização salva como 'evolucao_genetico.png'")

def avaliar_melhor_solucao(melhor_individuo: Dict, dados):

    # Extrair dados
    X_train = dados['X_train_balanced']
    y_train = dados['y_train_balanced']
    X_test = dados['X_test']
    y_test = dados['y_test']
    feature_names = dados['feature_names']
    
    # Selecionar features
    features_mask = np.array(melhor_individuo['features_selecionadas'])
    features_selecionadas = [feature_names[i] for i, selected in enumerate(features_mask) if selected]
    
    print(f"Hiperparâmetros otimizados:")
    for param, value in melhor_individuo.items():
        if param != 'features_selecionadas':
            print(f"   {param}: {value}")
    
    print(f"\nFeatures selecionadas ({sum(features_mask)}/{len(feature_names)}):")
    for i, feature in enumerate(features_selecionadas):
        print(f"   {i+1}. {feature}")
    
    # Treinar modelo final
    if hasattr(X_train, 'loc'):
        # DataFrame: seleciona pelas colunas
        X_train_selecionado = X_train.loc[:, features_selecionadas].values
        X_test_selecionado = X_test.loc[:, features_selecionadas].values
    else:
        # Numpy array: seleciona pela máscara booleana
        X_train_selecionado = X_train[:, features_mask]
        X_test_selecionado = X_test[:, features_mask]
    
    modelo_final = RandomForestClassifier(
        n_estimators=melhor_individuo['n_estimators'],
        max_depth=melhor_individuo['max_depth'],
        min_samples_split=melhor_individuo['min_samples_split'],
        min_samples_leaf=melhor_individuo['min_samples_leaf'],
        criterion=melhor_individuo['criterion'],
        random_state=42
    )
    
    modelo_final.fit(X_train_selecionado, y_train)
    
    # Avaliar no conjunto de teste
    y_pred = modelo_final.predict(X_test_selecionado)
    acuracia = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"\nPerformance no conjunto de teste:")
    print(f"   Acurácia: {acuracia:.4f}")
    print(f"   F1-Score Macro: {f1:.4f}")
    
    # Cross-validation
    scores_cv = cross_val_score(modelo_final, X_train_selecionado, y_train, cv=5, scoring='f1_macro')
    print(f"   CV F1-Score: {scores_cv.mean():.4f} (+/- {scores_cv.std()*2:.4f})")
    
    # Salvar modelo otimizado
    modelo_otimizado = {
        'modelo': modelo_final,
        'features_mask': features_mask,
        'feature_names': feature_names,
        'features_selecionadas': features_selecionadas,
        'hiperparametros': {k: v for k, v in melhor_individuo.items() if k != 'features_selecionadas'},
        'performance': {
            'acuracia': acuracia,
            'f1_score': f1,
            'cv_score': scores_cv.mean(),
            'cv_std': scores_cv.std()
        }
    }
    os.makedirs('data/processado', exist_ok=True)
    with open('data/processado/modelo_genetico_otimizado.pkl', 'wb') as f:
        pickle.dump(modelo_otimizado, f)
    
    print("✅ Modelo otimizado salvo como 'data/processado/modelo_genetico_otimizado.pkl'")
    
    return modelo_otimizado

def main():
    """Função principal"""
    print("🧬 ALGORITMO GENÉTICO PARA OTIMIZAÇÃO DE MODELOS")
    print("=" * 60)
    
    # Carregar dados processados
    print("📂 Carregando dados processados...")
    with open('data/processado/dados_processados.pkl', 'rb') as f:
        dados = pickle.load(f)
    
    # Extrair dados do preprocessamento_simples
    if 'X_train_balanced' in dados:
        dados_para_ga = dados
    else:
        # Se for do preprocessamento_dados, extrair dados internos
        dados_para_ga = dados['dados_processados']
        # Converter para formato esperado
        dados_para_ga = {
            'X_train_balanced': dados_para_ga['X_train'],
            'y_train_balanced': dados_para_ga['y_train'],
            'X_test': dados_para_ga['X_test'],
            'y_test': dados_para_ga['y_test'],
            'feature_names': dados_para_ga['feature_names']
        }
    
    print(f"Dados carregados: {dados_para_ga['X_train_balanced'].shape[0]} amostras, {dados_para_ga['X_train_balanced'].shape[1]} features")
    
    # Configurar e executar algoritmo genético
    ga = AlgoritmoGenetico(
        dados=dados_para_ga,
        tamanho_populacao=15,  # Reduzido para execução mais rápida
        taxa_mutacao=0.15,
        taxa_crossover=0.8,
        max_geracoes=10,  # Reduzido para demonstração
        elitismo=0.2
    )
    
    # Executar algoritmo genético
    melhor_solucao = ga.executar()
    
    # Visualizar evolução
    ga.visualizar_evolucao()
    
    # Avaliar melhor solução
    modelo_otimizado = avaliar_melhor_solucao(melhor_solucao, dados_para_ga)
    
    print("=" * 30)
    print("Algoritmo genético executado")
    print("Hiperparâmetros otimizados")
    print("Features selecionadas")
    print("Modelo final treinado")
    print("Visualizações criadas")
    print("Modelo salvo para uso futuro")

if __name__ == "__main__":
    main() 