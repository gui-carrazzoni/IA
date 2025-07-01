import pickle
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from predicoes_simples_knn import testar_predicoes_knn
from testar_predicoes_gb import testar_predicoes_gb
from testar_predicoes_elm import testar_predicoes_elm
from testar_predicoes_mlp import testar_predicoes_mlp
from testar_predicoes_genetico import testar_predicoes_genetico
from KNN import KNN
from ELM import ELMClassifier
from MLP import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
import os

# Carregar dados processados
with open('data/processado/dados_processados.pkl', 'rb') as f:
    dados = pickle.load(f)

X_train = dados['X_train_balanced']
y_train = dados['y_train_balanced']
X_test = dados['X_test']
y_test = dados['y_test']

# Binarizar y_test para AUC
classes = np.unique(y_test)
y_test_bin = label_binarize(y_test, classes=classes)

resultados = []

# KNN
scaler_knn = pickle.load(open('data/processado/dados_processados.pkl', 'rb'))['scaler']
X_train_knn = dados['X_train_original'].values
X_test_knn = dados['X_test'].values
scaler_knn = scaler_knn if hasattr(scaler_knn, 'transform') else None
if scaler_knn:
    X_train_knn = scaler_knn.transform(X_train_knn)
    X_test_knn = scaler_knn.transform(X_test_knn)
knn = KNN(k=5)
knn.fit(X_train_knn, dados['y_train_original'].values)
y_pred_knn = knn.predict(X_test_knn)
acc_knn = accuracy_score(y_test, y_pred_knn)
print('\nKNN:')
print('Acurácia:', acc_knn)
print('Predições:', y_pred_knn[:36])
y_test_np = np.array(y_test)
print('Valores reais:', y_test_np[:36])
print('Erros:', np.sum(y_test_np[:36] != y_pred_knn[:36]))
try:
    auc_knn = roc_auc_score(y_test_bin, label_binarize(y_pred_knn, classes=classes), average='macro', multi_class='ovr')
except:
    auc_knn = np.nan
resultados.append(['KNN', acc_knn, f1_score(y_test, y_pred_knn, average='macro'), auc_knn])

# Gradient Boosting
modelo_gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
modelo_gb.fit(X_train, y_train)
y_pred_gb = modelo_gb.predict(X_test)
acc_gb = accuracy_score(y_test, y_pred_gb)
print('\nGradient Boosting:')
print('Acurácia:', acc_gb)
print('Predições:', y_pred_gb[:36])
print('Valores reais:', y_test_np[:36])
print('Erros:', np.sum(y_test_np[:36] != y_pred_gb[:36]))
y_proba_gb = modelo_gb.predict_proba(X_test)
auc_gb = roc_auc_score(y_test_bin, y_proba_gb, average='macro', multi_class='ovr')
resultados.append(['Gradient Boosting', acc_gb, f1_score(y_test, y_pred_gb, average='macro'), auc_gb])

# ELM
with open('data/processado/elm_modelo.pkl', 'rb') as f:
    modelo_elm = pickle.load(f)
y_pred_elm = modelo_elm.predict(X_test)
acc_elm = accuracy_score(y_test, y_pred_elm)
print('\nELM:')
print('Acurácia:', acc_elm)
print('Predições:', y_pred_elm[:36])
print('Valores reais:', y_test_np[:36])
print('Erros:', np.sum(y_test_np[:36] != y_pred_elm[:36]))
y_proba_elm = modelo_elm.predict_proba(X_test)
auc_elm = roc_auc_score(y_test_bin, y_proba_elm, average='macro', multi_class='ovr')
resultados.append(['ELM', acc_elm, f1_score(y_test, y_pred_elm, average='macro'), auc_elm])

# MLP
mlp_scaler_path = 'data/processado/mlp_scaler.pkl'
if os.path.exists(mlp_scaler_path):
    with open(mlp_scaler_path, 'rb') as f:
        mlp_scaler = pickle.load(f)
    X_test_mlp = mlp_scaler.transform(X_test)
with open('data/processado/mlp_modelo.pkl', 'rb') as f:
    modelo_mlp = pickle.load(f)
y_pred_mlp = modelo_mlp.predict(X_test_mlp)
acc_mlp = accuracy_score(y_test, y_pred_mlp)
print('\nMLP:')
print('Acurácia:', acc_mlp)
print('Predições:', y_pred_mlp[:36])
print('Valores reais:', y_test_np[:36])
print('Erros:', np.sum(y_test_np[:36] != y_pred_mlp[:36]))
y_proba_mlp = modelo_mlp.predict_proba(X_test_mlp)
auc_mlp = roc_auc_score(y_test_bin, y_proba_mlp, average='macro', multi_class='ovr')
resultados.append(['MLP', acc_mlp, f1_score(y_test, y_pred_mlp, average='macro'), auc_mlp])

# Genético
with open('data/processado/modelo_genetico_otimizado.pkl', 'rb') as f:
    modelo_info = pickle.load(f)
modelo_gen = modelo_info['modelo']
features_mask = modelo_info['features_mask']
if hasattr(X_test, 'loc'):
    X_test_gen = X_test.loc[:, modelo_info['features_selecionadas']].values
else:
    X_test_gen = X_test[:, features_mask]
y_pred_gen = modelo_gen.predict(X_test_gen)
acc_gen = accuracy_score(y_test, y_pred_gen)
print('\nGenético:')
print('Acurácia:', acc_gen)
print('Predições:', y_pred_gen[:36])
print('Valores reais:', y_test_np[:36])
print('Erros:', np.sum(y_test_np[:36] != y_pred_gen[:36]))
y_proba_gen = modelo_gen.predict_proba(X_test_gen)
auc_gen = roc_auc_score(y_test_bin, y_proba_gen, average='macro', multi_class='ovr')
resultados.append(['Genético', acc_gen, f1_score(y_test, y_pred_gen, average='macro'), auc_gen])

# Mostrar resultados
print(f"{'Modelo':<18} {'Acurácia':<10} {'F1':<10} {'AUC':<10}")
for nome, acc, f1, auc in resultados:
    print(f"{nome:<18} {acc:<10.4f} {f1:<10.4f} {auc:<10.4f}")

# Visualização gráfica
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

modelos = [r[0] for r in resultados]
acuracias = [r[1] for r in resultados]
f1s = [r[2] for r in resultados]
aucs = [r[3] for r in resultados]

x = np.arange(len(modelos))
width = 0.25

fig, ax = plt.subplots(figsize=(10,6))
rects1 = ax.bar(x - width, acuracias, width, label='Acurácia')
rects2 = ax.bar(x, f1s, width, label='F1')
rects3 = ax.bar(x + width, aucs, width, label='AUC')

ax.set_ylabel('Pontuação')
ax.set_title('Comparação de Métricas dos Modelos')
ax.set_xticks(x)
ax.set_xticklabels(modelos, rotation=15)
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
ax.legend()

for rects in [rects1, rects2, rects3]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show() 