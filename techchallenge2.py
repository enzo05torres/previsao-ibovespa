# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# %%
df = pd.read_csv('dados_ibovespa15anos.csv', sep=',"')

# %%
df.head()

# %%
df.shape

# %%
df.columns = ['Data', 'Ultimo', 'Abertura', 'Maxima', 'Minima', 'Volume', 'Variacao']

# %%
df.head()

# %%
df.info()

# %%
df.describe()

# %%
df.duplicated().sum()

# %%
for col in ['Ultimo', 'Abertura', 'Maxima', 'Minima']:
    df[col] = df[col].str.replace('"', '').str.replace('.', '').str.replace(',', '.').astype(float)

# %%
def tratar_volume(vol):
    vol = vol.replace('"', '').replace(',', '.').strip()

    if vol == '':
        return None  # ou np.nan

    try:
        if 'K' in vol:
            return float(vol.replace('K', '')) * 1e3
        elif 'M' in vol:
            return float(vol.replace('M', '')) * 1e6
        elif 'B' in vol:
            return float(vol.replace('B', '')) * 1e9
        else:
            return float(vol)
    except:
        return None 

df['Volume'] = df['Volume'].apply(tratar_volume)

# %%
df['Variacao'] = df['Variacao'].str.replace('%', 
                                '').str.replace(',', '.').str.replace('"', '')
df['Variacao'] = df['Variacao'].astype(float)

# %%
df['Data'] = pd.to_datetime(df['Data'].str.replace('"', ''), format='%d.%m.%Y')

# %%
df.sort_values('Data', inplace=True)
df.reset_index(drop=True, inplace=True)

# %%
df.head()

# %%
df.shape

# %%
# Linha Temporal dos Preços
df['Data'] = pd.to_datetime(df['Data'])
df.sort_values('Data', inplace=True)

plt.figure(figsize=(12, 6))
for col in ['Ultimo', 'Abertura', 'Maxima', 'Minima']:
    plt.plot(df['Data'], df[col], label=col)
plt.title('Evolução dos Preços ao Longo do Tempo')
plt.legend()
plt.grid()
plt.show()

# %%
# Variação Percentual ao Longo do Tempo
df['Variacao'] = df['Variacao']
plt.figure(figsize=(12, 5))
sns.lineplot(x=df['Data'], y=df['Variacao'])
plt.title('Variação Percentual Diária')
plt.xticks(rotation=45)
plt.grid()

# %%
# Distribuição das Variações Diárias
sns.histplot(df['Variacao'], kde=True, bins=20)
plt.title('Distribuição das Variações Diárias (%)')

# %%
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[['Ultimo', 'Abertura', 'Maxima', 'Minima']])
plt.title('Boxplot dos Preços')

# %%
df['Target'] = (df['Ultimo'].shift(-1) > df['Ultimo']).astype(int)
df.dropna(inplace=True)

# %%
# Retornos passados
df['Retorno_1d'] = df['Ultimo'].pct_change(1)
df['Retorno_3d'] = df['Ultimo'].pct_change(3)
df['Retorno_7d'] = df['Ultimo'].pct_change(7)

# Médias móveis
df['SMA_5'] = df['Ultimo'].rolling(5).mean()
df['SMA_10'] = df['Ultimo'].rolling(10).mean()

# Volatilidade (desvio padrão)
df['Vol_5'] = df['Ultimo'].rolling(5).std()
df['Vol_10'] = df['Ultimo'].rolling(10).std()

# Limpeza final
# %%
df['Ultimo_Seguinte'] = df['Ultimo'].shift(-1)
df['tendencia'] = (df['Ultimo_Seguinte'] > df['Ultimo']).astype(int)

# Remover a última linha (sem valor futuro)
df.dropna(inplace=True)

# %%
for lag in range(1, 4):
    df[f'lag_Ultimo_{lag}'] = df['Ultimo'].shift(lag)
    df[f'lag_Variacao_{lag}'] = df['Variacao'].shift(lag)
    df[f'lag_Volume_{lag}'] = df['Volume'].shift(lag)

# %%
plt.figure(figsize=(14, 6))
plt.plot(df['Data'], df['Retorno_1d'], label='Retorno 1 dia')
plt.plot(df['Data'], df['Retorno_3d'], label='Retorno 3 dias')
plt.plot(df['Data'], df['Retorno_7d'], label='Retorno 7 dias')
plt.axhline(0, color='gray', linestyle='--')
plt.title("Retornos Percentuais ao Longo do Tempo")
plt.xlabel("Data")
plt.ylabel("Retorno (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(12, 5))
sns.histplot(df['Retorno_1d'], kde=True, bins=30, label='1d', color='blue')
sns.histplot(df['Retorno_3d'], kde=True, bins=30, label='3d', color='orange')
sns.histplot(df['Retorno_7d'], kde=True, bins=30, label='7d', color='green')
plt.title("Distribuição dos Retornos")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(8, 5))
sns.boxplot(data=df[['Vol_5', 'Vol_10']])
plt.title("Boxplot da Volatilidade")
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(10, 5))
sns.scatterplot(data=df, x='Retorno_1d', y='Retorno_3d', hue='tendencia', alpha=0.7)
plt.title("Retorno 1d vs Retorno 3d com Tendência")
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Selecionando features técnicas e de mercado
features = [
    'lag_Ultimo_1', 'lag_Ultimo_2', 'lag_Ultimo_3',
    'lag_Variacao_1', 'lag_Variacao_2', 'lag_Variacao_3',
    'lag_Volume_1', 'lag_Volume_2', 'lag_Volume_3', 'Retorno_1d', 'Retorno_3d', 
    'Retorno_7d', 'SMA_5', 'SMA_10',
       'Vol_5', 'Vol_10', 'Ultimo_Seguinte'
]

# %%
features2 = ['lag_Variacao_2', 'lag_Variacao_3',
    'lag_Volume_1', 'lag_Volume_2', 'lag_Volume_3', 'Retorno_1d', 
    'Retorno_3d', 'Retorno_7d',
       'Vol_5', 'Vol_10', 'Ultimo_Seguinte'
]

# %%
df = df.dropna().copy()

# %%
df.columns

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Adiciona a variável alvo temporariamente
df_corr = df[features2 + ['tendencia']].copy()

# Matriz de correlação
plt.figure(figsize=(12, 8))
sns.heatmap(df_corr.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de Correlação")
plt.show()

# %%
X = df[features]
y = df['tendencia']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('log_reg', LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Avaliação
acc = accuracy_score(y_test, y_pred)
print(f"Acurácia com Lags (Logistic Regression): {acc:.2%}")
print(classification_report(y_test, y_pred))

# %%
from sklearn.ensemble import RandomForestClassifier

pipeline_rf = Pipeline([
    ('rf', RandomForestClassifier(n_estimators=1000, random_state=42))
])

# Treinamento
pipeline_rf.fit(X_train, y_train)

# Predição
y_pred_rf = pipeline_rf.predict(X_test)

# Avaliação
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"Acurácia com Lags (Random Forest): {acc_rf:.2%}")
print(classification_report(y_test, y_pred_rf))

# %%
from xgboost import XGBClassifier

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

pipeline_xgb = Pipeline([
    ('scaler', StandardScaler()),  # Opcional no XGBoost
    ('xgb', XGBClassifier(
        n_estimators=2000,
        max_depth=20,
        learning_rate=0.3,
        use_label_encoder=False,
        eval_metric='error',
        random_state=42
    ))
])

# Treinamento
pipeline_xgb.fit(X_train, y_train)

# Predição
y_pred_xgb = pipeline_xgb.predict(X_test)

# Avaliação
acc_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"Acurácia com Lags (XGBoost): {acc_xgb:.2%}")
print(classification_report(y_test, y_pred_xgb))

# %%
print(df['tendencia'].value_counts(normalize=True))

# %%
df

# %%
y_pred_train = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)

# Acurácias
acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)

# Relatório de teste
print(f"Acurácia (Treino): {acc_train:.2%}")
print(f"Acurácia (Teste): {acc_test:.2%}")
print("\nRelatório de Classificação (Teste):")
print(classification_report(y_test, y_pred_test))

# Gráfico comparativo
labels = ['Treino', 'Teste']
valores = [acc_train * 100, acc_test * 100]

plt.figure(figsize=(6, 4))
plt.bar(labels, valores)
plt.ylim(80, 90)
plt.ylabel('Acurácia (%)')
plt.title('Comparação de Acurácia: Treino vs Teste')

# Mostrar os valores nas barras
for i, v in enumerate(valores):
    plt.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()


# %%
ultimo_mes = df.tail(30).copy()
dados_treino = df.iloc[:-30].copy()

# Features e alvo
X_train = dados_treino[features]
y_train = dados_treino['tendencia']

X_test_final = ultimo_mes[features]
y_test_final = ultimo_mes['tendencia']

# Passo 2 e 3: Treinar e prever
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('log_reg', LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)
y_pred_final = pipeline.predict(X_test_final)

# Passo 4: Avaliar
from sklearn.metrics import accuracy_score, classification_report

acc_final = accuracy_score(y_test_final, y_pred_final)
print(f"\nAcurácia no último mês (30 dias): {acc_final:.2%}")
print(classification_report(y_test_final, y_pred_final))

# %%
df['Data'] = pd.to_datetime(df['Data'])
df.set_index('Data')['Ultimo'].plot(figsize=(12, 5), 
                                    title='Fechamento Diário do IBOVESPA')

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.plot(y_test_final.values, label='Real', marker='o')
plt.plot(y_pred_final, label='Previsto', marker='x')
plt.title("Tendência Real vs Prevista (Últimos 30 dias)")
plt.xlabel("Dias")
plt.ylabel("Tendência (0 = Queda, 1 = Alta)")
plt.legend()
plt.grid(True)

# %%
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(pipeline, X_test_final, y_test_final)
plt.title("Matriz de Confusão - Último mês")

# %%
import seaborn as sns

lags_cols = [col for col in df.columns if col.startswith("lag_")]
correlacoes = df[lags_cols + ['tendencia']].corr()['tendencia'].drop('tendencia')

plt.figure(figsize=(10, 5))
sns.barplot(x=correlacoes.index, y=correlacoes.values)
plt.title("Correlação das Lags com a Tendência")
plt.tight_layout()
plt.ylabel("Correlação com 'tendência'")
plt.xlabel("Variáveis")
plt.grid(True)
plt.xticks(rotation=45)

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Lista de variáveis que você quer avaliar a correlação com a tendência
variaveis = ['Retorno_1d', 'Retorno_3d', 'Retorno_7d',
             'SMA_5', 'SMA_10',
             'Vol_5', 'Vol_10', 'Ultimo_Seguinte', 'tendencia']

# Calcula a correlação com a tendência
correlacoes = df[variaveis].corr()['tendencia'].drop('tendencia')

# Plota o gráfico
plt.figure(figsize=(10, 5))
sns.barplot(x=correlacoes.index, y=correlacoes.values)
plt.title("Correlação das Variáveis com a Tendência do IBOVESPA")
plt.ylabel("Correlação com 'tendência'")
plt.xlabel("Variáveis")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.plot(y_test_final.reset_index(drop=True), label='Real', marker='o')
plt.plot(y_pred_final, label='Previsto', marker='x')
plt.title('Tendência Real vs Prevista - Últimos 30 dias')
plt.xlabel('Dias')
plt.ylabel('Tendência (1 = Alta, 0 = Queda)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
import statsmodels.api as sm

# %%
residuos = y_test_final.reset_index(drop=True) - y_pred_final

# %%
plt.figure(figsize=(10, 4))
plt.plot(residuos, marker='o', linestyle='--', label='Resíduos')
plt.axhline(0, color='red', linestyle='--')
plt.title("Resíduos ao Longo do Tempo - Últimos 30 dias")
plt.xlabel("Dias")
plt.ylabel("Erro (Real - Previsto)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(6, 4))
sns.histplot(residuos, bins=10, kde=True)
plt.title("Distribuição dos Resíduos")
plt.xlabel("Erro")
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
