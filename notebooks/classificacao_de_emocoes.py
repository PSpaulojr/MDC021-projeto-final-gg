# %% [markdown]
# **Data Understanding**

# %%
# mkdir -p ~/.kaggle && echo  > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token

# %%
# !pip install kagglehub -q

# %%
import kagglehub

path = kagglehub.competition_download(
    "classificacao-de-emocoes-em-comentarios"
)

print(path)

# %%
import os

os.listdir(path)

# %%
import pandas as pd

train_df = pd.read_csv(f"{path}/train.csv")

print(train_df.shape)

train_df.head()

# %% [markdown]
# A estrutura da base contém 46185 registros e 29 colunas. A base contém uma coluna de texto e 28 colunas correspondentes a emoções.

# %%
emotion_columns = train_df.columns.drop("text")

train_df[emotion_columns].sum().sort_values(ascending=False)

# %%
#Quantidade de emoções por cometários

num_labels = train_df[emotion_columns].sum(axis=1)

pd.DataFrame(num_labels.value_counts().sort_index())

# %%
#Distribuição de exemplos por emoção
class_counts = train_df[emotion_columns].sum().sort_values(ascending=False)

plt.figure(figsize=(14, 7))
ax = sns.barplot(x=class_counts.index, y=class_counts.values, color="steelblue")

ax.bar_label(ax.containers[0], fontsize=9, padding=3)

plt.xticks(rotation=45, ha="right")
plt.title("Quantidade de comentários por emoção")
plt.xlabel("Emoção")
plt.ylabel("Quantidade de comentários")
plt.tight_layout()
plt.show()

# %%
pd.DataFrame(num_labels.value_counts(normalize=True).sort_index() * 100)

# %% [markdown]
# Foi identificado 680 comentários (1,47%) que não possui nenhuma emoção associada.

# %% [markdown]
# Embora o problema seja de classificação multirrótulo, podemos observar que a maioria dos comentários possui apenas uma emoção associada. Aproximadamente 81% dos exemplos apresentam um único rótulo, enquanto 15% possuem duas emoções. Comentários com três ou mais emoções são relativamente raros. Esses resultados indicam que, apesar da característica multirrótulo da tarefa, a maior parte dos exemplos apresenta uma única emoção predominante. Também identificamos 680 comentários sem nenhuma emoção.
#

# %% [markdown]
# A análise da distribuição das classes revelou forte desbalanceamento entre as emoções. Enquanto a classe Neutral possui mais de 12 mil ocorrências no conjunto de treino, emoções como Grief, Pride e Relief possuem menos de 300 exemplos. Esse comportamento representa um desafio adicional para os modelos de classificação, especialmente considerando a métrica F1 Macro, na qual classes raras possuem o mesmo peso das classes frequentes.

# %%
import numpy as np

text_lengths = train_df["text"].str.split().apply(len)

print("Média:", np.mean(text_lengths))
print("Mediana:", np.median(text_lengths))
print("Percentil 90:", np.percentile(text_lengths, 90))
print("Percentil 95:", np.percentile(text_lengths, 95))
print("Máximo:", np.max(text_lengths))

# %% [markdown]
# Os comentários do conjunto são curtos. A média observada foi de aproximadamente 13 palavras por comentário, com mediana de 12 palavras. Além disso, 95% dos exemplos possuem até 24 palavras e o maior comentário encontrado possui apenas 33 palavras. Esses resultados indicam que o conjunto é composto predominantemente por mensagens curtas, fato que pode influenciar positivamente o custo computacional de modelos baseados em Transformers.
#

# %%
#Verificando se exitem valores nulos

train_df.isnull().sum()

# %% [markdown]
# Não existem valores nulos no conjunto de treino.

# %%
#Calculando a co-ocorrência entre emoções

emotion_labels = train_df[emotion_columns]
total_comments = len(emotion_labels)

co_occurrence = emotion_labels.T.dot(emotion_labels)
p_joint = co_occurrence / total_comments
p_single = emotion_labels.mean()

lift = p_joint / np.outer(p_single, p_single)
lift = lift.mask(np.eye(len(lift), dtype=bool))

# %%
#Heatmap de lift entre emoções
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(14, 12))
sns.heatmap(lift, cmap="coolwarm", center=1, vmax=5)
plt.title("Lift de co-ocorrência entre emoções")
plt.show()

# %%
#Pares de emoções com maior lift (ignorando pares muito raros)
pairs = lift.stack().reset_index()
pairs.columns = ["emotion_a", "emotion_b", "lift"]
pairs = pairs[pairs["emotion_a"] < pairs["emotion_b"]]

pairs["count"] = [
    co_occurrence.loc[a, b]
    for a, b in zip(pairs["emotion_a"], pairs["emotion_b"])
]

pairs[pairs["count"] >= 20].sort_values("lift", ascending=False).head(15)

# %%
#Pares de emoções com menor lift (que tendem a não aparecer juntas)
pairs.sort_values("lift").head(15)

# %%
#Verificando se neutral aparece junto de alguma outra emoção
co_occurrence.loc["neutral"].drop("neutral").sum()

# %%
#Pares de emoções com menor lift, excluindo neutral e pares sem nenhuma ocorrência
pairs_no_neutral = pairs[
    (pairs["emotion_a"] != "neutral") & (pairs["emotion_b"] != "neutral")
]

pairs_no_neutral[pairs_no_neutral["count"] > 0].sort_values("lift").head(15)

# %% [markdown]
# Verificando a co-ocorrência, que foi feita com a métrica lift, que compara a frequência observada de cada par de emoções com a frequência esperada caso fossem independentes. Os pares com maior lift são semanticamente próximos: fear e nervousness (19,5), grief e sadness (7,3), embarrassment e remorse (4,6), excitement e joy (4,2) e disappointment e sadness (3,7).
#
# Também foi observado que neutral nunca aparece junto de outra emoção, ou seja, é mutuamente exclusiva em relação às demais.
#
# Vemos que pares como disapproval-nervousness, annoyance-joy, curiosity-relief tem lift muito baixo, mostrando que se evitam.

# %%
#Carregando o teste e o template de submissão
test_df = pd.read_csv(f"{path}/test.csv")
sample_df = pd.read_csv(f"{path}/sample_submission.csv")

print(test_df.shape)
print(sample_df.shape)
sample_df.head()

# %%
#Verificando se as colunas do template seguem a mesma ordem do treino
template_emotions = sample_df.columns.drop("ID")

print(list(template_emotions) == list(emotion_columns))
print(sample_df[template_emotions].stack().unique())

# %%
#Textos duplicados no treino e sobreposição com o teste
print("Textos duplicados no treino:", train_df["text"].duplicated().sum())
print("Comentários do teste que aparecem no treino:", test_df["text"].isin(train_df["text"]).sum())

# %% [markdown]
# Não foram encontrados comentários duplicados no conjunto de treino, portanto não há rótulos conflitantes para o mesmo texto. Também nenhum comentário do teste aparece no treino, de modo que o desempenho na validação não é inflado por memorização de textos repetidos.

# %% [markdown]
# O arquivo sample_submission.csv possui 11547 linhas e 29 colunas (ID e as 28 emoções), na mesma ordem das colunas do conjunto de treino. Todas as colunas de emoção estão preenchidas com -1. Iremos substituir por 0 ou 1 para submissão

# %%
#Tamanho dos comentários em tokens (tokenizer do DistilBERT)
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

token_lengths = train_df["text"].apply(lambda text: len(tokenizer.encode(text)))

print("Média:", np.mean(token_lengths))
print("Mediana:", np.median(token_lengths))
print("Percentil 95:", np.percentile(token_lengths, 95))
print("Percentil 99:", np.percentile(token_lengths, 99))
print("Máximo:", np.max(token_lengths))

# %%
#Histograma do tamanho dos comentários em tokens (cortado em 60 para não distorcer a escala)
plt.figure(figsize=(10, 5))
sns.histplot(token_lengths[token_lengths <= 60], binwidth=2)
plt.axvline(40, color="red", linestyle="--", label="max_length = 40")
plt.title("Tamanho dos comentários em tokens")
plt.xlabel("Tokens")
plt.legend()
plt.show()

# %% [markdown]
# Considerando o tokenizer do DistilBERT, 95% dos comentários possuem até 34 tokens. Dessa forma, será utilizado max_length = 40 (esse valor cobre cerca de 99% dos comentários, evitando truncar até esse percentual de comentários) nos modelos baseados em Transformers.

# %%
#Adversarial validation: classificador que tenta distinguir treino de teste
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

adv_texts = pd.concat([train_df["text"], test_df["text"]], ignore_index=True)
adv_labels = np.array([0] * len(train_df) + [1] * len(test_df))

adv_model = make_pipeline(
    TfidfVectorizer(),
    LogisticRegression(max_iter=1000)
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

adv_proba = cross_val_predict(
    adv_model,
    adv_texts,
    adv_labels,
    cv=skf,
    method="predict_proba"
)[:, 1]

print(f"AUC: {roc_auc_score(adv_labels, adv_proba):.4f}")

# %% [markdown]
# Para verificar se o conjunto de treino representa o conjunto de teste, foi treinado um classificador (TF-IDF + regressão logística) para distinguir comentários de treino e de teste, avaliado com validação cruzada de 5 folds. O AUC obtido foi de 0,4971, muito próximo de 0,5, o que indica que os dois conjuntos não são distinguíveis pelo texto. Portanto, a validação feita a partir do treino deve representar bem o teste.

# %% [markdown]
# **Data Processing**

# %%
import pandas as pd

train_df = pd.read_csv(f"{path}/train.csv")
test_df = pd.read_csv(f"{path}/test.csv")

print(train_df.shape)
print(test_df.shape)

# %%
test_df.head()

# %%
emotion_columns = train_df.columns.drop("text")

X = train_df["text"]
y = train_df[emotion_columns]

print(X.shape)
print(y.shape)

# %%
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
print(y_val.shape)

# %% [markdown]
# Como a base da competição não fornece um conjunto de validação separado, foi realizada uma divisão de 80% para treinamento e 20% para validação usando train_test_split.

# %%
#TF-IDF

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

print(X_train.shape)
print(X_train_tfidf.shape)
print(X_val.shape)
print(X_val_tfidf.shape)

# %% [markdown]
# Os comentários foram transformados usando TF-IDF. O vocabulário aprendido no conjunto de treinamento gerou 24698 features textuais. O conjunto de validação usou o mesmo vocabulário por meio da operação "transform", evitando vazamento de informação entre as etapas de treino e avlaiação.

# %% [markdown]
# **Logistic Regression**

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

model = OneVsRestClassifier(
    LogisticRegression(
        max_iter=1000
    ))

model.fit(
    X_train_tfidf,
    y_train
);

# %%
y_pred = model.predict(X_val_tfidf)


# %%
y_pred.shape

# %%
#avaliação do modelo - Métrica F1 Macro
from sklearn.metrics import f1_score

f1_macro = f1_score(
    y_val,
    y_pred,
    average="macro"
)

print(f"F1 Macro: {f1_macro:.4f}")

# %%
#Análise do desempenho por emoção

from sklearn.metrics import classification_report

print(
    classification_report(
        y_val,
        y_pred,
        target_names=emotion_columns,
        zero_division=0
    )
)


# %% [markdown]
# O modelo TF-IDF e regrassão logística obtiveram F1 macro = 0,1301 no conjunto de validação.
#
# As emoções com menor quantidade de exemplos apresentaram os piores resultados. Classes como grief, relief e nervousness obtiveram F1 = 0, indicando que o modelo não conseguiu identifica-las adequadamente. esse comportamento está de acordo com o forte desbalanceamento observado durante a análies exploratória.

# %%
feature_names = vectorizer.get_feature_names_out()

len(feature_names)

# %%
#Interpretando as palavras com maior impacto na emoção

emotion_idx = list(emotion_columns).index("gratitude")

classifier = model.estimators_[emotion_idx]

coefficients = classifier.coef_[0]

top_positive_idx = coefficients.argsort()[-10:][::-1]

for idx in top_positive_idx:
    print(
        feature_names[idx],
        round(coefficients[idx], 3)
    )

# %% [markdown]
# Para entender o comportamento do modelo, foram analisados os coeficientes da regressão logística. Analisando a emoção gratitude, as palavras "thanks", "thank", "welcome", "congrats", "congratulations".. apareceram com maior impacto positivo. Isso demonstra que o modelo apresentou assosiações coerentes entre termos de agradecimento e emoção correspondente.

# %%
emotion_idx = list(emotion_columns).index("anger")

classifier = model.estimators_[emotion_idx]

coefficients = classifier.coef_[0]

top_positive_idx = coefficients.argsort()[-10:][::-1]

for idx in top_positive_idx:
    print(
        feature_names[idx],
        round(coefficients[idx], 3)
    )

# %% [markdown]
# Analisando as palavras relacionadas a anger, também podemos observar que assim como gratitude, indicam que o modelo foi capaz de capturar padrões de palavras compatíveis com as emoções.

# %%
emotion_idx = list(emotion_columns).index("grief")

classifier = model.estimators_[emotion_idx]

coefficients = classifier.coef_[0]

top_positive_idx = coefficients.argsort()[-10:][::-1]

for idx in top_positive_idx:
    print(
        feature_names[idx],
        round(coefficients[idx], 3)
    )

# %% [markdown]
# Para a emoção grief, que temos menos exemplos e que possui F1 = 0, a análise dos coeficientes de regressão logística mostram que  modelo apresentou palavras semanticamente associadas com conceitos de luto. Isso sugere que o baixo desempenho está mais relacionado à escassez de exemplos de treniamento do que à incapacidade do modelo em identificar padrões relevnates.
