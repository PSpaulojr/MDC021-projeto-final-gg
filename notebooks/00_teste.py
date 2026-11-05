# %% [markdown]
# # 00 - Teste do Jupytext
# Cada pessoa edita SOMENTE a sua secao. Mantenham as secoes separadas como estao.

# %%
print("celula compartilhada: nao editar")

# %% [markdown]
# ## Secao Pessoa A

# %%
# Pessoa A: escreva seu codigo aqui
from datasets import load_dataset

dataset = load_dataset("google-research-datasets/go_emotions", "simplified")


# %%
print(dataset)

# %%
print(dataset["train"].num_rows)
print(dataset["validation"].num_rows)
print(dataset["test"].num_rows)

# %%
print(dataset["train"][0])

# %%
# Recuperar os nomes das classes
label_names = dataset["train"].features["labels"].feature.names

# %%
print(label_names[27])

# %% [markdown]
# ---

# %% [markdown]
# ## Secao Pessoa B

# %%
# Pessoa B: escreva seu codigo aqui


# %% [markdown]
# ---

# %% [markdown]
# ## Secao Pessoa C (opcional)

# %%
# Pessoa C: escreva seu codigo aqui
