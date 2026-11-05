# Guia: Jupytext

Só o `.py` vai pro Git. O `.ipynb` é local de cada pessoa e fica no `.gitignore`.
O Jupytext mantém os dois sincronizados.

## Setup (uma vez por pessoa)

```bash
git clone <url-do-repo>
cd <nome-do-repo>

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install jupytext jupyterlab

jupytext --sync notebooks/*.py     # gera os .ipynb locais
```

O arquivo `jupytext.toml` já configura o pareamento. Não precisa fazer mais nada.

## Antes de modificar

```bash
git pull
jupytext --sync notebooks/*.py     # atualiza seu .ipynb com o que veio dos colegas
```

Depois abra o notebook:

- **Jupyter Lab:** abra o `.ipynb` normalmente. Ou clique direito no `.py` → *Open With → Notebook*.
- **VS Code:** abra o `.py` direto e use *Run Cell* nas células `# %%`. Nesse caso nem precisa do `.ipynb`.

## Depois de modificar

1. Salve o notebook (o par é sincronizado ao salvar).
2. Garanta que está tudo sincronizado:
   ```bash
   jupytext --sync notebooks/*.py
   ```
3. Confira o que mudou. Devem aparecer **só arquivos `.py`**, nunca `.ipynb`:
   ```bash
   git status
   ```
4. Commit e push:
   ```bash
   git add notebooks/*.py
   git commit -m "descreva o que mudou"
   git push
   ```

## Regras rápidas

- Edite apenas a sua seção/arquivo, para evitar conflito.
- Nunca faça `git add` de um `.ipynb`.
- Notebook novo: crie um `.py` com células `# %%` (código) e `# %% [markdown]` (texto) dentro de `notebooks/`.
