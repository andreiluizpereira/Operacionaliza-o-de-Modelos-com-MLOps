# Como rodar o projeto

Execute os comandos abaixo na raiz do repositório.

## 1. Criar e ativar um ambiente virtual

No PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

No Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

## 2. Instalar dependências

```bash
pip install -r requirements.txt
```

## 3. Executar o pipeline

```bash
python main.py
```

Rode este comando antes de abrir a aplicação Streamlit, pois ele gera os artefatos usados pela interface.

## 4. Abrir a UI do MLflow

Depois de executar o pipeline:

```bash
mlflow ui --backend-store-uri mlruns
```

Acesse:

```text
http://localhost:5000
```

## 5. Abrir a aplicação Streamlit

```bash
streamlit run streamlit_app.py
```

Acesse a URL exibida pelo Streamlit no terminal. Normalmente:

```text
http://localhost:8501
```

## 6. Reexecutar após mudanças

Depois de alterar arquivos em `configs/` ou dados em `data/raw/`, rode novamente:

```bash
python main.py
```

