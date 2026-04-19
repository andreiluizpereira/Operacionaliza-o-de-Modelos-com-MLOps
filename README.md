# Projeto de Machine Learning - Adult Census Income

Este projeto foi refatorado para atender a estrutura basica de operacionalizacao de ML pedida na disciplina:

1. Ingestao de dados
2. Validacao de qualidade
3. Pre-processamento
4. Treino
5. Avaliacao

O objetivo e prever se a renda anual do individuo e maior que 50K.

## Metricas de negocio e escolha do modelo

Neste problema, a metrica de negocio principal nao e apenas a acuracia, porque o dataset tem classes desbalanceadas e o custo dos erros nao e simetrico.

- **Falso positivo**: prever renda `> 50K` para quem nao atinge esse patamar pode levar a uma decisao de negocio indevida.
- **Falso negativo**: classificar como `<= 50K` alguem que deveria estar na classe `> 50K` faz o modelo perder casos relevantes.

Por isso, a **metrica de referencia para escolha do modelo e o `F1-macro`**. Essa escolha e mais adequada do que a acuracia porque:

1. equilibra precision e recall;
2. avalia as duas classes com o mesmo peso, mesmo com desbalanceamento;
3. reduz o risco de selecionar um modelo que apenas aproveite a classe majoritaria.

No dataset deste projeto, a classe `<= 50K` e majoritaria, ha marcadores de ausente como `?` em variaveis categoricas e existem relacoes nao lineares entre as features. Esses fatores tornam a acuracia uma medida fraca para comparar modelos, porque ela pode parecer alta mesmo quando o modelo erra sistematicamente a classe minoritaria. O `F1-macro` captura melhor esse comportamento e, por isso, foi adotado como criterio principal de selecao.

Como apoio a analise, tambem vale observar:

- recall da classe `> 50K`, para medir quantos casos relevantes o modelo recupera;
- precision da classe `> 50K`, para evitar muitas previsoes positivas incorretas;
- matriz de confusao, para enxergar o tipo de erro cometido.

## Reducao de dimensionalidade

O projeto compara duas variantes de features dentro do pipeline:

- `baseline`: pre-processamento com scaling e one-hot encoding, sem reducao;
- `pca_95pct`: mesma preparacao, seguida de `PCA` configurado para preservar 95% da variancia.

Essa etapa foi implementada em `src/preprocessing.py` e parametrizada em `configs/preprocessing.yaml`. O treino em `src/train.py` executa os modelos nas duas variantes e registra a comparacao no `outputs/reports/model_comparison.csv`.

No experimento atual:

- o baseline preservou `41` features transformadas e obteve melhor desempenho;
- o `PCA` reduziu o espaco para `1` componente principal e explicou cerca de `99.7%` da variancia;
- apesar da compressao, a versao com reducao teve queda no `F1-macro`, entao o modelo final selecionado continuou sendo o baseline.

Na pratica, isso mostra que a reducao de dimensionalidade foi testada de forma correta, mas neste dataset ela nao superou a representacao original para a metrica de selecao escolhida.

## Como executar

```bash
python main.py
```

## Streamlit

A interface usa os artefatos gerados pelo pipeline:

- `artifacts/best_model.joblib`
- `outputs/reports/evaluation_report.json`
- `outputs/quality/quality_report.json`
- `data/processed/adult_income_clean.parquet`

Instale as dependencias e abra a aplicacao:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Se os artefatos ainda nao existirem, rode `python main.py` antes do comando do Streamlit.

Para habilitar o rastreamento de experimentos, confirme que o pacote `mlflow` esta instalado no ambiente:

```bash
pip install mlflow
```

O tracking esta configurado em `configs/modeling.yaml` no bloco `modeling.mlflow`.

## MLflow

Depois de executar o pipeline, abra a UI local:

```bash
mlflow ui --backend-store-uri mlruns
```

A interface fica disponivel em `http://localhost:5000`.

## Saidas geradas

- `data/interim/adult_income_ingested.parquet`
- `data/processed/adult_income_clean.parquet`
- `outputs/quality/quality_report.json`
- `outputs/reports/model_comparison.csv`
- `outputs/reports/training_summary.json`
- `outputs/reports/evaluation_report.json`
- `artifacts/best_model.joblib`
- `mlruns/`

## Observacao

O modelo selecionado neste fluxo foi `decision_tree` sem redução de dimensionalidade, com F1-macro de teste de aproximadamente `0.7864`.
