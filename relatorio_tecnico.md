# Relatório técnico do projeto

## 1. Resumo executivo

Este projeto implementa um fluxo de machine learning para prever se a renda anual de um indivíduo é maior que USD 50.000, usando o dataset Adult Census Income. O problema foi tratado como classificação binária, com foco em uma entrega reprodutível e operacionalizável por pipeline.

A abordagem final selecionada foi o modelo `decision_tree__baseline`, isto é, uma árvore de decisão treinada sobre as features pré-processadas sem redução de dimensionalidade. A seleção foi feita com base no maior `F1-macro` em validação cruzada, critério escolhido por causa do desbalanceamento entre as classes.

O projeto contém pipeline de dados, validação de qualidade, pré-processamento, treino, comparação de experimentos, avaliação final, persistência do modelo e rastreamento com MLflow.

## 2. Contexto do problema

O objetivo técnico é prever a variável `income_>50K`, que indica se a renda anual do indivíduo é maior que USD 50.000. O contexto prático adotado no projeto é apoiar uma decisão binária baseada em dados demográficos e socioeconômicos.

Como a classe `<= 50K` é majoritária, a acurácia isolada pode favorecer modelos que acertam a classe mais comum, mas deixam de identificar corretamente a classe minoritária. Por isso, a métrica principal de seleção foi o `F1-macro`, que pondera o desempenho das duas classes de forma equilibrada.

Métricas auxiliares acompanhadas:

- `accuracy`;
- `precision_macro`;
- `recall_macro`;
- matriz de confusão;
- relatório de classificação por classe.

## 3. Objetivos de engenharia

O projeto foi organizado para atender aos seguintes objetivos:

- separar experimentação de um fluxo executável de entrega;
- criar um pipeline reproduzível com entrada única em `main.py`;
- parametrizar decisões principais por arquivos YAML em `configs/`;
- usar pipelines do scikit-learn para evitar vazamento de dados;
- comparar modelos e variantes de features de forma padronizada;
- registrar métricas, parâmetros e metadados no MLflow;
- salvar o melhor modelo como artefato reutilizável.

## 4. Estrutura do pipeline

O ponto de entrada do projeto é `main.py`, que executa as etapas na seguinte ordem:

1. `src.ingestion.run`: leitura do CSV bruto, tratamento de marcadores `?` como ausentes e salvamento da base ingerida em parquet.
2. `src.validation.run`: verificação de qualidade, incluindo colunas esperadas, ausentes, duplicados, outliers numéricos e distribuição da classe alvo.
3. `src.preprocessing.run`: separação de features e alvo, remoção de colunas descartadas, divisão treino/teste estratificada e salvamento da base processada.
4. `src.train.run`: construção dos pipelines de modelagem, validação cruzada, comparação de modelos, registro no MLflow e persistência do melhor modelo.
5. `src.evaluate.run`: avaliação final no holdout, matriz de confusão e relatório de classificação.

As configurações ficam centralizadas em:

- `configs/pipeline.yaml`: caminhos, nome do pipeline e logging;
- `configs/data.yaml`: nomes dos arquivos, coluna alvo, colunas esperadas e colunas removidas;
- `configs/quality.yaml`: thresholds de qualidade;
- `configs/preprocessing.yaml`: colunas numéricas, categóricas e redução de dimensionalidade;
- `configs/modeling.yaml`: modelos, validação cruzada, holdout, MLflow e nomes dos artefatos.

## 5. Dados e qualidade

A base bruta analisada possui:

- 43.957 linhas;
- 15 colunas;
- alvo `income_>50K`;
- distribuição de classes: 76,07% para classe `0` e 23,93% para classe `1`.

O relatório de qualidade (`outputs/quality/quality_report.json`) indicou:

| Item | Resultado |
|---|---:|
| Linhas | 43.957 |
| Colunas | 15 |
| Duplicados | 45 |
| Percentual de duplicados | 0,1024% |
| Ausentes em `workclass` | 2.498 |
| Ausentes em `occupation` | 2.506 |
| Ausentes em `native-country` | 763 |
| Distribuição da classe positiva | 23,93% |
| Problemas críticos | 0 |
| Status geral | aprovado |

Também foram identificados outliers em `capital-gain` e `capital-loss`. Esses campos foram mantidos, pois valores extremos nessas variáveis podem representar sinais relevantes de renda e patrimônio, não apenas erros de coleta.

As colunas removidas foram:

- `fnlwgt`: peso amostral, pouco adequado para uso direto no modelo final neste fluxo;
- `education`: redundante em relação a `educational-num`;
- `native-country`: removida para reduzir dimensionalidade e complexidade, além de possuir valores ausentes.

## 6. Pré-processamento e controle de vazamento

O pré-processamento foi implementado com `Pipeline` e `ColumnTransformer` do scikit-learn. Essa escolha reduz o risco de vazamento de dados, pois as transformações são ajustadas apenas durante o `fit` dentro de cada fluxo de treino.

Transformações aplicadas:

- variáveis numéricas: `RobustScaler`, escolhido por ser menos sensível a outliers;
- variáveis categóricas: imputação pela categoria mais frequente e `OneHotEncoder`;
- divisão treino/teste: `train_test_split` estratificado;
- proporção de holdout: 30%;
- semente aleatória: 14.

As features numéricas configuradas foram:

- `age`;
- `educational-num`;
- `capital-gain`;
- `capital-loss`;
- `hours-per-week`.

As features categóricas configuradas foram:

- `workclass`;
- `marital-status`;
- `occupation`;
- `relationship`;
- `race`;
- `gender`.

## 7. Redução de dimensionalidade

Para avaliar o impacto de redução de dimensionalidade, o pipeline comparou duas variantes de features:

- `baseline`: pré-processamento sem redução;
- `pca_95pct`: pré-processamento seguido de PCA preservando 95% da variância.

A redução foi implementada dentro do pipeline de features, depois das transformações numéricas e categóricas. Isso evita ajustar o PCA antes da divisão de treino/teste ou fora da validação cruzada.

No experimento atual, o PCA reduziu as 41 features transformadas para 1 componente principal, com aproximadamente 99,7% de variância explicada. Apesar da compressão, houve perda relevante de desempenho, especialmente em `F1-macro`.

## 8. Desenho experimental

Foram comparados três algoritmos em duas variantes de features:

- `perceptron`;
- `decision_tree`;
- `linear_svm`;
- cada um nas variantes `baseline` e `pca_95pct`.

Configuração experimental:

- validação cruzada estratificada com 10 folds;
- `shuffle=true`;
- `random_state=14`;
- métrica de ordenação: `cv_f1_macro`;
- avaliação complementar no holdout;
- rastreamento de métricas e parâmetros no MLflow.

A árvore de decisão recebeu ajuste de hiperparâmetros via `RandomizedSearchCV`, com busca sobre:

- `criterion`;
- `max_depth`;
- `min_samples_leaf`;
- `class_weight`.

Perceptron e Linear SVM foram avaliados com parâmetros fixos definidos em configuração, funcionando como baselines adicionais de comparação.

## 9. Resultados comparativos

Os resultados abaixo foram consolidados a partir de `outputs/reports/model_comparison.csv` e `outputs/reports/training_summary.json`.

| Posição | Modelo | Variante | CV F1-macro | Holdout F1-macro | Accuracy holdout | Tempo treino (s) | Features finais |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | decision_tree | baseline | 0,7937 | 0,7864 | 0,8559 | 32,93 | 41 |
| 2 | linear_svm | baseline | 0,7763 | 0,7697 | 0,8472 | 1,72 | 41 |
| 3 | perceptron | baseline | 0,7338 | 0,7172 | 0,7950 | 2,03 | 41 |
| 4 | decision_tree | pca_95pct | 0,6890 | 0,7163 | 0,8094 | 16,75 | 1 |
| 5 | linear_svm | pca_95pct | 0,5984 | 0,6010 | 0,8015 | 1,04 | 1 |
| 6 | perceptron | pca_95pct | 0,5959 | 0,5915 | 0,7818 | 0,96 | 1 |

O melhor resultado em validação cruzada foi obtido pelo `decision_tree__baseline`, com:

- `cv_f1_macro`: 0,7937;
- desvio-padrão do CV: 0,0085;
- `holdout_f1_macro`: 0,7864;
- `holdout_accuracy`: 0,8559.

Os melhores hiperparâmetros encontrados para a árvore de decisão foram:

- `criterion`: `gini`;
- `max_depth`: 18;
- `min_samples_leaf`: 41;
- `class_weight`: `None`.

## 10. Análise dos resultados

O `decision_tree__baseline` apresentou o melhor equilíbrio entre desempenho em validação cruzada e desempenho no holdout. A diferença entre `cv_f1_macro` e `holdout_f1_macro` foi pequena, o que sugere comportamento relativamente consistente entre validação e teste.

O `linear_svm__baseline` foi competitivo e treinou mais rapidamente, mas ficou abaixo da árvore de decisão na métrica principal. Como o critério de seleção definido foi `F1-macro`, a redução de tempo não compensou a perda de desempenho.

O `perceptron__baseline` teve desempenho inferior aos demais modelos baseline, indicando que uma fronteira linear simples não capturou tão bem as relações entre as variáveis.

As variantes com PCA reduziram o número de features de 41 para 1 componente, diminuindo custo e complexidade da representação. Porém, a redução afetou negativamente o `F1-macro`, principalmente nos modelos lineares. Além disso, o PCA reduz a interpretabilidade direta das features, pois as variáveis originais deixam de aparecer diretamente na entrada do estimador final.

Assim, apesar de o PCA ter cumprido o objetivo de compressão, ele não foi adequado como abordagem final para este dataset e para a métrica escolhida.

## 11. Avaliação final do modelo escolhido

O modelo final selecionado foi `decision_tree__baseline`. No holdout, os resultados foram:

| Métrica | Valor |
|---|---:|
| Accuracy | 0,8559 |
| Precision macro | 0,8156 |
| Recall macro | 0,7670 |
| F1-macro | 0,7864 |

Matriz de confusão:

| Classe real / prevista | Previsto 0 | Previsto 1 |
|---|---:|---:|
| Real 0 | 9.406 | 626 |
| Real 1 | 1.274 | 1.882 |

Relatório por classe:

| Classe | Precision | Recall | F1-score | Suporte |
|---|---:|---:|---:|---:|
| 0 | 0,8807 | 0,9376 | 0,9083 | 10.032 |
| 1 | 0,7504 | 0,5963 | 0,6645 | 3.156 |

O modelo apresenta bom desempenho geral, mas ainda possui uma limitação importante: o recall da classe positiva é 0,5963. Isso significa que parte relevante dos indivíduos com renda `> 50K` ainda é classificada como `<= 50K`. Essa limitação deve ser considerada caso o objetivo de negócio priorize recuperar mais casos positivos.

## 12. Justificativa da abordagem final

A abordagem final escolhida foi a árvore de decisão com features baseline, pelos seguintes motivos:

- melhor `F1-macro` médio em validação cruzada;
- melhor `F1-macro` no holdout entre as variantes avaliadas;
- desempenho estável entre folds, com desvio-padrão baixo;
- preservação da interpretabilidade direta das features transformadas;
- integração completa em pipeline scikit-learn;
- persistência do pipeline completo em `artifacts/best_model.joblib`;
- registro de parâmetros e métricas no MLflow.

A variante com PCA não foi escolhida porque, apesar de reduzir dimensionalidade e tempo de treino, gerou perda de desempenho e reduziu interpretabilidade.

O Linear SVM também não foi escolhido, embora tenha sido mais rápido, porque ficou abaixo da árvore de decisão na métrica principal definida para o projeto.

## 13. MLflow e rastreabilidade

O MLflow está habilitado em `configs/modeling.yaml` com:

- `enabled: true`;
- `experiment_name: adult-income-experiments`;
- `tracking_uri: mlruns`;
- `run_name_prefix: baseline`;
- `log_model: true`.

O treino registra:

- parâmetros do modelo;
- métrica `cv_f1_macro`;
- desvio-padrão da validação cruzada;
- score de cada fold;
- métricas do holdout;
- tempo de treino;
- metadados de redução de dimensionalidade;
- quantidade de features antes e depois da redução.
- artefato do pipeline scikit-learn treinado em cada run, incluindo o melhor modelo selecionado.

A UI local pode ser aberta com:

```bash
mlflow ui --backend-store-uri mlruns
```

O projeto também salva os resultados em arquivos locais:

- `outputs/reports/model_comparison.csv`;
- `outputs/reports/training_summary.json`;
- `outputs/reports/evaluation_report.json`;
- `outputs/quality/quality_report.json`;
- `artifacts/best_model.joblib`.

Observação: a configuração atual usa `log_model: true`, portanto o pipeline scikit-learn treinado também é registrado como artefato de modelo no MLflow. O melhor modelo continua salvo localmente em `artifacts/best_model.joblib` para consumo direto pela aplicação.

## 14. Reprodutibilidade

Para executar o pipeline completo:

```bash
python main.py
```

Para instalar dependências:

```bash
pip install -r requirements.txt
```

Para abrir a interface Streamlit, depois de gerar os artefatos:

```bash
streamlit run streamlit_app.py
```

O pipeline foi executado com sucesso e gerou:

- dados ingeridos em `data/interim/adult_income_ingested.parquet`;
- dados processados em `data/processed/adult_income_clean.parquet`;
- relatório de qualidade;
- comparação de modelos;
- resumo de treino;
- relatório de avaliação;
- modelo final serializado;
- runs no MLflow.

## 15. Conclusão

O projeto atende ao entregável técnico de repositório organizado com pipeline de dados e modelos, código de experimentação e configuração de MLflow. Também possui resultados experimentais suficientes para justificar a abordagem final.

A abordagem final escolhida, `decision_tree__baseline`, foi selecionada por apresentar o melhor `F1-macro` entre os experimentos avaliados, mantendo interpretabilidade e integração direta com o pipeline scikit-learn. A redução de dimensionalidade via PCA foi testada, mas não foi adotada porque reduziu o desempenho da métrica principal.

Com este relatório, o segundo entregável também fica consolidado em um documento técnico estruturado, contendo decisões de projeto, análise comparativa de experimentos e justificativa da abordagem final.
