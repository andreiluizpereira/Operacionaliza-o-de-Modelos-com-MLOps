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

Para atender a rubrica de reducao de dimensionalidade, o projeto agora compara duas variantes de features dentro do pipeline:

- `baseline`: pre-processamento com scaling e one-hot encoding, sem reducao;
- `pca_95pct`: mesma preparacao, seguida de `PCA` configurado para preservar 95% da variancia.

Essa etapa foi implementada em `src/preprocessing.py` e parametrizada em `configs/preprocessing.yaml`. O treino em `src/train.py` executa os modelos nas duas variantes e registra a comparacao no `outputs/reports/model_comparison.csv`.

No experimento atual:

- o baseline preservou `41` features transformadas e obteve melhor desempenho;
- o `PCA` reduziu o espaco para `1` componente principal e explicou cerca de `99.7%` da variancia;
- apesar da compressao, a versao com reducao teve queda no `F1-macro`, entao o modelo final selecionado continuou sendo o baseline.

Na pratica, isso mostra que a reducao de dimensionalidade foi testada de forma correta, mas neste dataset ela nao superou a representacao original para a metrica de selecao escolhida.

## Estrutura

- `src/ingestion.py`: le o CSV bruto e salva uma copia em parquet.
- `src/validation.py`: executa checagens simples de qualidade e gera relatorio JSON.
- `src/preprocessing.py`: remove colunas, separa treino/teste, define o pre-processador e opcionalmente aplica reducao de dimensionalidade.
- `src/train.py`: treina Perceptron, Decision Tree e Linear SVM nas variantes baseline e reduzida.
- `src/evaluate.py`: gera o relatorio final do melhor modelo e registra a variante escolhida.
- `configs/*.yaml`: centralizam caminhos, colunas, regras de qualidade e parametros de treino.
- `main.py`: orquestra o fluxo completo.

## Como executar

```bash
python main.py
```

## Saidas geradas

- `data/interim/adult_income_ingested.parquet`
- `data/processed/adult_income_clean.parquet`
- `outputs/quality/quality_report.json`
- `outputs/reports/model_comparison.csv`
- `outputs/reports/training_summary.json`
- `outputs/reports/evaluation_report.json`
- `artifacts/best_model.joblib`

## Observacao

O modelo selecionado neste fluxo foi `decision_tree`, com F1-macro de teste de aproximadamente `0.7864`.
