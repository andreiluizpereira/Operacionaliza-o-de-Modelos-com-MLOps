# Projeto de Disciplina — Referências por Aula

> Cada instrução do projeto está anotada com a data e o timestamp da aula em que o professor abordou o conceito correspondente.  
> Formato: `📅 DD-MM-AAAA ⏱ HH:MM:SS`

---

## Parte 1 — Estruturação do Projeto de Machine Learning

- Mapear os experimentos já realizados, identificando modelos testados, métricas utilizadas, principais resultados e limitações. `📅 06-04-2026 ⏱ 02:00:25`

- Definir de forma explícita o objetivo técnico do projeto, critérios de sucesso e métricas de negócio associadas. `📅 16-03-2026 ⏱ 00:44:20`

- Reestruturar o projeto em código, garantindo que a lógica principal de preparação, treinamento e validação esteja em scripts ou módulos reutilizáveis, e que notebooks sejam usados apenas para exploração ou visualização. `📅 16-03-2026 ⏱ 01:39:10`

- Analisar os dados sob a ótica de engenharia, apontando riscos iniciais de qualidade, viés e generalização. `📅 16-03-2026 ⏱ 01:49:23`

---

## Parte 2 — Fundação de Dados e Diagnóstico Inicial

- Estruturar a ingestão de dados, definindo fontes, formatos e estratégias de amostragem. `📅 18-03-2026 ⏱ 00:19:39`

- Diagnosticar problemas de qualidade de dados, como valores ausentes, ruído, inconsistências e possíveis vieses. `📅 16-03-2026 ⏱ 01:49:23` / `📅 18-03-2026 ⏱ 00:20:09`

- Analisar o impacto desses problemas na generalização, estabilidade dos resultados e risco de overfitting. `📅 16-03-2026 ⏱ 00:25:45`

- Documentar limitações estruturais do dataset que não possam ser corrigidas apenas com modelagem. `📅 18-03-2026 ⏱ 00:08:34`

---

## Parte 3 — Experimentação Sistemática de Modelos

- Executar experimentos comparativos entre abordagens candidatas já exploradas no projeto anterior. `📅 16-03-2026 ⏱ 00:35:03`

- Selecionar modelos considerando desempenho preditivo, custo computacional, complexidade e interpretabilidade. `📅 16-03-2026 ⏱ 00:43:40`

- Construir pipelines end-to-end de preparação de dados, treinamento e validação utilizando scikit-learn. `📅 16-03-2026 ⏱ 01:39:42` / `📅 07-04-2026 ⏱ 01:55:06`

- Ajustar modelos com validação cruzada e busca de hiperparâmetros. `📅 16-03-2026 ⏱ 01:58:00` / `📅 07-04-2026 ⏱ 01:53:38`

- Registrar todos os experimentos no MLflow, incluindo parâmetros, métricas e versões de modelos. `📅 07-04-2026 ⏱ 01:08:16` / `📅 07-04-2026 ⏱ 01:36:53`

---

## Parte 4 — Controle de Complexidade e Redução de Dimensionalidade

- Analisar a necessidade de redução de dimensionalidade com base nos resultados experimentais obtidos anteriormente. `📅 30-03-2026 ⏱ 00:06:02` / `📅 16-03-2026 ⏱ 01:51:24`

- Escolher e aplicar duas técnicas de redução de dimensionalidade, dentre PCA, LDA e t-SNE, justificando explicitamente a escolha de cada uma em função das características dos dados e do objetivo do modelo. `📅 30-03-2026 ⏱ 00:42:13` / `📅 30-03-2026 ⏱ 01:39:50`

- Integrar a redução de dimensionalidade ao pipeline de modelagem e treinar novamente os classificadores. `📅 30-03-2026 ⏱ 01:22:26`

- Comparar o desempenho dos modelos com e sem redução de dimensionalidade, analisando impacto no resultado final da classificação, custo computacional e efeitos sobre a interpretabilidade. `📅 16-03-2026 ⏱ 01:51:42`

- Discutir os trade-offs observados e justificar se a redução de dimensionalidade é ou não adequada ao contexto do problema. `📅 16-03-2026 ⏱ 01:51:42` / `📅 30-03-2026 ⏱ 01:48:32`

---

## Parte 5 — Consolidação Experimental e Seleção Final

- Analisar comparativamente os experimentos registrados no MLflow. `📅 07-04-2026 ⏱ 01:57:28`

- Justificar a seleção da abordagem final com base em métricas técnicas, custo computacional, complexidade e viabilidade de operação. `📅 07-04-2026 ⏱ 02:02:13`

- Definir explicitamente o modelo candidato à operação. `📅 06-04-2026 ⏱ 02:00:39`

---

## Parte 6 — Operacionalização e Simulação de Produção

- Persistir modelos treinados em scikit-learn de forma versionada. `📅 07-04-2026 ⏱ 00:24:35`

- Executar inferência consistente a partir de modelos persistidos. `📅 18-03-2026 ⏱ 00:56:34`

- Empacotar modelos como artefatos de inferência. `📅 16-03-2026 ⏱ 01:44:50` / `📅 07-04-2026 ⏱ 00:24:40`

- Expor o modelo por meio de um serviço simples de inferência. `📅 16-03-2026 ⏱ 00:30:24` / `📅 16-03-2026 ⏱ 01:44:50`

- Integrar o deploy do modelo a um pipeline de CI/CD simulado ou real. `📅 16-03-2026 ⏱ 01:44:06`

- Definir métricas técnicas do modelo e métricas de impacto de negócio. `📅 16-03-2026 ⏱ 00:35:03`

- Detectar drift de dados e de modelo por meio de comparação estatística. `📅 18-03-2026 ⏱ 00:10:51`

- Monitorar métricas e versões no MLflow. `📅 07-04-2026 ⏱ 01:37:34`

- Planejar estratégias de re-treinamento e aprendizado contínuo. `📅 18-03-2026 ⏱ 00:13:46`
