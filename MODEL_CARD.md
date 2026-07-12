# Model Card — Sistema de Recomendação de Filmes

> Documento de transparência do modelo desenvolvido para o Tech Challenge — Fase 2  
> FIAP Pós-Tech em Machine Learning Engineering

---

## Índice

- [Visão Geral do Modelo](#visão-geral-do-modelo)
- [Uso Pretendido](#uso-pretendido)
- [Dataset](#dataset)
- [Arquitetura e Treinamento](#arquitetura-e-treinamento)
- [Pipeline de Dados](#pipeline-de-dados)
- [Métricas de Avaliação](#métricas-de-avaliação)
- [Limitações](#limitações)
- [Vieses Identificados](#vieses-identificados)
- [Considerações Éticas](#considerações-éticas)
- [Como Usar o Modelo](#como-usar-o-modelo)
- [Rastreamento de Experimentos](#rastreamento-de-experimentos)
- [Informações do Projeto](#informações-do-projeto)

---

## Visão Geral do Modelo

| Atributo | Descrição |
|---|---|
| **Nome** | Movie Recommendation System — MLP |
| **Versão** | 1.0.0 |
| **Tipo** | Rede Neural Multicamada (MLP) com Embeddings |
| **Framework** | PyTorch |
| **Tarefa** | Filtragem Colaborativa — Recomendação de Filmes (classificação binária: nota ≥ 4 → "gostou") |
| **Dataset** | MovieLens |
| **Status** | Acadêmico — artefatos versionados via DVC e experimentos rastreados no MLflow (Model Registry planejado como próximo passo) |

Este modelo implementa um sistema de recomendação de filmes baseado em aprendizado profundo. A partir do histórico de interações entre usuários e filmes, o modelo aprende representações latentes (embeddings) que capturam padrões de preferência e realiza predições personalizadas de filmes para cada usuário.

---

## Uso Pretendido

### Uso recomendado ✅

- Recomendar filmes a usuários com base no seu histórico de avaliações
- Gerar listas personalizadas dos top-K filmes mais relevantes para um usuário
- Apoiar plataformas de streaming ou catálogos digitais em decisões de curadoria de conteúdo
- Fins acadêmicos e de pesquisa em sistemas de recomendação

### Uso não recomendado ❌

- Recomendações para usuários sem histórico de interações (cold start)
- Ambientes de produção em larga escala sem retreinamento periódico
- Tomada de decisões críticas sem supervisão humana
- Contextos fora do domínio de filmes sem adaptação do modelo

---

## Dataset

### Fonte

**MovieLens** — Conjunto de dados público mantido pelo GroupLens Research Lab da Universidade de Minnesota, amplamente utilizado em pesquisas de sistemas de recomendação.

### Características

| Atributo | Descrição |
|---|---|
| **Domínio** | Avaliações de filmes por usuários |
| **Tipo de interação** | Avaliações explícitas (ratings) |
| **Escala de avaliação** | 0.5 a 5.0 estrelas |
| **Mínimo de avaliações por usuário** | 20 interações |

### Estrutura dos Dados

```
data/
├── raw/           # Dados brutos originais do MovieLens
├── processed/     # Dados após pré-processamento
└── features/      # Atributos engenheirados prontos para treino
```

### Pré-processamento Aplicado

- Binarização das avaliações: `label = 1` se a nota ≥ 4, caso contrário `0` (a tarefa é tratada como classificação binária de "gostou / não gostou")
- Amostragem determinística (seed 42) para até 100 mil interações quando o dataset é maior
- Codificação de usuários e filmes com `LabelEncoder`
- Divisão estratificada por `label` em treino (70%), validação (15%) e teste (15%)
- Persistência dos encoders em `artifacts/encoders/`

---

## Arquitetura e Treinamento

### Arquitetura do Modelo

O modelo é uma rede MLP com camadas de embedding que aprende representações densas de usuários e filmes:

```
Entrada
  ├── Embedding de Usuário  (user_id → vetor denso, dim=64)
  └── Embedding de Filme    (movie_id → vetor denso, dim=64)
        │
        ▼
  Concatenação dos Embeddings (128 dimensões)
        │
        ▼
  Camadas Totalmente Conectadas
  ├── Linear(128 → 128) → ReLU → Dropout(0.2)
  ├── Linear(128 → 64)  → ReLU
  └── Linear(64 → 1)    → logit
        │
        ▼
  Saída: logit único (probabilidade de "gostar")
        │
        ▼
  Sigmoid aplicado na inferência → score em [0, 1]
```

> A perda `BCEWithLogitsLoss` aplica a sigmoid internamente durante o treino;
> por isso a camada final devolve o *logit* e a sigmoid é aplicada apenas na
> inferência/avaliação para obter o score de probabilidade.

### Hiperparâmetros

| Parâmetro | Valor |
|---|---|
| Dimensão dos Embeddings | 64 |
| Camadas ocultas | [128, 64] |
| Dropout | 0.2 |
| Função de perda | BCEWithLogitsLoss |
| Otimizador | Adam |
| Learning rate | 0.001 |
| Batch size | 1024 |
| Épocas máximas | 10 |
| Early stopping (patience) | 3 épocas |
| Seed fixado | 42 |

### Reprodutibilidade

Seeds fixados em `torch`, `numpy` e `random` para garantir reprodutibilidade dos experimentos. O ambiente é gerenciado com `uv` e o `uv.lock` está commitado no repositório.

---

## Pipeline de Dados

O pipeline é gerenciado pelo **DVC** e executado via `dvc repro`:

```
preprocess
    │  Limpeza e normalização dos dados brutos
    ▼
build_features
    │  Engenharia de atributos e codificação de entidades
    ▼
train
    │  Treinamento da rede MLP com PyTorch
    ▼
evaluate
    │  Cálculo de métricas e registro no MLflow
```

Para reproduzir o pipeline completo:

```bash
uv run dvc repro
```

---

## Métricas de Avaliação

O modelo é comparado com dois baselines do Scikit-Learn para validar o ganho obtido com a rede neural.

### Métricas Utilizadas

| Métrica | Descrição |
|---|---|
| **Accuracy** | Proporção de predições corretas |
| **Precision** | Proporção de recomendações relevantes entre as sugeridas |
| **Recall** | Proporção de itens relevantes que foram recuperados |
| **F1 Score** | Média harmônica entre Precision e Recall |

### Comparação com Baselines

| Modelo | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Baseline (`DummyClassifier`, referência) | — | — | — | — |
| **MLP PyTorch (este modelo)** | **0.7007** | **0.6893** | **0.7142** | **0.7015** |

> Métricas calculadas no split de teste com `seed=42` (reprodutíveis). A tarefa
> é uma classificação binária (nota ≥ 4 → "gostou"), o que torna
> Accuracy/Precision/Recall/F1 as métricas apropriadas.


Os resultados de cada experimento estão disponíveis no MLflow Tracking Server e os artefatos em:

```
artifacts/metrics/metrics.json
```

---

## Limitações

### Problema de Cold Start

O modelo não é capaz de gerar recomendações para:
- **Novos usuários** sem histórico de avaliações
- **Novos filmes** que ainda não foram avaliados por nenhum usuário

Nesses casos, uma estratégia de fallback baseada em popularidade ou filtragem baseada em conteúdo deve ser utilizada.

### Dependência de Dados Históricos

A qualidade das recomendações é diretamente proporcional à quantidade de avaliações do usuário. Usuários com poucas interações tendem a receber recomendações de menor qualidade.

### Deriva Temporal (Data Drift)

O modelo foi treinado com dados de um período específico. Mudanças de comportamento dos usuários ao longo do tempo podem degradar a performance — retreinamento periódico é necessário.

### Escopo do Dataset

O modelo foi desenvolvido e validado exclusivamente sobre o dataset MovieLens. Aplicações em outros domínios (séries, música, e-commerce) requerem retreinamento com dados específicos.

### Escala

O modelo atual não foi otimizado para servir recomendações em tempo real a milhões de usuários simultâneos. Para produção em larga escala, são necessárias otimizações de infraestrutura.

---

## Vieses Identificados

### Viés de Popularidade

O modelo tende a recomendar filmes mais avaliados com maior frequência, favorecendo títulos populares em detrimento de produções de nicho ou catálogo longo. Isso pode criar um efeito de câmara de eco nas recomendações.

### Viés de Seleção

O dataset reflete o comportamento de usuários que escolheram avaliar filmes, o que não representa a população geral de espectadores. Usuários mais engajados têm maior influência no aprendizado do modelo.

### Viés Cultural e Demográfico

O MovieLens é predominantemente composto por usuários norte-americanos, o que pode levar a recomendações tendenciosas para filmes de língua inglesa ou produções hollywoodianas, subrepresentando o cinema internacional.

### Viés Temporal

Filmes mais antigos possuem um volume maior de avaliações acumuladas ao longo do tempo, o que pode inflacionar artificialmente sua relevância em comparação com lançamentos recentes.

---

## Considerações Éticas

- O modelo foi desenvolvido exclusivamente com dados públicos e anonimizados
- Nenhuma informação pessoal identificável (PII) é utilizada no treinamento
- O sistema não realiza inferências sobre características sensíveis como gênero, etnia ou localização geográfica
- As recomendações geradas devem ser tratadas como sugestões, não como verdades absolutas
- Recomenda-se auditoria periódica das recomendações para identificar e mitigar vieses emergentes
- O modelo não deve ser utilizado como único critério em decisões que impactem diretamente usuários

---

## Como Usar o Modelo

### Pré-requisitos

```bash
uv sync
```

### Executar o Pipeline Completo

```bash
uv run dvc repro
```

### Iniciar a API de Recomendações

```bash
uv run uvicorn src.api.main:app --reload
```

### Requisitar Recomendações

```bash
curl -X POST http://localhost:8000/api/v1/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "top_k": 10}'
```

### Verificar Saúde da API

```bash
curl http://localhost:8000/api/v1/health
```

### Executar via Docker

```bash
docker-compose up --build
```

---

## Rastreamento de Experimentos

Todos os experimentos são rastreados com **MLflow**, incluindo:

- Hiperparâmetros utilizados em cada run
- Métricas de treino e validação por época (`train_loss` / `val_loss`)
- Métricas de teste (accuracy, precision, recall, F1)
- Artefatos gerados (modelo, encoders, métricas)
- Versionamento no Model Registry (Staging → Production) previsto como evolução futura

Para visualizar os experimentos:

```bash
uv run mlflow ui
```

Acesse: `http://localhost:5000`

---

## Informações do Projeto

| Atributo | Descrição |
|---|---|
| **Instituição** | FIAP — Faculdade de Informática e Administração Paulista |
| **Programa** | Pós-Tech em Machine Learning Engineering |
| **Fase** | 2 — Big Data Architecture |
| **Atividade** | Tech Challenge |
| **Repositório** | [github.com/RM371356/recommendation-system](https://github.com/RM371356/recommendation-system) |
| **Finalidade** | Acadêmica |

---

*Model Card elaborado seguindo as diretrizes de boas práticas de documentação de modelos de ML.*  
*Referência: Mitchell et al. (2019) — "Model Cards for Model Reporting"*
