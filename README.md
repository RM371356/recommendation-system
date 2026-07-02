# README

> Sistema de recomendação de filmes desenvolvido com **Python**, **PyTorch**, **FastAPI**, **MLflow**, **DVC** e **uv**.

---

# Visão Geral

Este projeto implementa um sistema de recomendação de filmes baseado em Machine Learning utilizando o dataset MovieLens.

## Principais funcionalidades

* Pipeline de treinamento
* Engenharia de atributos
* Modelo MLP em PyTorch
* API REST com FastAPI
* Rastreamento de experimentos com MLflow
* Pipeline com DVC
* Estrutura modular para inferência

---

# Estrutura do Projeto

```text
src/
├── api/
├── config/
├── data/
├── evaluation/
├── features/
├── inference/
├── models/
├── services/
├── training/
└── utils/
```

---

# Tecnologias

* Python 3.12+
* uv
* PyTorch
* FastAPI
* Uvicorn
* Pandas
* Scikit-learn
* MLflow
* DVC

---

# Instalação

```bash
uv sync
```

---

# Treinamento

```bash
uv run python -m src.training.train
```

---

# Executando a API

```bash
uv run uvicorn src.api.main:app --reload
```

Swagger:

```text
http://localhost:8000/docs
```

---

# Endpoints

## Health

```http
GET /api/v1/health
```

## Recommendation

```http
POST /api/v1/recommend
```

Exemplo:

```json
{
  "user_id": 1,
  "top_k": 10
}
```

Resposta esperada:

```json
{
  "user_id": 1,
  "total": 10,
  "recommendations": []
}
```

---

# Pipeline

```text
Raw Data
    ↓
Preprocess
    ↓
Feature Engineering
    ↓
Training
    ↓
Artifacts
    ↓
Inference
    ↓
FastAPI
```

---

# Roadmap

* Persistir encoders
* Checkpoint completo do modelo
* Camada de inferência profissional
* Docker
* Testes automatizados
* CI/CD

---
---

# DVC (Data Version Control)

O projeto utiliza o **DVC** para gerenciar o pipeline de Machine Learning, permitindo reproduzir experimentos, controlar versões dos dados e automatizar as etapas de processamento.

## Benefícios

* Versionamento de datasets
* Reprodutibilidade dos experimentos
* Execução automática do pipeline
* Controle das dependências entre etapas
* Integração com Git
* Suporte a armazenamento remoto (S3, Azure Blob Storage, Google Cloud Storage, SSH, etc.)

---

## Estrutura do Pipeline

```text
Dataset
    │
    ▼
Preprocess
    │
    ▼
Feature Engineering
    │
    ▼
Training
    │
    ▼
Evaluation
    │
    ▼
Artifacts
```

---

## Inicializando o DVC

Caso o projeto ainda não esteja inicializado:

```bash
uv run dvc init
```

---

## Executando o Pipeline

Executa automaticamente todas as etapas necessárias:

```bash
uv run dvc repro
```

---

## Executando apenas um estágio

```bash
uv run dvc repro train
```

---

## Verificando o Pipeline

```bash
uv run dvc dag
```

Exemplo:

```text
preprocess
      │
      ▼
build_features
      │
      ▼
train
      │
      ▼
evaluate
```

---

## Verificando alterações

```bash
uv run dvc status
```

---

## Forçando uma nova execução

```bash
uv run dvc repro --force
```

---

## Arquivo dvc.yaml

O pipeline é definido no arquivo:

```text
dvc.yaml
```

Exemplo simplificado:

```yaml
stages:

  preprocess:
    cmd: uv run python -m src.data.preprocess

  build_features:
    cmd: uv run python -m src.features.build_features

  train:
    cmd: uv run python -m src.training.train

  evaluate:
    cmd: uv run python -m src.evaluation.evaluate
```

---

## Artefatos Esperados

Após o treinamento, o pipeline deverá produzir:

```text
artifacts/
├── model/
│   └── best_model.pt
│
├── metadata/
│   ├── movies.csv
│   ├── features.csv
│   └── config.json
│
├── encoders/
│   ├── user_encoder.pkl
│   └── movie_encoder.pkl
│
└── metrics/
    └── metrics.json
```

---

# Ruff

O projeto utiliza o **Ruff** como ferramenta de linting e formatação de código.

O Ruff foi desenvolvido para ser extremamente rápido e substituir diversas ferramentas tradicionais, como:

* Flake8
* pycodestyle
* pyflakes
* isort
* autoflake
* parte das funcionalidades do pylint

---

## Benefícios

* Alta performance
* Código padronizado
* Identificação automática de problemas
* Organização dos imports
* Correção automática de diversos erros

---

## Executando o Ruff

Verificar problemas:

```bash
uv run ruff check .
```

Corrigir automaticamente:

```bash
uv run ruff check . --fix
```

Formatar todo o projeto:

```bash
uv run ruff format .
```

Formatar apenas um diretório:

```bash
uv run ruff format src/
```

---

## Fluxo recomendado antes de realizar um commit

```bash
uv run ruff check . --fix

uv run ruff format .

pytest
```

---

## Configuração

O Ruff é configurado diretamente no arquivo:

```text
pyproject.toml
```

Exemplo:

```toml
[tool.ruff]

line-length = 88

target-version = "py312"

exclude = [
    ".venv",
    "__pycache__",
    "build",
    "dist"
]

[tool.ruff.lint]

select = [
    "E",
    "F",
    "I",
    "UP",
    "B",
]

ignore = []
```

---

## Integração com o Fluxo de Desenvolvimento

Durante o desenvolvimento, recomenda-se a seguinte sequência:

```text
Editar código
        │
        ▼
Ruff Check
        │
        ▼
Ruff Format
        │
        ▼
Pytest
        │
        ▼
DVC Repro
        │
        ▼
MLflow
        │
        ▼
Git Commit
```

Essa abordagem garante que o código permaneça padronizado, que o pipeline de Machine Learning seja reproduzível e que todos os experimentos fiquem devidamente registrados.

# Licença

Projeto desenvolvido para fins acadêmicos (Tech Challenge).

Projeto de Recomendação da FIAP