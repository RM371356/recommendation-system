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

## Opção 1 — Ambiente completo com `uv` (recomendada)

Instala todas as dependências (runtime + dev) exatamente como estão fixadas no `uv.lock`:

```bash
uv sync
```

## Opção 2 — Instalação via `pip` (fallback)

Caso o `uv` não esteja disponível na máquina, é possível instalar tudo com `pip` a partir do `pyproject.toml`:

```bash
python -m venv .venv
.venv\Scripts\activate            # PowerShell / CMD (Windows)
# source .venv/bin/activate       # Linux / macOS

pip install -e ".[dev]"
```

## Dependências necessárias para executar os testes

A suíte de testes foi projetada para funcionar com o mesmo conjunto de dependências do projeto. Nenhuma dependência adicional foi adicionada ao `pyproject.toml`. Se você optar por instalar manualmente, o mínimo requerido é:

**Runtime (obrigatórias):**

* `pandas>=2.2.2`
* `numpy>=2.1.0`
* `scikit-learn>=1.5.1`
* `torch>=2.4.0`
* `mlflow>=2.16.0`
* `pydantic>=2.8.2`
* `pydantic-settings>=2.4.0`
* `python-dotenv>=1.0.1`
* `tqdm>=4.66.5`
* `fastapi` (para os endpoints REST)
* `httpx` (necessário para o `TestClient` do FastAPI)

**Desenvolvimento / testes:**

* `pytest>=8.3.2`
* `pytest-cov>=5.0.0`
* `ruff>=0.6.4`

Instalação manual mínima (fora do `uv`) para rodar apenas os testes:

```bash
pip install pytest pytest-cov httpx pydantic-settings python-dotenv \
            pandas numpy scikit-learn torch mlflow tqdm fastapi
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
* Testes automatizados ✔ (ver seção **Testes automatizados**)
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

---

# Testes automatizados

O projeto conta com uma suíte de **143 testes** distribuídos em **8 categorias**, com foco em cobertura funcional, de contrato, segurança e resiliência da API de recomendação. Toda a suíte roda em uma máquina limpa (sem exigir artefatos treinados) em cerca de **6 segundos**.

## Estrutura da suíte

```text
tests/
├── conftest.py                        # Fixtures compartilhadas
├── smoke/                             # Smoke tests (build sanity)
├── functional/                        # Regras de negócio isoladas
├── api/                               # Contrato REST + schemas Pydantic
├── integration/                       # Fluxo end-to-end da inferência
├── security/                          # Vulnerabilidades / ataques simulados
├── resilience/                        # Fallback e tratamento de erros
├── data/                              # Qualidade e integridade dos dados
└── performance/                       # Latência (soft budgets)
```

## Categorias e cobertura

| Categoria       | Marker         | O que valida |
| --------------- | -------------- | ------------ |
| **Smoke**       | `smoke`        | Imports do pacote `src`, boot do FastAPI, OpenAPI disponível |
| **Funcional**   | `functional`   | `RatingBinarizer`, `RecommendationMetrics`, `MLPRecommender`, `ModelFactory`, `EarlyStopping`, `negative_sampling`, `set_seed`, `InferenceCache`, `RatingsDataset` |
| **API**         | `api`          | Contrato dos schemas Pydantic (`RecommendationRequest/Response`, `HealthResponse`) e formato das rotas REST |
| **Integração**  | `integration`  | Pipeline `ModelLoader → EncoderLoader → MovieRepository → Predictor → RecommendationService` com artefatos fake em `tmp_path` |
| **Segurança**   | `security`     | Injection (SQLi, XSS, Log4Shell), path traversal, boundary attacks, tipos maliciosos, JSON malformado, HTTP verbs indevidos, information disclosure |
| **Resiliência** | `resilience`   | Comportamento com artefatos ausentes, `FileNotFoundError`, cold-start, exception handlers |
| **Dados**       | `data`         | Schema dos splits (`train.csv`, `val.csv`, `test.csv`), ausência de nulos, labels binárias, ratings em `[0.5, 5.0]`, ausência de vazamento entre splits |
| **Performance** | `performance`  | Soft budgets de latência (`/health` < 500ms, `/recommend` < 2s, forward do modelo < 500ms) |

## Executando os testes

Rodar a suíte completa:

```bash
uv run pytest
```

ou, sem `uv`:

```bash
pytest
```

Rodar apenas uma categoria via marker:

```bash
pytest -m smoke
pytest -m security
pytest -m "not performance"          # exclui performance (útil em CI rápido)
pytest -m "security or resilience"
```

Rodar apenas os arquivos de um diretório:

```bash
pytest tests/integration
pytest tests/api -v
```

## Cobertura de código

O `pytest-cov` já está incluído nas dependências de desenvolvimento:

```bash
pytest --cov=src --cov-report=term-missing
pytest --cov=src --cov-report=html    # gera htmlcov/index.html
```

## Fixtures principais (`tests/conftest.py`)

| Fixture                    | Escopo   | Descrição |
| -------------------------- | -------- | --------- |
| `_reset_inference_cache`   | autouse  | Zera o `InferenceCache` singleton **e** invalida o `lru_cache` do `get_recommendation_service` antes/depois de cada teste, garantindo determinismo entre casos |
| `api_client`               | function | `TestClient` da FastAPI executando o `lifespan` real (com fallback natural quando não há artefatos) |
| `synthetic_ratings_df`     | function | DataFrame estilo MovieLens determinístico (seed=42) para testes de preprocess |
| `encoded_ratings_df`       | function | DataFrame já com colunas `label`, `user_idx`, `movie_idx` |
| `fake_artifacts_dir`       | function | Gera em `tmp_path` uma pasta `artifacts/` completa (modelo MLP mínimo, `config.json`, encoders `.pkl`, `movies.csv`) permitindo exercitar o fluxo real de inferência sem depender de modelo treinado |

## Markers registrados

Os markers estão declarados em `[tool.pytest.ini_options]` do `pyproject.toml`:

```toml
markers = [
    "smoke: testes de smoke (fumaça) — validação mínima do build",
    "functional: testes unitários das regras de negócio",
    "api: testes de contrato dos endpoints REST e schemas",
    "integration: testes de integração end-to-end da inferência",
    "security: testes de segurança / vulnerabilidade",
    "resilience: testes de resiliência e tratamento de erros",
    "data: testes de qualidade / integridade dos dados",
    "performance: testes de performance (soft budgets)",
]
```

## Fluxo recomendado antes do commit

```bash
uv run ruff check . --fix

uv run ruff format .

uv run pytest
```

---

# Licença

Projeto desenvolvido para fins acadêmicos (Tech Challenge).

Projeto de Recomendação da FIAP