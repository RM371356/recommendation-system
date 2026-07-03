# Roteiro de Vídeo — Tech Challenge FIAP

## Sistema de Recomendação de Filmes (MovieLens)

**Formato:** Vídeo de apresentação • **Modelo:** STAR (Situação, Tarefa, Ação, Resultado) • **Duração-alvo:** 5 minutos

---

### Como usar este roteiro

- A coluna **[tempo]** indica o momento acumulado do vídeo.
- O texto em *itálico* é a **fala** (o que você diz).
- As linhas com **[TELA]** indicam o que mostrar (slide ou demonstração ao vivo).
- Ritmo alvo: ~130–140 palavras por minuto. Ensaide 1–2 vezes cronometrando.

---

## Abertura — [0:00 – 0:20] (20s) · Slide 1 (Capa)

**[TELA]** Slide de capa com título, dataset e stack.

> *"Olá! Neste vídeo eu vou apresentar o nosso Tech Challenge da FIAP: um sistema de recomendação de filmes de ponta a ponta, que vai do dado bruto do MovieLens até uma API REST pronta para produção. Vou estruturar a explicação no modelo STAR: Situação, Tarefa, Ação e Resultado."*

---

## S — Situação — [0:20 – 1:00] (40s) · Slide 2

**[TELA]** Slide "Situação".

> *"A situação é a seguinte: plataformas de streaming lidam com catálogos enormes, e o usuário se perde no volume de opções. Sem uma boa recomendação, o engajamento cai e o conteúdo relevante fica invisível."*
>
> *"Para o desafio, usamos o dataset MovieLens, com mais de 100 mil avaliações de filmes feitas por usuários reais. O problema era claro: como prever, para cada usuário, quais filmes ele realmente vai gostar — e entregar isso de forma rápida e confiável?"*

---

## T — Tarefa — [1:00 – 1:40] (40s) · Slide 3

**[TELA]** Slide "Tarefa".

> *"A nossa tarefa foi construir um sistema completo de recomendação baseado em Machine Learning. Isso significava três compromissos: primeiro, modelar o problema; segundo, ter um pipeline reprodutível de dados e treino; e terceiro, servir as recomendações por uma API."*
>
> *"Modelamos o problema como uma classificação binária: em vez de tentar prever a nota exata, prevemos se o usuário daria nota 4 ou mais — ou seja, se ele 'gostaria' do filme. E a entrega principal é simples: dado um user_id, retornar o top-K de filmes recomendados, com título, gênero e score."*

---

## A — Ação — [1:40 – 4:00] (140s) · Slides 4 a 8

### A.1 — Pipeline e Arquitetura — [1:40 – 2:15] · Slide 4

**[TELA]** Slide do pipeline (Preprocess → Features → Split → Train → Evaluate).

> *"Na ação, começo pela arquitetura. Todo o fluxo de dados é orquestrado com DVC, garantindo reprodutibilidade. São cinco estágios: pré-processamento, onde binarizamos as notas; engenharia de atributos, com LabelEncoders para usuários e filmes; a divisão estratificada em treino, validação e teste, sem vazamento de dados; o treinamento; e a avaliação."*

### A.2 — O Modelo — [2:15 – 2:55] · Slide 5

**[TELA]** Slide do modelo MLP com embeddings.

> *"O coração é um modelo em PyTorch: uma rede neural MLP com embeddings. Cada usuário e cada filme viram um vetor latente aprendido durante o treino. Concatenamos esses vetores e passamos por camadas densas com ReLU e dropout, até uma saída única que estima a probabilidade de o usuário gostar do filme."*
>
> *"Treinamos com a função de perda BCEWithLogitsLoss, otimizador Adam e early stopping baseado na perda de validação, salvando sempre o melhor modelo. Usamos padrões de projeto como Factory para criar modelos e Strategy no pré-processamento, o que deixa o sistema extensível."*

### A.3 — MLOps — [2:55 – 3:20] · Slide 6

**[TELA]** Slide MLflow + DVC.

> *"Do lado de MLOps, o MLflow registra cada experimento: parâmetros, hiperparâmetros e as perdas de treino e validação por época. Junto com o DVC e a fixação de seeds, conseguimos reproduzir qualquer resultado — algo essencial para confiança e evolução do modelo."*

### A.4 — API e Inferência — [3:20 – 3:45] · Slide 7

**[TELA]** Slide da API (ou demonstração ao vivo do Swagger em `/docs`).

> *"Para servir, usamos FastAPI. A camada de inferência é modular: um cache singleton mantém o modelo e os encoders em memória, um Predictor pontua todos os filmes candidatos em um único batch, e um serviço de recomendação orquestra tudo. Há dois endpoints: health e recommend. E um detalhe importante: se o modelo não estiver disponível ou o usuário for novo, o sistema faz fallback para recomendações por popularidade — ele nunca quebra."*

### A.5 — Qualidade e Testes — [3:45 – 4:00] · Slide 8

**[TELA]** Slide de testes (143 testes, 8 categorias).

> *"E não paramos no 'funciona na minha máquina': construímos uma suíte de 143 testes automatizados em 8 categorias — incluindo segurança, resiliência e performance — que roda em cerca de 6 segundos, sem precisar de um modelo treinado."*

---

## R — Resultado — [4:00 – 4:40] (40s) · Slide 9

**[TELA]** Slide "Resultado" (métricas + demonstração da resposta da API).

> *"O resultado é um sistema completo e funcional. Temos uma API REST que devolve recomendações personalizadas, avaliada com métricas de classificação no conjunto de teste: acurácia de cerca de 70%, com precisão, recall e F1 todos na casa dos 0,70 — um resultado equilibrado. Ganhamos reprodutibilidade total com DVC e MLflow, robustez com o fallback de popularidade, e uma base de qualidade forte graças aos 143 testes."*
>
> *"Na prática: uma requisição POST no endpoint recommend, com um user_id, retorna em menos de dois segundos a lista dos melhores filmes para aquele usuário."*

---

## Encerramento — [4:40 – 5:00] (20s) · Slide 10

**[TELA]** Slide de próximos passos / encerramento.

> *"Como próximos passos, temos a integração do negative sampling ao pipeline, o registro do modelo no MLflow Model Registry e a esteira de CI/CD. Obrigado por assistir — esse foi o nosso Sistema de Recomendação de Filmes do Tech Challenge FIAP!"*

---

## Checklist de gravação

- [ ] Testar áudio e iluminação.
- [ ] Deixar o Swagger (`http://localhost:8000/docs`) aberto caso queira demonstração ao vivo.
- [ ] Ter uma resposta de exemplo do endpoint `/api/v1/recommend` pronta para mostrar.
- [ ] Cronometrar cada bloco no ensaio (margem de ±10s por seção).
- [ ] Falar em ritmo calmo; a soma dos blocos já está calibrada para ~5 minutos.

## Resumo de tempo

| Seção | Início | Fim | Duração |
| --- | --- | --- | --- |
| Abertura | 0:00 | 0:20 | 20s |
| S — Situação | 0:20 | 1:00 | 40s |
| T — Tarefa | 1:00 | 1:40 | 40s |
| A — Ação | 1:40 | 4:00 | 140s |
| R — Resultado | 4:00 | 4:40 | 40s |
| Encerramento | 4:40 | 5:00 | 20s |
| **Total** | | | **5:00** |
