# Семантический поиск по текстовым документам

> Проект по разработке и сравнению лексического и нейросетевого поиска на англоязычном научном корпусе SciFact и русскоязычном корпусе MIRACL.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow)](https://huggingface.co/docs/transformers)
[![Задача](https://img.shields.io/badge/Задача-Information%20Retrieval-0F766E)]()

## О проекте

Обычный поиск по ключевым словам не всегда находит релевантный документ: запрос и подходящий текст могут выражать одну мысль разными словами. В этом проекте реализован семантический поиск: запросы и документы переводятся в векторные представления, после чего документы ранжируются по cosine similarity.

Проект включает два независимых эксперимента.

| Эксперимент | Язык | Датасет | Модели |
|---|---|---|---|
| Поиск научных утверждений | Английский | [BEIR SciFact](https://huggingface.co/datasets/BeIR/scifact) | TF-IDF, MiniLM, дообученная MiniLM, BGE-M3 |
| Поиск документов | Русский | [MIRACL](https://huggingface.co/datasets/mteb/MIRACLRetrievalHardNegatives) | TF-IDF, многоязычная MiniLM, дообученная MiniLM, BGE-M3 |

## Что реализовано

- Лексический baseline на **TF-IDF** с униграммами и биграммами.
- Dense retrieval на Transformer-эмбеддингах.
- **Mean pooling** контекстных векторов токенов для получения одного вектора текста.
- L2-нормализация эмбеддингов: после неё скалярное произведение равно cosine similarity.
- Контрастивное дообучение MiniLM:
  - in-batch negatives и симметричная cross-entropy на SciFact;
  - `MultipleNegativesRankingLoss` в русскоязычном эксперименте.
- Собственная реализация метрик **NDCG@10**, **Recall@10**, **MRR@10** и **MAP@100** по разметке релевантности `qrels`.
- Функция поиска `top-10`: возвращает позицию, cosine similarity, идентификатор, заголовок и фрагмент документа.
- Сохранение модели, индекса эмбеддингов, метаданных и таблицы метрик для повторного инференса.

## Как работает поиск

```text
запрос / документ
        ↓
токенизация
        ↓
Transformer encoder
        ↓
векторы токенов → mean pooling → L2-нормализация
        ↓
cosine similarity с векторами документов
        ↓
ранжированный top-k документов
```

## Результаты

### SciFact: англоязычные научные утверждения

**Корпус:** 5 183 научных документа  
**Тестовая выборка:** 300 запросов, 339 релевантных пар «запрос — документ»

| Модель | NDCG@10 | Recall@10 | MRR@10 | MAP@100 |
|---|---:|---:|---:|---:|
| TF-IDF | 0.5617 | 0.7087 | 0.5205 | 0.5195 |
| MiniLM | 0.6479 | 0.7900 | 0.6055 | 0.6043 |
| **MiniLM после дообучения, 3 эпохи** | **0.6899** | **0.8428** | **0.6457** | **0.6429** |
| BGE-M3 | 0.6519 | 0.7851 | 0.6188 | 0.6090 |

Дообучение MiniLM увеличило NDCG@10 на **0.1283** относительно TF-IDF и на **0.0420** относительно исходной MiniLM. На специализированном английском наборе дообученная компактная модель превзошла более крупную BGE-M3 — это показывает ценность адаптации к домену и задаче поиска.

### MIRACL RU: русскоязычный поиск документов

**Тестовая выборка:** 150 запросов  
**Корпус для оценки:** 5 750 документов — все документы из `qrels` и 3 000 дополнительных кандидатов.

| Модель | NDCG@10 | Recall@10 | MRR@10 | MAP@100 |
|---|---:|---:|---:|---:|
| TF-IDF | 0.4957 | 0.5610 | 0.5591 | 0.4362 |
| Многоязычная MiniLM | 0.6489 | 0.7242 | 0.6914 | 0.5942 |
| Многоязычная MiniLM после дообучения, 3 эпохи | 0.7642 | 0.8380 | 0.8046 | 0.7099 |
| **BGE-M3** | **0.9275** | **0.9408** | **0.9522** | **0.9080** |

> Результаты MIRACL относятся к описанной сокращённой учебной выборке, а не к официальному рейтингу на полном корпусе MIRACL. Для всех моделей использовались одинаковые корпус, `qrels`, test split и реализация метрик.

## Выводы

1. Dense retrieval лучше TF-IDF обрабатывает случаи, когда смысл совпадает, а слова в запросе и документе различаются.
2. Дообучение MiniLM улучшило качество поиска на обоих датасетах.
3. На русском корпусе BGE-M3 показала наилучший результат благодаря многоязычному обучению и специализации на retrieval-задачах.
4. На SciFact дообученная MiniLM превзошла BGE-M3: размер модели сам по себе не гарантирует лучший результат без адаптации к домену.

## Использованные модели

| Модель | Роль | Причина выбора |
|---|---|---|
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Компактный английский encoder | Быстрый baseline и модель для дообучения |
| [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | Многоязычный encoder | Компактная модель для русскоязычного эксперимента |
| [BGE-M3](https://huggingface.co/BAAI/bge-m3) | Сильная многоязычная retrieval-модель | Поддерживает dense, sparse и multi-vector retrieval; в проекте применены dense-векторы |

## Структура репозитория

```text
semantic_search/
├── semantic_search_english.ipynb  # SciFact: TF-IDF, MiniLM, BGE-M3
└── rus_semantic_search1.ipynb     # MIRACL RU: TF-IDF, MiniLM, BGE-M3
├── requirements.txt
└── README.md
```

## Запуск

```bash
git clone https://github.com/dud0k3/semantic_search.git
cd semantic_search

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
jupyter notebook
```

При первом запуске скачаются датасеты и модели. Наличие CUDA или Apple Silicon MPS ускоряет кодирование документов, но не является обязательным.

## Технологии

`Python` · `PyTorch` · `Hugging Face Transformers` · `Sentence Transformers` · `FlagEmbedding` · `scikit-learn` · `BEIR` · `MIRACL` · `pandas` · `NumPy` · `Matplotlib`


## Автор
[GitHub](https://github.com/dud0k3)
