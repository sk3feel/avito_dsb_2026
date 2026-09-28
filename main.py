import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def clean(text):
    return " ".join(re.findall(r"\w+", str(text).lower().replace("ё", "е")))

def top(row, mask=None):
    if mask is None:
        positions = np.arange(row.nnz)
    else:
        positions = np.flatnonzero(mask[row.indices])
    if len(positions) == 0:
        return []
    count = min(300, len(positions))
    # сначала выбираем лучшие 300, потом сортируем только их
    best = np.argpartition(row.data[positions], -count)[-count:]
    positions = positions[best]
    order = np.argsort(row.data[positions])[::-1]
    return row.indices[positions[order]]

folder = Path(__file__).resolve().parent
data = folder.parent / "NLP_avito_interns"
items = pd.read_parquet(data / "benchmark_items.parquet")
queries = pd.read_parquet(data / "benchmark_queries.parquet")

# почти все выбранные объявления в train относятся к услугам с id 114
items = items[items.item_category_id == 114].reset_index(drop=True)

titles = items.item_title_raw.map(clean)

# повторяем заголовок, чтобы его слова сильнее влияли на поиск
docs = titles + " " + titles + " " + items.item_description_raw.str[:1000].map(clean)
docs = docs + " " + items.item_infm_params_text.str[:600].map(clean)
texts = queries.search_query.map(clean)
word_bm25 = CountVectorizer(ngram_range=(1, 2), min_df=2, max_features=400000, dtype=np.float32)
char_tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=300000, dtype=np.float32)

matrices = []

for model, source in zip([word_bm25, char_tfidf], [docs, titles]):
    item_matrix = model.fit_transform(source).tocsr()
    query_matrix = model.transform(texts)
    if model is word_bm25:

        # bm25 учитываетчастоту слов и длину текста, k1 = 1.2 и b = 0.75
        lengths = np.asarray(item_matrix.sum(axis=1)).ravel()
        norm = 1.2 * (0.25 + 0.75 * lengths / lengths.mean())
        frequency = np.bincount(item_matrix.indices, minlength=item_matrix.shape[1])
        idf = np.log1p((item_matrix.shape[0] - frequency + 0.5) / (frequency + 0.5))

        # Считаем веса сразу для всех непустых элементов матрицы
        tf = item_matrix.data
        denominator = tf + np.repeat(norm, np.diff(item_matrix.indptr))
        item_matrix.data[:] = idf[item_matrix.indices] * tf * 2.2 / denominator
        del tf, denominator

        # каждое слово запроса учитываем один раз
        query_matrix.data[:] = 1

    # преобразуем матрицу один раз, чтобы не делать это для каждого запроса
    item_matrix = item_matrix.T.tocsr()
    matrices.append((item_matrix, query_matrix))

ids = items.item_id.to_numpy()
locations = items.item_location_id.to_numpy()

answers = []

for n, query in queries.iterrows():
    local = locations == query.search_location_id
    scores = {}
    for item_matrix, query_matrix in matrices:
        row = query_matrix[n] @ item_matrix

        # вклад объявлений в том же месте в rrf делаем вдвое больше
        for candidates, weight in [(top(row), 1), (top(row, local), 2)]:
            for rank, idx in enumerate(candidates, 1):

                # складываем баллы за места в списках, а не сами оценки поиска
                scores[idx] = scores.get(idx, 0) + weight / (60 + rank)
    for idx in scores:
        if locations[idx] == query.search_location_id:

            # небольшой доп за ту же локу
            scores[idx] += 0.003
    best = sorted(scores, key=scores.get, reverse=True)[:50]
    answers.append(" ".join(ids[best]))

result = pd.DataFrame({"query_id": queries.query_id, "answer": answers})
result.to_csv(folder / "answer.csv", index=False)
print(folder / "answer.csv")
