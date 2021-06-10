import nltk
nltk.download("stopwords")  # Для dataframe

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import scipy as sp
import os  # Список файлов из директории с текстами

DB_PATH = "db"  # Путь к папке с базой статей
FILE_FORMAT = ".txt"  # Формат файлов со статьями
RES_SIZE = 3  # Количество рекомендуемых статей на выходе


class OurModel(CountVectorizer):
    def __init__(self, **kv):
        super(OurModel, self).__init__(**kv)
        # Оставляем корень слова
        self._stemmer = nltk.stem.snowball.RussianStemmer(
            'russian'
        )

    def build_analyzer(self):
        analyzer = super(OurModel, self).build_analyzer()
        return lambda doc: (self._stemmer.stem(w) for w in analyzer(doc))


# евклидова метрика расстояния между векторами
def euclid_metric(vec1, vec2):
    delta = vec1 / sp.linalg.norm(vec1.toarray()) - vec2 / sp.linalg.norm(vec2.toarray())
    return sp.linalg.norm(delta.toarray())


def main():
    # Список файлов с текстами
    file_list = []
    # Первый список - путь, второй - имя файла
    file_index = []
    for root, dirs, files in os.walk(DB_PATH):
        for file in files:
            if (file.endswith(FILE_FORMAT)):
                file_index.append(file)
                file_list.append(os.path.join(root, file))

    # Список содержимого файлов
    texts = []
    headers = []
    for file_name in file_list:
        fd = open(file_name, 'r', encoding="cp1251")
        link = fd.readline().strip()
        header = fd.readline().strip()
        headers.append((link, header))
        text = fd.read()
        texts.append(header + "\n" + text)

    # Содержимое базы данных
    print("Содержание базы данных -")
    for i in range(len(headers)):
        n = i + 1
        print(f"({n})", headers[i][1], "-", headers[i][0])

    # Класс модели задавая шаблон для выделения слов
    vectorizer = OurModel(
        min_df=1,
        # token_pattern - формат регулярного выражения, где в списке указаны все возможные символы
        token_pattern=r'[ЁАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё]{4,}'
    )

    # Тензор, где каждому слову сопоставлена его частота в конкретном тексте
    x = vectorizer.fit_transform(texts)

    texts_num, words_num = x.shape

    # Dataframe корпуса текстов
    words_index = vectorizer.get_feature_names()
    # Матрица частоты слов
    nums = x.toarray().transpose()
    df = pd.DataFrame(nums, words_index, file_index)
    # Экспорт в csv
    df.to_csv('dataframe.csv', encoding="cp1251")

    # Запрос понравившейся статьи
    target = input("\nВведите № понравившейся статьи - ")
    target = int(target) - 1

    # Проход по матрице и подсчет метрики между целевым вектором и остальными
    res = []
    for i in range(0, texts_num):
        res.append([i, euclid_metric(x[target], x[i])])
    # Сортировка
    res.sort(key=lambda x: x[1])

    # Вывод результата
    print("\nСписок рекомендуемых статей на основании понравившейся:")
    for i in range(1, min(RES_SIZE + 1, len(res))):
        n = res[i][0] + 1
        euc = res[i][1]
        print(i, "-", f"({n})", headers[res[i][0]][1], "-", headers[res[i][0]][0])


if __name__ == "__main__":
    main()