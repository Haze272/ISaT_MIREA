# 1. Применить для тестовых вариантов и вариантов из репозиториев различные алгоритмы поиска ассоциативных правил при одинаковых начальных
# условиях (при одинаковых пороговых значениях для поддержки и достоверности) и сравнить полученные результаты. Для тестовых вариантов выполнить
# ручные расчеты (например, с применением MS Excel) и расчеты с применением
# программных библиотек на языке Python. Для вариантов из репозиториев выполнить расчеты с применением программных библиотек на языке Python.
# В качестве алгоритмов поиска ассоциативных правил использовать алгоритмы:
#   Apriori (https://pypi.org/project/apriori-python/);
#   Efficient Apriori (https://pypi.org/project/efficient-apriori/);
#   FPGrowth (https://pypi.org/project/fpgrowth-py/).
# 2. Сформировать базы ассоциативных правил с уровнем минимальной достоверности 60% и 80%. Вычислить для ассоциативных правил поддержку, достоверность, значимость.
# 3. Оценить время формирования искомых ассоциативных правил с применением различных алгоритмов и построить диаграммы, позволяющие выполнить сравнительный анализ.
# 4. Выполнить визуализацию ассоциативных правил (https://pypi.org/project/pyarmviz/).
#
# Dataset: csv where { Date, Time, Transaction, Item }
import csv
from apriori_python import apriori
import pandas as pd
from efficient_apriori import apriori as eff_apriori
from fpgrowth_py import fpgrowth
import time


def main():
    min_sup = 0.01
    min_conf = 0.2
    # df = pd.read_csv("/home/berkunov/Documents/GitHub/ISaT_MIREA/practice1 (1-2)/online_retail.csv")
    # df = df.groupby('InvoiceNo')['Description'].apply(list).tolist()

    df = pd.read_csv("/home/berkunov/Documents/GitHub/ISaT_MIREA/practice1 (1-2)/BreadBasket_DMS.csv")
    df = df.groupby('Transaction')['Item'].apply(list).tolist()

    # apriori_python
    print("\n-----------------------apriori_python-----------------------------")

    start_time1 = time.time()
    freq_item_set1, rules1 = apriori(df, minSup=min_sup, minConf=min_conf)
    end_time1 = time.time()
    apriori_python_time = end_time1 - start_time1
    for rule in rules1:
        print(rule)

    # efficient_apriori
    print("\n-----------------------efficient_apriori-----------------------------")
    start_time2 = time.time()
    freq_item_set2, rules2 = eff_apriori(df, min_support=min_sup, min_confidence=min_conf)
    end_time2 = time.time()
    efficient_apriori_time = end_time2 - start_time2

    rules_rhs = filter(lambda rule: len(rule.lhs) == 2 and len(rule.rhs) == 1, rules2)
    for rule in sorted(rules_rhs, key=lambda rule: rule.lift):
        print(rule)

    # fpgrowth
    print("-----------------------fpgrowth-----------------------------")
    start_time3 = time.time()
    freq_item_set3, rules3 = fpgrowth(df, minSupRatio=min_sup, minConf=min_conf)
    end_time3 = time.time()
    fpgrowth_py_time = end_time3 - start_time3
    for s in rules3:
        print(*s)

    df = [
        ["Хлеб", "Горячий_шоколад", "Печенье", "Кофе"],
        ["Горячий_шоколад", "Маффин", "Вода", "Сок"],
        ["Хлеб", "Горячий_шоколад", "Печенье", "Кофе", "Маффин"],
        ["Чай", "Сок"],
        ["Хлеб", "Горячий_шоколад", "Печенье", "Кофе", "Маффин", "Вода"],
        ["Вода", "Чай"],
        ["Хлеб", "Горячий_шоколад", "Печенье", "Кофе", "Маффин", "Сок"],
        ["Вода", "Чай", "Сок"],
        ["Хлеб", "Горячий_шоколад", "Кофе"],
        ["Горячий_шоколад", "Кофе", "Маффин", "Чай", "Сок"],
        ["Печенье", "Кофе", "Маффин"],
        ["Хлеб", "Печенье", "Кофе", "Вода", "Чай"],
        ["Вода", "Чай"],
        ["Хлеб", "Горячий_шоколад", "Кофе", "Маффин", "Сок"],
        ["Горячий_шоколад", "Печенье", "Кофе", "Маффин"],
        ["Хлеб", "Вода", "Чай"],
        ["Горячий_шоколад", "Кофе", "Маффин"],
        ["Печенье", "Вода"],
        ["Хлеб", "Горячий_шоколад", "Печенье", "Кофе", "Маффин", "Сок"],
        ["Печенье", "Вода"],
        ["Хлеб", "Горячий_шоколад", "Печенье", "Кофе", "Чай"],
        ["Хлеб", "Горячий_шоколад", "Кофе", "Маффин", "Сок"],
        ["Хлеб", "Горячий_шоколад", "Печенье", "Кофе", "Маффин", "Вода"],
        ["Горячий_шоколад", "Печенье", "Кофе", "Маффин", "Чай", "Сок"],
        ["Хлеб", "Горячий_шоколад", "Печенье"],
        ["Горячий_шоколад", "Кофе", "Маффин", "Вода", "Чай"],
        ["Хлеб", "Кофе", "Маффин"],
        ["Хлеб", "Горячий_шоколад", "Сок"],
        ["Хлеб", "Горячий_шоколад", "Печенье", "Чай"],
        ["Хлеб", "Горячий_шоколад"]
    ]

    # # apriori_py
    # freq_item_set1, rules1 = apriori(df, minSup=0.4, minConf=0.6)
    # for rule in rules1:
    #     print(rule)

    # efficient
    print("\n-----------------------efficient_apriori berkunov-----------------------------")
    freq_item_set2, rules2 = eff_apriori(df, min_support=0.4, min_confidence=0.6)
    rules_rhs = filter(lambda rule: len(rule.lhs) == 1 and len(rule.rhs) == 1, rules2)
    for rule in sorted(rules_rhs, key = lambda rule: rule.confidence):
        print(rule)

    # fpgrowth
    print("\n-----------------------fpgrowth berkunov-----------------------------")
    freq_item_set, rules = fpgrowth(df, minSupRatio=0.4, minConf=0.6)
    for s in rules:
        print(*s)

    print("\n----------------------------------------------------")
    print(f"apriori_python: {apriori_python_time} seconds")
    print(f"efficient_apriori: {efficient_apriori_time} seconds")
    print(f"fpgrowth: {fpgrowth_py_time} seconds")


if __name__ == '__main__':
    main()
