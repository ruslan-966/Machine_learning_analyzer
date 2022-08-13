from model_param import *
import numpy as np
from numpy import genfromtxt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import logging

logging.basicConfig(
    level=logging.INFO,
    filename="Rezults.log",
    format="%(asctime)s - %(module)s - %(levelname)s - %(funcName)s: %(lineno)d - %(message)s",
    datefmt='%H:%M:%S',
    encoding='utf-8',)


def analizator():
    # Загрузили преобразованную матрицу предложений:
    x = genfromtxt(FILE_COUNT_X, delimiter=',')
    logging.info('*' * 10 + 'Предварительная подготовка данных:' + '*' * 10)
    logging.debug('Считали матрицу описания телефонов:')
    logging.debug(x)
    # Загрузили преобразованный вектор ответов:
    df_y = pd.read_csv(FILE_COUNT_Y, delimiter=',')
    y = df_y.iloc[:, -1].values
    logging.debug('Считали вектор результатов:')
    logging.debug(y)
    # Оцениваем количество положительных и нулевых ответов в У:
    zero = np.sum(y == 0.)
    pls = np.sum(y == 1.)
    str_ = 'у состоит из {0} нулевых значений и {1} еденичных значений. Общее количество: {2}'.format(zero, pls,
                                                                                                      len(y))
    print(str_)
    logging.info(str_)
    str_0 = f'Количество предложений: {len(x)}.'
    print(str_0)
    logging.info(str_0)
    # Начинаем подготовку данных для модели:
    # test_size показывает, какой объем данных нужно выделить для тестового набора
    # Random_state — просто сид для случайной генерации
    # Этот параметр можно использовать для воссоздания определённого результата:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=2)
    # Балансируем данные
    os = SMOTE(random_state=0)
    os_data_x, os_data_y = os.fit_resample(x_train, y_train)
    # Проверяем рузультаты балансировки:
    logging.debug('Отбалансированные данные для обучения модели:')
    logging.debug(os_data_x)
    logging.debug(os_data_x.shape)
    logging.debug(os_data_y)
    logging.debug(os_data_y.shape)
    #  Состав данных для проверки качества модели:
    test_zro = np.sum(y_test == 0.)
    test_pls = np.sum(y_test == 1.)
    str_1 = 'Данные для тестирования результатов обучения модели содержат ' \
            'нулевых значений: {0}, положительных значений: {1}'.format(test_zro, test_pls)
    print(str_1)
    logging.info(str_1)
    os_data_y_zro = np.sum(os_data_y == 0.)
    os_data_y_pls = np.sum(os_data_y == 1.)
    str_2 = 'Отбалансированные данные для обучения модели содержат ' \
            'нулевых значений: {0}, положительных значений: {1}'.format(os_data_y_zro, os_data_y_pls)
    print(str_2)
    logging.info(str_2)
    print('*' * 30)
    # Применяем Логистическую регрессию:
    str_3 = f'Расчет по модели Логистическая регрессия:'
    print(str_3)
    logging.info(str_3)
    classifier = LogisticRegression(solver='lbfgs', random_state=0)
    clf = classifier.fit(x_train, y_train)
    str_4 = f'Коэффициенты b1, b2, b3 ...:\n {clf.coef_}\nКоэффициент b0:{clf.intercept_}'
    print(str_4)
    logging.info(str_4)
    predicted_y = classifier.predict(x_test)
    logging.debug(predicted_y)
    str_5 = 'Точность предсказания методом RL на тестовом примере{:.2f}%.'.format(classifier.score(x_test, y_test))
    print(str_5)
    logging.info(str_5)
    # Классификатор Случайный лес:
    str_6 = f'Расчет по модели Случайный лес:'
    print(str_6)
    logging.info(str_6)
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)
    rfc_predicted_y = rfc.predict(x_test)
    logging.debug(rfc_predicted_y)
    str_7 = 'Точность предсказания по модели Случайный лес на тестовом примере '
    str_7 = str_7 + 'составила: {:.2f}%.'.format(rfc.score(x_test, y_test))
    print(str_7)
    logging.info(str_7)
    str_8 = '*' * 10 + 'Расчет окончен.' + '*' * 10
    print(str_8)
    logging.info(str_8)
    return


def main():
    analizator()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
