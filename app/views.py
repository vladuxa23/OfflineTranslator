"""
Модуль для взаимодействия бэка и фронта
"""

import textwrap

from flask import render_template, request

from app import flask_app
from app.models import models_list

# Словарь значений для веб интерфейса
lang_dict = {'en': 'Английский',
             'af': 'Афганский',
             'bg': 'Болгарский',
             'da': 'Датский',
             'he': 'Иврит',
             'es': 'Испанский',
             'lv': 'Латышский',
             'lt': 'Литовский',
             'no': 'Норвежский',
             'sl': 'Словенский',
             'uk': 'Украинский',
             'fi': 'Финский',
             'fr': 'Французский',
             'sv': 'Шведский',
             'et': 'Эстонский'}


@flask_app.route('/', methods=['POST', 'GET'])
def index() -> str:
    """
    Метод для рендера начальной страницы

    :return: страница
    """

    # если делаем запрос со страницы
    if request.method == 'POST':
        data = request.form  # Данные с формы сайта
        target_text = translate(data["source_lang"], data["target_lang"], data["source_text"])  # переводим текст
        return render_template('index.html', title='Главная', lang_dict=lang_dict, source_text=data["source_text"],
                               target_text=target_text)

    # если просто обновляем или грузим страницу
    else:
        return render_template('index.html', title='Главная', lang_dict=lang_dict, source_text="", target_text="")


def translate(source_lang: str, target_lang: str, source_text: str):
    """
    Функция перевода текста с исходного на целевой язык

    :param source_lang: языковой код вида ('ru', 'en', 'es' и т.д.)
    :param target_lang: языковой код вида ('ru', 'en', 'es' и т.д.)
    :param source_text: исходный текст
    :return: переведённый текст
    """

    result = []
    all_paragraphs = [x for x in source_text.split("\n") if x not in ["\n", "\r", "\t"]]  # Делим текст на параграфы
    # :
    for paragraph in all_paragraphs:  # Делим текст на 512 символов
        if len(paragraph) < 512:

            tokenized_text = models_list[(source_lang, target_lang)]['tokenizer']([paragraph], return_tensors='pt',
                                                                                  padding=True)
            translation = models_list[(source_lang, target_lang)]['model'].generate(**tokenized_text)
            translated = models_list[(source_lang, target_lang)]['tokenizer'].batch_decode(translation,
                                                                                           skip_special_tokens=True)
            result.append(translated)

        else:  # Если параграф длинее 512 символов, то обрабаиываем его по частям
            temp_str = ""
            for elem_text in textwrap.wrap(paragraph, 512):
                tokenized_text = models_list[(source_lang, target_lang)]['tokenizer']([elem_text], return_tensors='pt',
                                                                                      padding=True)
                translation = models_list[(source_lang, target_lang)]['model'].generate(**tokenized_text)
                translated = models_list[(source_lang, target_lang)]['tokenizer'].batch_decode(translation,
                                                                                               skip_special_tokens=True)
                temp_str += translated[0]
            result.append([temp_str])

    return '\n\n'.join([x[0] for x in result])
