"""
Модуль содержащий вспомогательные функции, для работы с моделями для перевода
"""

import os
from glob import glob

from huggingface_hub import list_models
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def download_model(source: str, target: str) -> None:
    """
    Функция выкачивает модель по шаблону с сайта https://huggingface.co/
    По умолчанию модели хранятся в C:\\Users\\{{user_name}}\\cache\\huggingface\\transformers

    :param source: исходный язык
    :param target: язык перевода
    :return: None
    """

    model_name = f"Helsinki-NLP/opus-mt-{source}-{target}"
    try:
        print(f"check and download: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    except OSError:
        print(f'Error while fetching model "{model_name}".')
        print("Please check its availability on the Huggingface Hub.")
        return


def list_online() -> list:
    """
    Функция возвращает список всех доступных для скачивания моделей

    :return: None
    """

    return [m.modelId for m in list_models() if m.pipeline_tag == 'translation' and "Helsinki-NLP" in m.modelId]


def list_online_by_target(target: str) -> list:
    """
    Функция возвращает список всех доступных для скачивания моделей, по исходному языку

    :param target: языковой код вида ('ru', 'en', 'es' и т.д.)
    :return: список моделей
    """

    return [x for x in list_online() if f"-{target}" in x[-3:]]


def list_online_by_source(source: str) -> list:
    """
    Функция возвращает список всех доступных для скачивания моделей, по целевому языку

    :param source: языковой код вида ('ru', 'en', 'es' и т.д.)
    :return: список моделей
    """

    return [x for x in list_online() if f"-{source}-" in x]


def get_source_target_lang_by_model_name(model_name: str) -> tuple:
    """
    Функция возвращает языковую пару модели по шаблону 'Helsinki-NLP/opus-mt-{source}-{target}'

    :param model_name: название модели
    :return: кортеж вида (исходный, целевой) языки
    """

    source = model_name.split('-')[-2]
    target = model_name.split('-')[-1]

    return source, target


def get_local_models(cache_dir: str = None) -> set:
    """
    Функция возвращает названия всех оффлайн моделей

    :param cache_dir: путь к папке /huggingface/transformers/, если он изменен по умолчанию
    :return: список (множество) загруженных (локальных) моделей
    """

    if cache_dir is None:
        cache_dir = os.path.expanduser('~')
        cache_dir += "/cache/huggingface/transformers/"

    # filenames = glob(cache_dir + '\\' + '*.json')
    filenames = [os.path.join(cache_dir, x) for x in os.listdir(cache_dir) if x.endswith(".json")]
    result = []

    for filename in filenames:
        content = open(filename, 'r').read()
        current = content.split('/')[3:5]
        current = '/'.join(current)
        result = result + [current]

    return set(result)


def get_local_models_by_target(target: str, cache_dir: str = None) -> list:
    """
    Функция возвращает список моделей по языку перевода

    :param cache_dir:
    :param target: языковой код вида ('ru', 'en', 'es' и т.д.)
    :return: список моделей
    """

    return [x for x in get_local_models(cache_dir) if f"-{target}" in x[-3:]]


def get_local_models_by_source(source: str, cache_dir: str = None) -> list:
    """
    Функция возвращает список моделей по исходному языку

    :param cache_dir:
    :param source: языковой код вида ('ru', 'en', 'es' и т.д.)
    :return: список моделей
    """

    return [x for x in get_local_models(cache_dir) if f"-{source}-" in x]


def get_pairs(models_list: list) -> list:
    """
    Функция возвращает список языковых пар доступных моделей

    :param models_list: список моделей
    :return: список кортежей с языковыми парами
    """

    return [(x.split('-')[-2], x.split('-')[-1]) for x in models_list]


if __name__ == '__main__':
    pass
    # Сценарии использования:

    # GET OFFLINE MODELS
    # get_local_models()

    # GET OFFLINE MODELS BY TARGET/SOURCE
    # print(get_local_models_by_source('ru'))
    # print(get_local_models_by_target('ru'))

    # LIST ALL ONLINE MODELS"
    # print(list_online())

    # LIST ALL ONLINE MODELS BY SOURCE PATTERN "Helsinki-NLP/opus-mt-{source}-{target}"
    # print(list_online_by_source('ru'))

    # LIST ALL ONLINE MODELS BY TARGET PATTERN "Helsinki-NLP/opus-mt-{source}-{target}"
    # print(list_online_by_target('ru'))

    # DOWNLOAD ALL MODELS WITH RU
    # for elem in list_online_by_target('ru'):
    #     source, target = get_source_target_lang_by_model_name(elem)
    #     download_model(source, target)

    # for elem in list_online_by_source('ru'):
    #     source, target = get_source_target_lang_by_model_name(elem)
    #     download_model(source, target)
