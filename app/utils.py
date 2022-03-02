import os
from glob import glob

from huggingface_hub import list_models
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def download_model(source: str, target: str) -> None:
    model_name = f"Helsinki-NLP/opus-mt-{source}-{target}"
    try:
        print(f"check and download: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, force_download=True)

    except OSError:
        print(f'Error while fetching model "{model_name}".')
        print("Please check its availability on the Huggingface Hub.")
        return


def list_online():
    # print(list_models())
    return [m.modelId for m in list_models() if m.pipeline_tag == 'translation' and "Helsinki-NLP" in m.modelId]


def list_online_by_target(target):
    return [x for x in list_online() if f"-{target}" in x[-3:]]


def list_online_by_source(source):
    return [x for x in list_online() if f"-{source}-" in x]


def get_source_target_lang_by_model_name(model_name):
    source = model_name.split('-')[-2]
    target = model_name.split('-')[-1]

    return source, target


def get_local_models(cachedir: str = None):
    if cachedir is None:
        cachedir = os.path.expanduser('~')
        cachedir += "/.cache/huggingface/transformers/"
        print(f"\ncache_folder: {cachedir}\n")

    filenames = glob(cachedir + '*.json')
    result = []

    for filename in filenames:
        content = open(filename, 'r').read()
        current = content.split('/')[3:5]
        current = '/'.join(current)
        result = result + [current]

    for elem in set(result):
        print(elem)

    return set(result)


def get_local_models_by_target(target):
    return [x for x in get_local_models() if f"-{target}" in x[-3:]]


def get_local_models_by_source(source):
    return [x for x in get_local_models() if f"-{source}-" in x]


if __name__ == '__main__':
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
    for elem in list_online_by_target('ru'):
        source, target = get_source_target_lang_by_model_name(elem)
        download_model(source, target)

    # for elem in list_online_by_source('ru'):
    #     source, target = get_source_target_lang_by_model_name(elem)
    #     download_model(source, target)
    #
