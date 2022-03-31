"""
Модуль инициализирует модели для перевода
"""

# import huggingface_hub.constants as c
# c.HUGGINGFACE_HUB_CACHE = "cache"
import os

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from app.utils import get_pairs, get_local_models_by_target

CACHE_PATH = os.path.normpath(os.path.join(os.getcwd(), "cache/huggingface/transformers/"))
print(f'Cache directory: {CACHE_PATH}')

lang_pairs = get_pairs(get_local_models_by_target('ru', CACHE_PATH))


models_list = {}
# список языковых пар, для которых не надо загружать модели
black_models = [('ja', 'ru'), ('vi', 'ru'),
                ('ka', 'ru'), ('ko', 'ru'),
                ('ka', 'ru'), ('eo', 'ru'),
                ('ar', 'ru'), ('hy', 'ru'),
                ('rn', 'ru'), ('eu', 'ru')]

lang_pairs = sorted([x for x in lang_pairs if x not in black_models])
# lang_pairs = [("en", "ru"), ("uk", "ru")]

# загрузка моделей
print("\nLoad language models:")
for index, elem in enumerate(lang_pairs, start=1):
    if elem in black_models:
        continue
    model_name = f"Helsinki-NLP/opus-mt-{elem[0]}-{elem[1]}"

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, cache_dir=CACHE_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True, cache_dir=CACHE_PATH)
    models_list.update({elem: {"tokenizer": tokenizer, "model": model}})

    print(f'init {index} model from {len(lang_pairs)}')
