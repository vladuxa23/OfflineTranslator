from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from app.utils import get_pairs, get_local_models_by_target

lang_pairs = get_pairs(get_local_models_by_target('ru'))


models_list = {}
black_models = [('ja', 'ru'), ('vi', 'ru'),
                ('ka', 'ru'), ('ko', 'ru'),
                ('ka', 'ru'), ('eo', 'ru'),
                ('ar', 'ru'), ('hy', 'ru'),
                ('rn', 'ru')]

lang_pairs = [x for x in lang_pairs if x not in black_models]
print(lang_pairs)

for index, elem in enumerate(lang_pairs, start=1):
    if elem in black_models:
        continue
    model_name = f"Helsinki-NLP/opus-mt-{elem[0]}-{elem[1]}"

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
    models_list.update({elem: {"tokenizer": tokenizer, "model": model}})

    print(f'init {index} model from {len(lang_pairs)}')