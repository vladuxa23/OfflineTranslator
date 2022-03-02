import logging

from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def translate(text: List[str], source: str, target: str) -> List[str]:

    model_name = f"Helsinki-NLP/opus-mt-{source}-{target}"
    try:
        # Initialize the tokenizer and the model
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)

    except OSError:
        logging.disable(logging.ERROR)
        print(f'Error while loading model "{model_name}".')
        print("Has it been correctly downloaded beforehand?")
        return ['']

    # Tokenize text
    tokenized_text = tokenizer(text, return_tensors='pt', padding=True)

    # Perform translation and decode the output 
    translation = model.generate(**tokenized_text)
    translated = tokenizer.batch_decode(translation, skip_special_tokens=True)

    return translated


