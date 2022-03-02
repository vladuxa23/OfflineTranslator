from textwrap import wrap

from flask import render_template, request, jsonify, redirect, url_for

from app import flask_app
from app.models import models_list, lang_pairs


# TODO twice import and initialize models


@flask_app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        data = request.form
        target_text = translate(data["source_lang"], data["target_lang"], data["source_text"])
        return render_template('index.html', title='Главная', lang_pairs=lang_pairs, source_text=data["source_text"],
                               target_text=target_text)
    else:
        return render_template('index.html', title='Главная', lang_pairs=lang_pairs, source_text="", target_text="")


def translate(source_lang, target_lang, source_text):
    result = []
    for elem_text in wrap(source_text, 512):
        tokenized_text = models_list[(source_lang, target_lang)]['tokenizer']([elem_text], return_tensors='pt',
                                                                              padding=True)

        translation = models_list[(source_lang, target_lang)]['model'].generate(**tokenized_text)
        translated = models_list[(source_lang, target_lang)]['tokenizer'].batch_decode(translation,
                                                                                       skip_special_tokens=True)
        result.append(translated)

    return ' '.join([x[0] for x in result])
