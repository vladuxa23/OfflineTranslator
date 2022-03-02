from flask import render_template, request, jsonify


from app import flask_app

@flask_app.route('/')
def index():
    return render_template('index.html', title='Главная')

