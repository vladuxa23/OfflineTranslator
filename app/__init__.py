from flask import Flask
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

flask_app = Flask(__name__)

from app import views


