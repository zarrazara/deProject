from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from werkzeug.utils import secure_filename
import fitz
from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
from gtts import gTTS
import uuid
import torch
from TTS.api import TTS
from openai import OpenAI
import soundfile as sf
from dotenv import load_dotenv
import edge_tts
import asyncio
import pyttsx3

load_dotenv()

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'epub', 'pdf'}
MAX_TEXT_LENGTH_FOR_TTS = 5000

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static', exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
coqui_tts = TTS("tts_models/de/thorsten/vits").to(device)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_txt(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        raise Exception(f"Ошибка при чтении TXT файла: {str(e)}")


def extract_text_from_pdf(filepath):
    try:
        doc = fitz.open(filepath)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        raise Exception(f"Ошибка при чтении PDF файла: {str(e)}")


def synthesize_gtts(text, lang='de'):
    """Синтез речи с помощью gTTS"""
    try:
        text_for_tts = text[:MAX_TEXT_LENGTH_FOR_TTS]
        audio_filename = f"audio_gtts_{uuid.uuid4().hex}.mp3"
        output_path = os.path.join('static', audio_filename)

        tts = gTTS(text=text_for_tts, lang=lang, slow=False)
        tts.save(output_path)
        return audio_filename
    except Exception as e:
        raise Exception(f"Ошибка при синтезе речи (gTTS): {str(e)}")


def synthesize_coqui(text):
    """Синтез речи с помощью Coqui TTS (VITS)"""
    try:
        text_for_tts = text[:MAX_TEXT_LENGTH_FOR_TTS]
        audio_filename = f"audio_coqui_{uuid.uuid4().hex}.wav"
        output_path = os.path.join('static', audio_filename)

        coqui_tts.tts_to_file(
            text=text_for_tts,
            file_path=output_path
        )

        return audio_filename
    except Exception as e:
        raise Exception(f"Ошибка при синтезе речи (Coqui): {str(e)}")


def synthesize_openai(text, voice="fable"):
    """Синтез речи с помощью OpenAI TTS"""
    try:
        text_for_tts = text[:MAX_TEXT_LENGTH_FOR_TTS]
        audio_filename = f"audio_openai_{uuid.uuid4().hex}.mp3"
        output_path = os.path.join('static', audio_filename)

        response = client.audio.speech.create(
            model="tts-1-hd",
            voice=voice,
            input=text_for_tts,
            speed=1.0
        )

        response.stream_to_file(output_path)
        return audio_filename
    except Exception as e:
        raise Exception(f"Ошибка при синтезе речи (OpenAI): {str(e)}")


async def synthesize_edge_async(text, voice="de-DE-KatjaNeural"):
    """Синтез речи с помощью Edge TTS"""
    try:
        audio_filename = f"audio_edge_{uuid.uuid4().hex}.mp3"
        output_path = os.path.join('static', audio_filename)

        communicate = edge_tts.Communicate(text[:MAX_TEXT_LENGTH_FOR_TTS], voice)
        await communicate.save(output_path)
        return audio_filename
    except Exception as e:
        raise Exception(f"Ошибка при синтезе речи (Edge TTS): {str(e)}")


def synthesize_edge(text, voice="de-DE-KatjaNeural"):
    return asyncio.run(synthesize_edge_async(text, voice))


def synthesize_pyttsx3(text, voice_id=None):
    """Синтез речи с помощью pyttsx3 (оффлайн)"""
    try:
        audio_filename = f"audio_pyttsx3_{uuid.uuid4().hex}.wav"
        output_path = os.path.join('static', audio_filename)

        engine = pyttsx3.init()

        if voice_id:
            engine.setProperty('voice', voice_id)

        engine.save_to_file(text[:MAX_TEXT_LENGTH_FOR_TTS], output_path)
        engine.runAndWait()

        return audio_filename
    except Exception as e:
        raise Exception(f"Ошибка при синтезе речи (pyttsx3): {str(e)}")


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'book' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['book']

        if file.filename == '':
            return render_template('index.html', error='No selected file')

        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                ext = filename.rsplit('.', 1)[1].lower()
                if ext == 'txt':
                    book_text = extract_text_from_txt(filepath)
                elif ext == 'pdf':
                    book_text = extract_text_from_pdf(filepath)
                else:
                    return render_template('index.html', error=f"Формат {ext} не поддерживается.")

                session_id = uuid.uuid4().hex
                temp_text_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}.txt")
                with open(temp_text_file, 'w', encoding='utf-8') as f:
                    f.write(book_text)

                return render_template('index.html',
                                       book_text=book_text,
                                       session_id=session_id,
                                       show_text=True)

            except Exception as e:
                error = str(e)
                if os.path.exists(filepath):
                    os.remove(filepath)
                return render_template('index.html', error=error)
        else:
            return render_template('index.html', error='Недопустимый тип файла. Разрешены только txt, pdf, epub.')

    return render_template('index.html')


@app.route('/generate_audio', methods=['POST'])
def generate_audio():
    try:
        session_id = request.form.get('session_id')
        tts_engine = request.form.get('tts_engine', 'gtts')  # По умолчанию gTTS

        if not session_id:
            return jsonify({'error': 'Session ID missing'}), 400

        temp_text_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}.txt")

        if not os.path.exists(temp_text_file):
            return jsonify({'error': 'Text not found'}), 404

        with open(temp_text_file, 'r', encoding='utf-8') as f:
            book_text = f.read()

        if tts_engine == 'coqui':
            audio_filename = synthesize_coqui(book_text)
        elif tts_engine == 'openai':
            audio_filename = synthesize_openai(book_text)
        elif tts_engine == 'edge':
            audio_filename = synthesize_edge(book_text)
        elif tts_engine == 'pyttsx3':
            audio_filename = synthesize_pyttsx3(book_text)
        else:
            audio_filename = synthesize_gtts(book_text)


        return jsonify({
            'audio_path': url_for('static', filename=audio_filename),
            'engine': tts_engine
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.secret_key = 'super_secret_key'
    app.run(debug=True)
