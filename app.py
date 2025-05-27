from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
from gtts import gTTS
import uuid
import torch
from TTS.api import TTS
import soundfile as sf

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'epub', 'pdf'}
MAX_TEXT_LENGTH_FOR_TTS = 5000  # Ограничение для TTS в символах

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Максимальный размер файла 16MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static', exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
coqui_tts = TTS("tts_models/de/thorsten/vits").to(device)


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


def extract_text_from_epub(filepath):
    try:
        book = epub.read_epub(filepath)
        text = ""
        for item in book.get_items():
            if item.get_type() == ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text += soup.get_text() + "\n"
        return text
    except Exception as e:
        raise Exception(f"Ошибка при чтении EPUB файла: {str(e)}")


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

        # Генерация аудио с помощью Coqui TTS (без speaker_wav и lang)
        coqui_tts.tts_to_file(
            text=text_for_tts,
            file_path=output_path
        )

        return audio_filename
    except Exception as e:
        raise Exception(f"Ошибка при синтезе речи (Coqui): {str(e)}")



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
                elif ext == 'epub':
                    book_text = extract_text_from_epub(filepath)
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
        else:
            audio_filename = synthesize_gtts(book_text)

        os.remove(temp_text_file)

        return jsonify({
            'audio_path': url_for('static', filename=audio_filename),
            'engine': tts_engine
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.secret_key = 'super_secret_key'
    app.run(debug=True)