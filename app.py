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
import time
import librosa

load_dotenv()

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf'}
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


def get_audio_duration(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        return duration
    except Exception as e:
        return 0.0


def get_audio_size_kb(file_path):
    try:
        return round(os.path.getsize(file_path) / 1024, 2)
    except:
        return 0.0


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


def synthesize_gtts(text, lang='de'):
    try:
        import time
        text_for_tts = text[:MAX_TEXT_LENGTH_FOR_TTS]
        audio_filename = f"audio_gtts_{uuid.uuid4().hex}.mp3"
        output_path = os.path.join('static', audio_filename)

        start_time = time.time()

        tts = gTTS(text=text_for_tts, lang=lang, slow=False)
        tts.save(output_path)

        generation_time = time.time() - start_time
        duration = round(get_audio_duration(output_path), 2)
        size_kb = get_audio_size_kb(output_path)
        speed = round(duration / generation_time, 2) if generation_time > 0 else 0

        return {
            "filename": audio_filename,
            "phonemes": None,
            "duration": duration,
            "size_kb": size_kb,
            "generation_time": round(generation_time, 2),
            "speed": speed
        }
    except Exception as e:
        raise Exception(f"Error (gTTS): {str(e)}")


def synthesize_coqui(text):
    try:
        import time
        text_for_tts = text[:MAX_TEXT_LENGTH_FOR_TTS]
        audio_filename = f"audio_coqui_{uuid.uuid4().hex}.wav"
        output_path = os.path.join('static', audio_filename)

        start_time = time.time()

        # Убедись, что метод phonemize доступен, иначе убери эту строку
        phonemes = None
        if hasattr(coqui_tts, 'phonemize'):
            phonemes = coqui_tts.phonemize(text_for_tts)

        coqui_tts.tts_to_file(text=text_for_tts, file_path=output_path)

        generation_time = time.time() - start_time
        duration = round(get_audio_duration(output_path), 2)
        size_kb = get_audio_size_kb(output_path)
        speed = round(duration / generation_time, 2) if generation_time > 0 else 0

        return {
            "filename": audio_filename,
            "phonemes": phonemes,
            "duration": duration,
            "size_kb": size_kb,
            "generation_time": round(generation_time, 2),
            "speed": speed
        }
    except Exception as e:
        raise Exception(f"Ошибка при синтезе речи (Coqui): {str(e)}")


def synthesize_openai(text, voice="fable"):
    try:
        import time
        text_for_tts = text[:MAX_TEXT_LENGTH_FOR_TTS]
        audio_filename = f"audio_openai_{uuid.uuid4().hex}.mp3"
        output_path = os.path.join('static', audio_filename)

        start_time = time.time()

        response = client.audio.speech.create(
            model="tts-1-hd",
            voice=voice,
            input=text_for_tts,
            speed=1.0
        )
        response.stream_to_file(output_path)

        generation_time = time.time() - start_time
        duration = round(get_audio_duration(output_path), 2)
        size_kb = get_audio_size_kb(output_path)
        speed = round(duration / generation_time, 2) if generation_time > 0 else 0

        return {
            "filename": audio_filename,
            "phonemes": None,
            "duration": duration,
            "size_kb": size_kb,
            "generation_time": round(generation_time, 2),
            "speed": speed
        }
    except Exception as e:
        raise Exception(f"Error (OpenAI): {str(e)}")


async def synthesize_edge_async(text, voice="de-DE-KatjaNeural"):
    try:
        import time
        audio_filename = f"audio_edge_{uuid.uuid4().hex}.mp3"
        output_path = os.path.join('static', audio_filename)

        start_time = time.time()

        communicate = edge_tts.Communicate(text[:MAX_TEXT_LENGTH_FOR_TTS], voice)
        await communicate.save(output_path)

        generation_time = time.time() - start_time
        duration = round(get_audio_duration(output_path), 2)
        size_kb = get_audio_size_kb(output_path)
        speed = round(duration / generation_time, 2) if generation_time > 0 else 0

        return {
            "filename": audio_filename,
            "phonemes": None,
            "duration": duration,
            "size_kb": size_kb,
            "generation_time": round(generation_time, 2),
            "speed": speed
        }
    except Exception as e:
        raise Exception(f"Error (Edge TTS): {str(e)}")


def synthesize_edge(text, voice="de-DE-KatjaNeural"):
    return asyncio.run(synthesize_edge_async(text, voice))


def synthesize_pyttsx3(text, voice_id=None):
    try:
        import time
        audio_filename = f"audio_pyttsx3_{uuid.uuid4().hex}.wav"
        output_path = os.path.join('static', audio_filename)

        engine = pyttsx3.init()
        if voice_id:
            engine.setProperty('voice', voice_id)

        start_time = time.time()

        engine.save_to_file(text[:MAX_TEXT_LENGTH_FOR_TTS], output_path)
        engine.runAndWait()

        generation_time = time.time() - start_time
        duration = round(get_audio_duration(output_path), 2)
        size_kb = get_audio_size_kb(output_path)
        speed = round(duration / generation_time, 2) if generation_time > 0 else 0

        return {
            "filename": audio_filename,
            "phonemes": None,
            "duration": duration,
            "size_kb": size_kb,
            "generation_time": round(generation_time, 2),
            "speed": speed
        }
    except Exception as e:
        raise Exception(f"Error (pyttsx3): {str(e)}")


@app.route('/generate_audio', methods=['POST'])
def generate_audio():
    try:
        session_id = request.form.get('session_id')
        tts_engine = request.form.get('tts_engine', 'gtts')

        if not session_id:
            return jsonify({'error': 'Session ID missing'}), 400

        temp_text_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}.txt")
        if not os.path.exists(temp_text_file):
            return jsonify({'error': 'Text not found'}), 404

        with open(temp_text_file, 'r', encoding='utf-8') as f:
            book_text = f.read()

        if tts_engine == 'coqui':
            result = synthesize_coqui(book_text)
        elif tts_engine == 'openai':
            result = synthesize_openai(book_text)
        elif tts_engine == 'edge':
            result = synthesize_edge(book_text)
        elif tts_engine == 'pyttsx3':
            result = synthesize_pyttsx3(book_text)
        else:
            result = synthesize_gtts(book_text)

        return jsonify({
            'audio_path': url_for('static', filename=result['filename']),
            'engine': tts_engine,
            'phonemes': result.get('phonemes'),
            'duration': result.get('duration'),
            'size_kb': result.get('size_kb'),
            'generation_time': result.get('generation_time'),
            'speed': result.get('speed')
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 10000))
    app.secret_key = 'super_secret_key'
    app.run(host="0.0.0.0", port=port)

