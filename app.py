import os
import fitz
from flask import Flask, json, make_response, request, render_template
import pdfkit
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
import re
import requests
from io import BytesIO
import os
from supabase import create_client, Client
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Inisialisasi Supabase
url = "https://scnmodukpcrmdwlmyozl.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InNjbm1vZHVrcGNybWR3bG15b3psIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NzczMzgyMSwiZXhwIjoyMDYzMzA5ODIxfQ.iXacNmGMspo0wvQ0h_PCfQxSLuCmJFKGKX2E9pcYdgc"
supabase: Client = create_client(url, key)

# Inisialisasi Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Download resource NLTK
nltk.download('punkt')
nltk.download('stopwords')

def extract_text_from_pdf(pdf_input):
    doc = fitz.open(stream=pdf_input, filetype="pdf") if isinstance(pdf_input, BytesIO) else fitz.open(pdf_input)
    text = ""
    for page in doc:
        text += page.get_text()
    if not text.strip():
        print("PDF kosong atau tidak bisa dibaca.")
    text = hapus_daftar_pustaka(text)  # Hapus daftar pustaka
    return text

def extract_text_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        return f.read()

custom_stopwords = set([
    'tabel', 'gambar', 'hasil', 'pembahasan', 'pendahuluan', 'kesimpulan',
    'daftar', 'pustaka', 'bab', 'subbab', 'abstrak', 'judul', 'penulis', 
    'tahun', 'prosedur', 'pengujian', 'metode', 'penelitian', 'data',
    'analisis', 'bab', 'tinjauan', 'literatur', 'teori', 'kajian',
    'universitas', 'vol', 'no', 'nomor', 'doi', 'artikel', 'halaman','1',
    '2','3','4','5','6','7','8','9','0','BAB','isnn',1,2,3,4,5,6,7,8,9,
])

def hapus_daftar_pustaka(teks):
    """
    Menghapus bagian 'DAFTAR PUSTAKA' dan isinya dari teks.
    """
    pola = r'daftar pustaka[\s\S]*'  # dari "daftar pustaka" sampai akhir
    return re.sub(pola, '', teks, flags=re.IGNORECASE).strip()


def preprocess_text(text):
    text = text.lower()
    text = hapus_daftar_pustaka(text)  # pastikan daftar pustaka tidak ikut
    text = re.sub(r'\d+', ' ', text)  # hapus angka
    text = re.sub(r'[^a-z\s]', ' ', text)  # hapus simbol dan tanda baca
    text = re.sub(r'\s+', ' ', text).strip()  # hapus spasi berlebih
    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('indonesian'))
    stop_words.update(custom_stopwords)

    tokens = [word for word in tokens if word not in stop_words]
    
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(tokens)

def read_reference_docs():
    docs = []
    response = supabase.table("jurnal").select("*").execute()
    data = response.data

    for item in data:
        url = item['url']
        filename = item['nama_file']
        try:
            # Unduh file dari URL Supabase Storage
            response = requests.get(url)
            response.raise_for_status()

            # Buka PDF dari memori (tanpa disimpan ke file)
            pdf_data = BytesIO(response.content)
            text = extract_text_from_pdf(pdf_data)
            processed = preprocess_text(text)
            docs.append((filename, text, processed))
        except Exception as e:
            print(f"❌ Gagal mengunduh atau memproses file {filename}: {e}")
    return docs


def find_matching_sentences(test_sentences, reference_sentences, threshold=0.7, min_words=7):
    test_cleaned = [preprocess_text(s) for s in test_sentences]
    ref_cleaned = [preprocess_text(s) for s in reference_sentences]

    test_valid = [(s, c) for s, c in zip(test_sentences, test_cleaned) if c.strip()]
    ref_valid = [(s, c) for s, c in zip(reference_sentences, ref_cleaned) if c.strip()]

    if not test_valid or not ref_valid:
        return []

    test_originals, test_clean = zip(*test_valid)
    ref_originals, ref_clean = zip(*ref_valid)

    all_sentences = list(test_clean) + list(ref_clean)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_sentences)

    test_matrix = tfidf_matrix[:len(test_clean)]
    ref_matrix = tfidf_matrix[len(test_clean):]

    similarities = cosine_similarity(test_matrix, ref_matrix)

    matching_sentences = []
    for i, test_sentence in enumerate(test_originals):
        for j, ref_sentence in enumerate(ref_originals):
            score = similarities[i][j]
            if score > threshold:
                common_words = set(test_clean[i].split()) & set(ref_clean[j].split())
                if len(common_words) >= min_words:
                    matching_sentences.append((test_sentence, ref_sentence, score))

    return matching_sentences

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/hasil', methods=['GET', 'POST'])
def hasil():
    if request.method == 'POST':
        uploaded_file = request.files.get('testfile', None)
        text_input = request.form.get('text')  # Ambil teks dari textarea
        filename = None

        if text_input and (not uploaded_file or uploaded_file.filename == ''):
            # Jika textarea diisi dan file tidak diupload
            test_text = text_input
            filename = "input_text.txt"  # Bisa dibuat nama default
        elif uploaded_file and uploaded_file.filename != '':
            filename = secure_filename(uploaded_file.filename)
            if filename.endswith('.txt') or filename.endswith('.pdf'):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                uploaded_file.save(filepath)

                if filename.endswith('.txt'):
                    test_text = extract_text_from_txt(filepath)
                elif filename.endswith('.pdf'):
                    test_text = extract_text_from_pdf(filepath)
                else:
                    return "Format file tidak didukung!", 400
            else:
                return "File harus berupa .txt atau .pdf!", 400
        else:
            return "Harap isi teks atau unggah file!", 400

        # Tokenisasi kalimat dari teks uji
        test_sentences = sent_tokenize(test_text)
        reference_docs = read_reference_docs()

        results = []
        matched_sentences = set()  # Untuk menghindari hitung duplikat kalimat

        for ref_filename, ref_text, _ in reference_docs:
            ref_sentences = sent_tokenize(ref_text)
            matches = find_matching_sentences(test_sentences, ref_sentences)

            for test_sentence, _, _ in matches:
                matched_sentences.add(test_sentence)

            if matches:
                # Ambil nama_jurnal berdasarkan nama_file ref_filename
                nama_jurnal = None
                response = supabase.table("jurnal").select("nama_jurnal").eq("nama_file", ref_filename).execute()
                if response.data and len(response.data) > 0:
                    nama_jurnal = response.data[0].get('nama_jurnal')

                results.append({
                    'ref_file': ref_filename,
                    'nama_jurnal': nama_jurnal,
                    'matches': matches
                })

        # Hitung total kalimat uji yang panjangnya > 7 kata
        total_sentences = len([s for s in test_sentences if len(s.split()) > 7])
        total_matches = len(matched_sentences)

        plagiarism_percent = round((total_matches / total_sentences * 100), 2) if total_sentences > 0 else 0.0
        if plagiarism_percent > 100:
            plagiarism_percent = 100.0  # Maksimal 100%

        # Hapus file setelah diproses (jika ada)
        if uploaded_file and filename and os.path.exists(filepath):
            os.remove(filepath)

        return render_template(
            'results.html',
            filename=filename,
            results=results,
            plagiarism_percent=plagiarism_percent
        )

    return render_template('upload.html')


@app.route('/upload_jurnal', methods=['GET', 'POST'])
def upload_jurnal():
    message = None
    if request.method == 'POST':
        file = request.files['file']
        nama_jurnal = request.form['nama_jurnal']
        if file and nama_jurnal:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Upload ke Supabase Storage
            bucket_name = 'jurnal'  # ganti sesuai nama bucket kamu
            with open(file_path, 'rb') as f:
                # Coba upload
                try:
                    supabase.storage.from_(bucket_name).upload(filename, f, {"content-type": "application/pdf"})
                    public_url = supabase.storage.from_(bucket_name).get_public_url(filename)

                    # Simpan metadata ke tabel jurnal
                    supabase.table("jurnal").insert({
                        "nama_file": filename,
                        "url": public_url,
                        "nama_jurnal": nama_jurnal
                    }).execute()
                    message = f"Sukses mengunggah dan menyimpan data jurnal: {filename}"
                except Exception as e:
                    message = f"Gagal mengunggah file: {e}"

    return render_template('upload_jurnal.html', message=message)
@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    filename = request.form.get('filename')
    plagiarism_percent = request.form.get('plagiarism_percent')
    results_json = request.form.get('results_json')
    results = json.loads(results_json)

    rendered = render_template(
        'results.html',
        filename=filename,
        plagiarism_percent=plagiarism_percent,
        results=results
    )

    config = pdfkit.configuration(wkhtmltopdf=r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe')

    pdf = pdfkit.from_string(rendered, False, configuration=config)

    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=hasil_plagiarisme.pdf'
    return response


if __name__ == '__main__':
    app.run(debug=True)

