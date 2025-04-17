import os
import fitz  # PyMuPDF
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
import re

# Inisialisasi Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Download resource NLTK
nltk.download('punkt')
nltk.download('stopwords')

# --- Fungsi ---
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    if not text.strip():
        print(f"⚠️ PDF kosong atau tidak bisa dibaca: {pdf_path}")
    return ''.join(page.get_text() for page in doc)

def extract_text_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        return f.read()

custom_stopwords = set([
    'tabel', 'gambar', 'hasil', 'pembahasan', 'pendahuluan', 'kesimpulan',
    'daftar', 'pustaka', 'bab', 'subbab', 'abstrak', 'judul', 'penulis', 
    'tahun', 'prosedur', 'pengujian', 'metode', 'penelitian', 'data',
    'analisis', 'bab', 'tinjauan', 'literatur', 'teori', 'kajian',
    'universitas', 'vol', 'no', 'nomor', 'doi', 'artikel', 'halaman','1',
    '2','3','4','5','6','7','8','9','0','BAB','isnn'
])

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('indonesian')) 
    stop_words.update(custom_stopwords)
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

def read_reference_docs(directory='dataset/reference'):
    docs = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            path = os.path.join(directory, filename)
            text = extract_text_from_pdf(path)
            processed = preprocess_text(text)
            docs.append((filename, text, processed))
    return docs

def find_matching_sentences(test_sentences, reference_sentences, threshold=0.7):
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
            if score > threshold and len(test_sentence.split()) > 7:
                matching_sentences.append((test_sentence, ref_sentence))

    return matching_sentences

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['testfile']
        filename = secure_filename(uploaded_file.filename)

        if uploaded_file and (filename.endswith('.txt') or filename.endswith('.pdf')):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(filepath)

            if filename.endswith('.txt'):
                test_text = extract_text_from_txt(filepath)
            elif filename.endswith('.pdf'):
                test_text = extract_text_from_pdf(filepath)
            else:
                return "Format file tidak didukung!", 400

            test_sentences = sent_tokenize(test_text)
            reference_docs = read_reference_docs()
            results = []
            total_matches = 0

            for ref_filename, ref_text, _ in reference_docs:
                ref_sentences = sent_tokenize(ref_text)
                matches = find_matching_sentences(test_sentences, ref_sentences)

                if matches:
                    total_matches += len(matches)
                    results.append({
                        'ref_file': ref_filename,
                        'matches': matches
                    })

            total_sentences = len([s for s in test_sentences if len(s.split()) > 7])
            plagiarism_percent = round((total_matches / total_sentences * 100), 2) if total_sentences > 0 else 0.0

            if os.path.exists(filepath):
                os.remove(filepath)

            return render_template('results.html', filename=filename, results=results, plagiarism_percent=plagiarism_percent)

        return "File harus berupa .txt atau .pdf!", 400

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
