import numpy as np
import requests
import re

# Implementasi SkipGram model dasar
class SkipGramModel:
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # Bobot input-ke-hidden (embeddings)
        self.W1 = np.random.randn(vocab_size, embedding_dim)
        # Bobot hidden-ke-output
        self.W2 = np.random.randn(embedding_dim, vocab_size)

    def forward(self, one_hot_vector):
        hidden_layer = np.dot(one_hot_vector, self.W1)
        output_layer = np.dot(hidden_layer, self.W2)
        output_layer = self._softmax(output_layer)
        return hidden_layer, output_layer

    def backward(self, one_hot_vector, target_vector, learning_rate=0.01):
        hidden_layer, output_layer = self.forward(one_hot_vector)
        error = target_vector - output_layer

        # Hitung gradien
        output_layer_gradient = np.outer(hidden_layer, error)
        hidden_layer_gradient = np.outer(one_hot_vector, np.dot(self.W2, error))

        # Perbarui bobot
        self.W1 -= learning_rate * hidden_layer_gradient
        self.W2 -= learning_rate * output_layer_gradient

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()


# Fungsi untuk mengambil data dari NewsAPI
def fetch_news_data(api_key, query="latest", page_size=50):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "pageSize": page_size,
        "apiKey": api_key,
        "language": "en"
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print("Error fetching news data:", response.text)
        return []
    data = response.json()
    articles = data.get("articles", [])
    texts = []
    for article in articles:
        # Menggabungkan judul, deskripsi, dan konten (jika ada)
        title = article.get("title", "")
        description = article.get("description", "")
        content = article.get("content", "")
        text = f"{title} {description} {content}"
        texts.append(text)
    return texts


# Fungsi untuk melakukan preprocessing terhadap teks (lowercase, hapus tanda baca, tokenisasi)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = text.split()
    return tokens


# Membuat data training (training pairs) berdasarkan window size yang diberikan
def build_training_data(texts, window_size=1):
    corpus = []
    for text in texts:
        tokens = preprocess_text(text)
        corpus.extend(tokens)
    # Buat vocabulary unik
    vocab = list(set(corpus))
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    training_pairs = []
    for i, word in enumerate(corpus):
        target_idx = word2idx[word]
        # Tentukan rentang konteks dengan window size
        start = max(0, i - window_size)
        end = min(len(corpus), i + window_size + 1)
        for j in range(start, end):
            if j != i:
                context_idx = word2idx[corpus[j]]
                training_pairs.append((target_idx, context_idx))
    return training_pairs, word2idx


# Fungsi untuk melatih model SkipGram
def train_skipgram(training_pairs, vocab_size, embedding_dim, epochs=5, learning_rate=0.01):
    model = SkipGramModel(vocab_size, embedding_dim)
    total_pairs = len(training_pairs)
    for epoch in range(epochs):
        total_loss = 0
        for target_idx, context_idx in training_pairs:
            # Representasi one-hot untuk target dan konteks
            target_vector = np.zeros(vocab_size)
            target_vector[target_idx] = 1
            context_vector = np.zeros(vocab_size)
            context_vector[context_idx] = 1
            # Forward pass dan hitung loss
            _, output = model.forward(target_vector)
            loss = -np.log(output[context_idx] + 1e-9)  # menambahkan epsilon untuk menghindari log(0)
            total_loss += loss
            # Backward pass untuk update bobot
            model.backward(target_vector, context_vector, learning_rate)
        avg_loss = total_loss / total_pairs
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")
    return model


# Fungsi untuk menghitung cosine similarity antar vektor
def cosine_similarity(vec1, vec2):
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# Fungsi untuk mengevaluasi model: menampilkan kata-kata dengan embedding yang paling mirip
def evaluate_model(model, word2idx, target_word, top_n=5):
    if target_word not in word2idx:
        print(f"Kata '{target_word}' tidak ditemukan dalam vocabulary.")
        return
    target_idx = word2idx[target_word]
    target_embedding = model.W1[target_idx]
    similarities = {}
    for word, idx in word2idx.items():
        if word == target_word:
            continue
        embedding = model.W1[idx]
        sim = cosine_similarity(target_embedding, embedding)
        similarities[word] = sim
    # Urutkan berdasarkan similarity secara menurun
    sorted_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    print(f"\nKata-kata yang paling mirip dengan '{target_word}':")
    for word, sim in sorted_words[:top_n]:
        print(f"{word}: {sim:.4f}")


# Main function: Melakukan training untuk setiap kombinasi parameter dan evaluasi model
def main():
    api_key = "a707aa9c8c0d40738c4b1261380aca86"  
    query = "technology"  # Topik pencarian berita
    texts = fetch_news_data(api_key, query=query, page_size=50)
    if not texts:
        print("Tidak ada data berita yang berhasil diambil.")
        return

    # Variasi parameter
    window_sizes = [1, 2, 3]
    embedding_dims = [10, 20, 50]
    results = {}  # Menyimpan model dan vocabulary untuk setiap kombinasi

    for window_size in window_sizes:
        print(f"\n=== Proses dengan window size = {window_size} ===")
        training_pairs, word2idx = build_training_data(texts, window_size)
        vocab_size = len(word2idx)
        for embedding_dim in embedding_dims:
            print(f"\nMelatih model dengan embedding dim = {embedding_dim}")
            # Melatih model (jumlah epoch dapat disesuaikan)
            model = train_skipgram(training_pairs, vocab_size, embedding_dim, epochs=5, learning_rate=0.01)
            results[(window_size, embedding_dim)] = (model, word2idx)
            
            # Evaluasi
            for word in ["technology", "innovation", "science", "data"]:
                evaluate_model(model, word2idx, word, top_n=3)
    
    print("\nTugas selesai.")

if __name__ == "__main__":
    main()
