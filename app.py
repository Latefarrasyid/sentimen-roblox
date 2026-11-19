import re
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

st.set_page_config(page_title="Sentimen Ulasan Roblox", page_icon="🎮", layout="wide")

# ----------------------------------------------------------------------
# CSS: rapikan sidebar (naikkan ke atas) dan kotak keterangan
# ----------------------------------------------------------------------
st.markdown("""
<style>
/* Sidebar: hilangkan ruang kosong di atas */
[data-testid="stSidebarHeader"] {
    padding: 0.4rem 1rem 0 1rem !important;
    height: auto !important;
    min-height: 0 !important;
    margin-bottom: 0 !important;
}
[data-testid="stSidebarUserContent"] {
    padding-top: 0.5rem !important;
}
section[data-testid="stSidebar"] .block-container,
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}
[data-testid="stSidebar"] h3 {
    margin: 0 0 0.2rem 0 !important;
    padding: 0 !important;
}
[data-testid="stSidebar"] hr {
    margin: 0.8rem 0 !important;
}
/* Konten utama sedikit lebih naik */
.block-container { padding-top: 2.5rem !important; }
</style>
""", unsafe_allow_html=True)

RANDOM_STATE = 42
LABELS = ["negatif", "netral", "positif"]
PALET = {"negatif": "#d9534f", "netral": "#f0ad4e", "positif": "#5cb85c"}
DEFAULT_CSV = "roblox_ulasan_dominan.csv"

# ======================================================================
# PREPROCESSING (sama dengan notebook)
# ======================================================================
STOPWORDS = set("""
aku saya gue gua gw kamu kalian kita kami dia mereka anda ku mu
ini itu yang di ke dari dan atau dengan untuk pada dalam oleh karena
sih deh dong kok ya yaa lah kah nih tuh aja saja juga jadi ada adalah akan
sudah udah lagi pun si nya kan ni lho loh nah kek lu yg sama kalau
sangat banget sekali bgt amat the is it to a an and of i my you
roblox roblok mau buat bikin kasih sedang masih
""".split())
NEGASI = {"tidak", "bukan", "belum", "jangan", "tanpa", "kurang", "tak"}
STOPWORDS -= NEGASI

PROTECT_STEM = {"perbaiki", "kembalikan", "turunkan", "hapus", "bagus", "game", "robux",
                "verifikasi", "tolong", "mahal", "murah", "seru", "mantap", "keren", "server",
                "update", "akun", "marketplace", "chat", "lag", "bug", "error", "login",
                "download", "aturan", "peraturan"}


@st.cache_resource
def get_stemmer():
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        return StemmerFactory().create_stemmer()
    except ImportError:
        return None


_stem_cache = {}


def stem_word(w):
    if w in PROTECT_STEM:
        return w
    if w not in _stem_cache:
        stemmer = get_stemmer()
        if stemmer is not None:
            _stem_cache[w] = stemmer.stem(w)
        else:
            _stem_cache[w] = re.sub(r"(lah|kah|pun)$", "", w) if len(w) > 5 else w
    return _stem_cache[w]


def preprocess(text, return_stages=False):
    s0 = str(text)
    s1 = s0.lower()
    s2 = re.sub(r"https?://\S+|www\.\S+|@\w+|#\w+", " ", s1)
    s3 = re.sub(r"\s+", " ", re.sub(r"[^a-z\s]", " ", s2)).strip()
    s4 = re.sub(r"(.)\1{2,}", r"\1", s3)
    s5 = s4.split()
    s6 = [w for w in s5 if w not in STOPWORDS]
    s7 = [stem_word(w) for w in s6]
    s8 = [w for w in s7 if len(w) > 1]
    if return_stages:
        return [s0, s1, s2, s3, s4, " | ".join(s5), " | ".join(s6), " | ".join(s7), " | ".join(s8)]
    return " ".join(s8)


STAGE_NAMES = ["Asli", "1. Case folding", "2. Hapus URL/mention", "3. Hapus simbol",
               "4. Huruf berulang", "5. Tokenisasi", "6. Stopword", "7. Stemming",
               "8. Filter pendek"]

# Penjelasan awam untuk tiap tahap preprocessing
PENJELASAN_TAHAP = pd.DataFrame({
    "Tahap": ["1. Case folding", "2. Hapus URL/mention", "3. Hapus simbol",
              "4. Huruf berulang", "5. Tokenisasi", "6. Stopword",
              "7. Stemming", "8. Filter pendek"],
    "Apa yang dilakukan (bahasa sehari-hari)": [
        "Semua huruf diseragamkan menjadi huruf kecil agar 'Bagus' dan 'bagus' dianggap kata yang sama.",
        "Alamat web (link), @nama pengguna, dan #hashtag dibuang karena tidak menunjukkan perasaan penulis.",
        "Angka, tanda baca, dan emoji dibuang sehingga yang tersisa hanya huruf.",
        "Huruf yang diulang-ulang untuk penekanan dirapikan, misalnya 'baguuus' menjadi 'bagus'.",
        "Kalimat dipecah menjadi potongan kata satu per satu agar bisa dihitung dan dianalisis.",
        "Kata penghubung yang terlalu umum (seperti 'yang', 'di', 'dan') dibuang karena tidak membawa makna sentimen. "
        "Kata penyangkal seperti 'tidak' sengaja dipertahankan karena bisa membalik makna.",
        "Kata diubah ke bentuk dasarnya, misalnya 'memperbaiki' menjadi 'baik', agar kata sekeluarga dihitung sebagai satu.",
        "Kata yang hanya satu huruf dibuang karena biasanya hanya sisa pengetikan."],
    "Contoh": ["Bagus → bagus", "cek https://x.com @budi → cek", "seru!!! 100% → seru",
               "baguuus → bagus", "game seru → game | seru", "saya dan kamu → (dibuang)",
               "memperbaiki → baik", "x → (dibuang)"],
})


# ======================================================================
# DATA & MODEL
# ======================================================================
@st.cache_data(show_spinner="Preprocessing teks...")
def load_and_clean(raw: pd.DataFrame):
    df = raw.copy()
    df["text_clean"] = df["content"].apply(preprocess)
    df = df[df["text_clean"].str.strip() != ""].reset_index(drop=True)
    return df


@st.cache_resource(show_spinner="Melatih model...")
def train_model(df, test_size, alpha, min_df, ngram_max, balance):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    idx_train, idx_test = next(sss.split(df, df["sentimen"]))
    train_df, test_df = df.iloc[idx_train].copy(), df.iloc[idx_test].copy()

    if balance:
        n_max = train_df["sentimen"].value_counts().max()
        train_fit = pd.concat([
            g.sample(n=n_max, replace=len(g) < n_max, random_state=RANDOM_STATE)
            for _, g in train_df.groupby("sentimen")
        ]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    else:
        train_fit = train_df

    tfidf = TfidfVectorizer(ngram_range=(1, ngram_max), min_df=min_df, sublinear_tf=True)
    X_fit = tfidf.fit_transform(train_fit["text_clean"])
    X_train = tfidf.transform(train_df["text_clean"])
    X_test = tfidf.transform(test_df["text_clean"])

    nb = MultinomialNB(alpha=alpha).fit(X_fit, train_fit["sentimen"])
    return dict(train_df=train_df, test_df=test_df, train_fit=train_fit,
                tfidf=tfidf, nb=nb,
                pred_train=nb.predict(X_train), pred_test=nb.predict(X_test))


def hitung(y, p):
    return {"Akurasi": accuracy_score(y, p),
            "Precision": precision_score(y, p, average="macro", zero_division=0),
            "Recall": recall_score(y, p, average="macro", zero_division=0),
            "F1-score": f1_score(y, p, average="macro", zero_division=0)}


def wordcloud_fig(text, cmap):
    wc = WordCloud(width=1000, height=600, background_color="white", colormap=cmap,
                   max_words=100, random_state=RANDOM_STATE).generate(text)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout()
    return fig


def tabel_kalimat_sering(df, sentimen, top=10):
    d = df[df["sentimen"] == sentimen][["content", "kategori"]].copy()
    d["kalimat"] = d["content"].str.split(r"\s*\|\s*")
    d = d.explode("kalimat")
    d["kalimat"] = d["kalimat"].str.strip().str.lower()
    d = d[(d["kalimat"].str.len() > 0) & (d["kalimat"].str.len() <= 100)]
    out = (d.groupby("kalimat")
             .agg(Frekuensi=("kalimat", "size"),
                  Kategori_dominan=("kategori", lambda x: x.mode().iat[0]))
             .sort_values("Frekuensi", ascending=False).head(top).reset_index()
             .rename(columns={"kalimat": "Kalimat", "Kategori_dominan": "Kategori dominan"}))
    out.index += 1
    return out


def top_bigram(df, sentimen, top=10):
    try:
        cv = CountVectorizer(ngram_range=(2, 2), min_df=2)
        X = cv.fit_transform(df.loc[df["sentimen"] == sentimen, "text_clean"])
    except ValueError:
        return pd.DataFrame(columns=["Frasa", "Frekuensi"])
    frek = np.asarray(X.sum(axis=0)).ravel()
    t = pd.DataFrame({"Frasa": cv.get_feature_names_out(), "Frekuensi": frek})
    t = t.sort_values("Frekuensi", ascending=False).head(top).reset_index(drop=True)
    t.index += 1
    return t


def kata_dominan(df, sentimen, top=7):
    baris = {}
    for kat, g in df[df["sentimen"] == sentimen].groupby("kategori"):
        c = Counter(w for t in g["text_clean"] for w in t.split() if w not in NEGASI)
        baris[kat] = ", ".join(f"{w} ({n})" for w, n in c.most_common(top))
    return pd.DataFrame({"Kata dominan (jumlah)": baris}).rename_axis("Kategori")


# ======================================================================
# FUNGSI PENJELASAN OTOMATIS (mengikuti data yang tampil)
# ======================================================================
def kata_teratas(df, sentimen, top=8):
    c = Counter(w for t in df.loc[df["sentimen"] == sentimen, "text_clean"]
                for w in t.split() if w not in NEGASI)
    return c.most_common(top)


def penjelasan_wordcloud(df, sentimen):
    """Penjelasan word cloud berdasarkan data sentimen yang sedang ditampilkan."""
    n_ulasan = int((df["sentimen"] == sentimen).sum())
    top = kata_teratas(df, sentimen, 8)
    if not top:
        return "Belum ada kata yang bisa dijelaskan."
    daftar = ", ".join(f"**{w}** ({n}x)" for w, n in top[:5])
    arti = ("hal-hal yang disukai atau dipuji pengguna" if sentimen == "positif"
            else "keluhan atau masalah yang dirasakan pengguna" if sentimen == "negatif"
            else "hal yang disampaikan secara biasa tanpa perasaan kuat")
    return (f"**Cara membaca:** semakin besar tulisan sebuah kata, semakin sering kata itu muncul "
            f"dalam ulasan {sentimen}. Gambar ini dibuat dari **{n_ulasan:,} ulasan {sentimen}**.\n\n"
            f"**Yang tampil:** kata yang paling menonjol adalah {daftar}. "
            f"Kata-kata ini menggambarkan {arti}.")


def penjelasan_wordcloud_gabungan(df):
    """Kata yang muncul di kedua word cloud."""
    a = {w for w, _ in kata_teratas(df, "positif", 15)}
    b = {w for w, _ in kata_teratas(df, "negatif", 15)}
    sama = sorted(a & b)
    if sama:
        return (f"**Catatan:** kata {', '.join(f'**{w}**' for w in sama)} muncul di kedua word cloud. "
                "Artinya kata tersebut sering dibicarakan baik oleh pengguna yang puas maupun yang kecewa, "
                "sehingga maknanya perlu dilihat dari kata-kata di sekitarnya.")
    return ("**Catatan:** kata-kata yang paling menonjol di sentimen positif dan negatif berbeda, "
            "sehingga kedua kelompok ulasan membicarakan hal yang cukup berbeda.")


def kualitas(v):
    if v >= 0.80:
        return "sangat baik"
    if v >= 0.70:
        return "baik"
    if v >= 0.60:
        return "cukup"
    return "masih perlu ditingkatkan"


def penjelasan_metrik(hasil):
    acc, pre = hasil.loc["Akurasi", "Testing"], hasil.loc["Precision", "Testing"]
    rec, f1 = hasil.loc["Recall", "Testing"], hasil.loc["F1-score", "Testing"]
    return (
        f"**Cara membaca:** nilai makin mendekati 100% berarti model makin pandai menebak sentimen.\n\n"
        f"- **Akurasi {acc:.1%}**: dari 100 ulasan baru, sekitar **{acc * 100:.0f}** ditebak dengan benar.\n"
        f"- **Precision {pre:.1%}**: ketika model menyebut suatu sentimen, tebakannya benar "
        f"sekitar **{pre * 100:.0f}%** (ukuran ketepatan).\n"
        f"- **Recall {rec:.1%}**: dari semua ulasan yang sebenarnya bersentimen tertentu, model berhasil "
        f"menemukan sekitar **{rec * 100:.0f}%** (ukuran kelengkapan).\n"
        f"- **F1-score {f1:.1%}**: gabungan precision dan recall; kinerja model secara keseluruhan "
        f"tergolong **{kualitas(f1)}**."
    )


def penjelasan_banding(hasil):
    tr, te = hasil.loc["Akurasi", "Training"], hasil.loc["Akurasi", "Testing"]
    gap = tr - te
    if gap > 0.10:
        isi = (f"Akurasi training ({tr:.1%}) jauh lebih tinggi daripada testing ({te:.1%}), selisih {gap:.1%}. "
               "Model cenderung terlalu hafal data latihnya dan kurang baik pada ulasan baru (overfitting).")
    elif gap > 0.03:
        isi = (f"Akurasi training ({tr:.1%}) sedikit lebih tinggi daripada testing ({te:.1%}), selisih {gap:.1%}. "
               "Ini masih wajar dan model tergolong cukup seimbang.")
    else:
        isi = (f"Akurasi training ({tr:.1%}) dan testing ({te:.1%}) hampir sama (selisih {abs(gap):.1%}). "
               "Model stabil dan tidak sekadar menghafal data latih.")
    return ("**Training vs testing:** training adalah nilai pada data yang dipakai belajar, "
            "testing adalah nilai pada data yang belum pernah dilihat model. " + isi)


def penjelasan_report(rep, labels):
    f1s = {l: rep[l]["f1-score"] for l in labels if l in rep}
    sup = {l: int(rep[l]["support"]) for l in labels if l in rep}
    best, worst = max(f1s, key=f1s.get), min(f1s, key=f1s.get)
    teks = (f"**Cara membaca:** tabel ini memecah kinerja model per sentimen. Kolom *support* adalah "
            f"jumlah ulasan testing untuk tiap sentimen ({', '.join(f'{l}: {sup[l]}' for l in sup)}).\n\n"
            f"Sentimen yang paling mudah dikenali model adalah **{best}** (F1 {f1s[best]:.1%}), "
            f"sedangkan yang paling sulit adalah **{worst}** (F1 {f1s[worst]:.1%}).")
    if sup[worst] == min(sup.values()) and worst != best:
        teks += f" Sebabnya kemungkinan jumlah ulasan {worst} paling sedikit sehingga model punya lebih sedikit contoh untuk belajar."
    return teks


def penjelasan_cm(cm, labels):
    total, benar = cm.sum(), np.trace(cm)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    teks = (f"**Cara membaca:** baris adalah sentimen sebenarnya, kolom adalah tebakan model. "
            f"Angka di diagonal (kiri atas ke kanan bawah) adalah tebakan yang benar; "
            f"makin gelap di diagonal, makin bagus.\n\n"
            f"Pada data testing, **{benar:,} dari {total:,} ulasan** ditebak dengan benar.")
    if off.max() > 0:
        i, j = np.unravel_index(off.argmax(), off.shape)
        teks += (f" Kesalahan terbanyak: ulasan yang sebenarnya **{labels[i]}** ditebak sebagai "
                 f"**{labels[j]}** sebanyak **{off[i, j]:,} kali**.")
    else:
        teks += " Tidak ada kesalahan tebakan pada data testing."
    return teks


def penjelasan_kata_berpengaruh(top_fitur):
    baris = [f"- **{k}**: {', '.join(f'*{w}*' for w in v[:5])}" for k, v in top_fitur.items()]
    return ("**Cara membaca:** ini kata-kata yang paling kuat mendorong model memilih suatu sentimen. "
            "Semakin atas peringkatnya, semakin besar pengaruhnya. "
            "Kata yang sering dipakai di banyak sentimen bisa muncul di lebih dari satu kolom.\n\n"
            "Lima kata teratas tiap sentimen saat ini:\n" + "\n".join(baris))


def penjelasan_preprocessing(raw, df):
    sebelum = df["content"].astype(str).str.split().str.len().mean()
    sesudah = df["text_clean"].str.split().str.len().mean()
    dibuang = len(raw) - len(df)
    return (f"**Ringkasan pada data Anda:** rata-rata ulasan awalnya **{sebelum:.1f} kata**, setelah dibersihkan "
            f"menjadi **{sesudah:.1f} kata** (berkurang {max(0, 1 - sesudah / sebelum):.0%}). "
            f"Sebanyak **{dibuang:,} ulasan** dibuang karena kosong setelah pembersihan.")


# ======================================================================
# SIDEBAR
# ======================================================================
st.sidebar.markdown("### ⚙️ Pengaturan")

st.sidebar.markdown("**📁 Data**")
upload = st.sidebar.file_uploader("Upload CSV hasil pelabelan leksikon", type="csv",
                                  label_visibility="collapsed")
st.sidebar.caption("Kolom wajib: `content`, `kategori`, `sentimen`.")

raw = None
if upload is not None:
    raw = pd.read_csv(upload)
else:
    try:
        raw = pd.read_csv(DEFAULT_CSV)
        st.sidebar.success(f"Memakai file lokal: {DEFAULT_CSV}")
    except FileNotFoundError:
        pass

st.sidebar.markdown("---")
st.sidebar.markdown("**🧠 Parameter model**")
test_size = st.sidebar.slider("Ukuran data testing", 0.1, 0.4, 0.2, 0.05,
                              help="Porsi ulasan yang disimpan untuk menguji model. 0.2 berarti 20% untuk uji, 80% untuk belajar.")
alpha = st.sidebar.slider("Alpha (Naive Bayes)", 0.1, 2.0, 0.5, 0.1,
                          help="Pengaturan 'kehalusan' model agar tidak kaku pada kata yang jarang muncul.")
ngram_max = st.sidebar.selectbox("N-gram maksimum", [1, 2, 3], index=1,
                                 help="1 = kata tunggal, 2 = kata tunggal + pasangan 2 kata, dst.")
min_df = st.sidebar.slider("min_df TF-IDF", 1, 5, 2,
                           help="Kata harus muncul minimal di sekian ulasan agar dipakai model.")
balance = st.sidebar.checkbox("Balancing (oversampling) data training", value=True,
                              help="Menyamakan jumlah ulasan tiap sentimen pada data latih agar model tidak berat sebelah.")

# ======================================================================
# HEADER
# ======================================================================
st.title("🎮 Klasifikasi Sentimen Ulasan Roblox")
st.caption("TF-IDF + Multinomial Naive Bayes · pelabelan leksikon per kategori")

if raw is None:
    st.info("Upload file `roblox_ulasan_dominan.csv` di sidebar untuk memulai "
            "(atau taruh file itu satu folder dengan `app.py`).")
    st.stop()

missing = [c for c in ["content", "kategori", "sentimen"] if c not in raw.columns]
if missing:
    st.error(f"Kolom tidak ditemukan di CSV: {missing}")
    st.stop()

df = load_and_clean(raw)
labels_aktif = [l for l in LABELS if l in df["sentimen"].values]

if df["sentimen"].nunique() < 2:
    st.error("Data harus punya minimal 2 kelas sentimen.")
    st.stop()

m = train_model(df, test_size, alpha, min_df, ngram_max, balance)
train_df, test_df = m["train_df"], m["test_df"]
tfidf, nb = m["tfidf"], m["nb"]

tabs = st.tabs(["📄 Data", "🧹 Preprocessing", "☁️ Word Cloud", "📊 Evaluasi",
                "🔮 Prediksi", "💬 Kalimat Sering", "💼 Nilai Bisnis"])

# ---------------------------------------------------------------- Data
with tabs[0]:
    c1, c2, c3 = st.columns(3)
    c1.metric("Baris data mentah", len(raw))
    c2.metric("Setelah buang teks kosong", len(df))
    c3.metric("Jumlah fitur TF-IDF", len(tfidf.get_feature_names_out()))

    st.subheader("Distribusi sentimen")
    col_a, col_b = st.columns(2)
    with col_a:
        st.dataframe(df["sentimen"].value_counts().rename("Jumlah"), use_container_width=True)
    with col_b:
        dist = pd.DataFrame({
            "Training (asli)": train_df["sentimen"].value_counts(),
            "Training (dipakai fit)": m["train_fit"]["sentimen"].value_counts(),
            "Testing": test_df["sentimen"].value_counts(),
        }).fillna(0).astype(int).loc[labels_aktif]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        dist.plot(kind="bar", ax=ax, color=["#6baed6", "#3182bd", "#fd8d3c"], width=0.8)
        for c in ax.containers:
            ax.bar_label(c, fontsize=8, padding=2)
        ax.set_xlabel(""); plt.xticks(rotation=0); plt.tight_layout()
        st.pyplot(fig)
    st.dataframe(dist, use_container_width=True)

    st.subheader("Cuplikan data")
    cols = [c for c in ["content", "kategori", "sentimen", "text_clean"] if c in df.columns]
    st.dataframe(df[cols].head(50), use_container_width=True)

# ------------------------------------------------------- Preprocessing
with tabs[1]:
    st.markdown("Delapan tahap preprocessing: case folding → hapus URL/mention → hapus simbol → "
                "huruf berulang → tokenisasi → stopword (negasi dipertahankan) → stemming "
                "(Sastrawi) → filter token pendek.")
    if get_stemmer() is None:
        st.warning("Sastrawi tidak terpasang, memakai stemmer sederhana. "
                   "Jalankan `pip install Sastrawi`.")
    n = st.number_input("Jumlah contoh", 1, 10, 4)
    contoh = df["content"].sample(min(int(n), len(df)), random_state=RANDOM_STATE).tolist()
    tabel = pd.DataFrame([preprocess(t, True) for t in contoh], columns=STAGE_NAMES).T
    tabel.columns = [f"Contoh {i + 1}" for i in range(len(contoh))]
    st.dataframe(tabel, use_container_width=True)

    st.info("**Apa itu preprocessing?** Ulasan pengguna biasanya berantakan: ada singkatan, tanda baca, "
            "huruf berulang, dan kata yang tidak penting. Komputer tidak bisa langsung memahaminya, "
            "jadi teks dibersihkan bertahap seperti pada tabel di atas. Baca dari atas ke bawah "
            "untuk melihat perubahan satu ulasan dari bentuk asli sampai siap dianalisis.")
    st.markdown("**Penjelasan tiap tahap**")
    st.dataframe(PENJELASAN_TAHAP.set_index("Tahap"), use_container_width=True)
    st.success(penjelasan_preprocessing(raw, df))

# ----------------------------------------------------------- Word Cloud
with tabs[2]:
    cols = st.columns(2)
    for col, (sent, cmap, judul) in zip(cols, [("positif", "Greens", "POSITIF"),
                                                ("negatif", "Reds", "NEGATIF")]):
        with col:
            st.subheader(f"Sentimen {judul}")
            teks = " ".join(df.loc[df["sentimen"] == sent, "text_clean"])
            if teks.strip():
                st.pyplot(wordcloud_fig(teks, cmap))
                st.info(penjelasan_wordcloud(df, sent))
            else:
                st.write("Tidak ada data.")
    if all(s in df["sentimen"].values for s in ["positif", "negatif"]):
        st.success(penjelasan_wordcloud_gabungan(df))

# -------------------------------------------------------------- Evaluasi
with tabs[3]:
    y_train, y_test = train_df["sentimen"], test_df["sentimen"]
    hasil = pd.DataFrame({"Training": hitung(y_train, m["pred_train"]),
                          "Testing": hitung(y_test, m["pred_test"])})

    st.subheader("Metrik (macro average)")
    mc = st.columns(4)
    for c, (nama, val) in zip(mc, hasil["Testing"].items()):
        c.metric(f"{nama} (testing)", f"{val:.2%}",
                 delta=f"{val - hasil.loc[nama, 'Training']:+.2%} vs training")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.dataframe(hasil.round(4), use_container_width=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        hasil.plot(kind="bar", ax=ax, color=["#3182bd", "#fd8d3c"], width=0.7)
        for c in ax.containers:
            ax.bar_label(c, labels=[f"{v:.1%}" for v in c.datavalues], fontsize=8, padding=2)
        ax.set_ylim(0, 1.15); plt.xticks(rotation=0); plt.tight_layout()
        st.pyplot(fig)
    with col_b:
        st.markdown("**Classification report (testing)**")
        rep = classification_report(y_test, m["pred_test"], labels=labels_aktif,
                                    output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).T.round(4), use_container_width=True)
        st.info(penjelasan_report(rep, labels_aktif))

    st.info(penjelasan_metrik(hasil))
    st.info(penjelasan_banding(hasil))

    st.subheader("Confusion matrix")
    cm_tr = confusion_matrix(y_train, m["pred_train"], labels=labels_aktif)
    cm_te = confusion_matrix(y_test, m["pred_test"], labels=labels_aktif)
    cm_te_n = cm_te / np.maximum(cm_te.sum(axis=1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, cm, judul, fmt in [(axes[0], cm_tr, "Training", "d"),
                               (axes[1], cm_te, "Testing", "d"),
                               (axes[2], cm_te_n, "Testing (per baris)", ".1%")]:
        sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", cbar=False, ax=ax,
                    xticklabels=labels_aktif, yticklabels=labels_aktif)
        ax.set_title(judul, fontweight="bold")
        ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual")
    plt.tight_layout()
    st.pyplot(fig)
    st.info(penjelasan_cm(cm_te, labels_aktif) +
            "\n\nGrafik ketiga menampilkan persentase per baris: berapa persen ulasan tiap sentimen "
            "yang berhasil ditebak benar.")

    st.subheader("Kata paling berpengaruh per kelas (Naive Bayes)")
    fitur = np.array(tfidf.get_feature_names_out())
    top_fitur = {k: fitur[np.argsort(nb.feature_log_prob_[i])[::-1][:12]]
                 for i, k in enumerate(nb.classes_)}
    st.dataframe(pd.DataFrame(top_fitur).rename_axis("Peringkat").rename(index=lambda i: i + 1),
                 use_container_width=True)
    st.info(penjelasan_kata_berpengaruh(top_fitur))

# -------------------------------------------------------------- Prediksi
with tabs[4]:
    st.subheader("Coba prediksi ulasan baru")
    teks = st.text_area("Tulis ulasan:", "game nya seru banget tapi sering lag dan harga robux mahal",
                        height=100)
    if st.button("Prediksi", type="primary"):
        bersih = preprocess(teks)
        if not bersih:
            st.warning("Teks kosong setelah preprocessing. Coba kalimat lain.")
        else:
            vec = tfidf.transform([bersih])
            proba = nb.predict_proba(vec)[0]
            pred = nb.classes_[proba.argmax()]
            warna = {"positif": "success", "negatif": "error", "netral": "warning"}[pred]
            getattr(st, warna)(f"Prediksi: **{pred.upper()}** ({proba.max():.1%})")
            st.write("Teks setelah preprocessing:", f"`{bersih}`")
            pdf = pd.DataFrame({"Kelas": nb.classes_, "Probabilitas": proba})
            fig, ax = plt.subplots(figsize=(6, 2.5))
            ax.barh(pdf["Kelas"], pdf["Probabilitas"],
                    color=[PALET.get(k, "#999") for k in pdf["Kelas"]])
            ax.set_xlim(0, 1)
            for i, v in enumerate(pdf["Probabilitas"]):
                ax.text(v + 0.01, i, f"{v:.1%}", va="center")
            plt.tight_layout()
            st.pyplot(fig)

    st.markdown("---")
    st.subheader("Prediksi massal (CSV)")
    batch = st.file_uploader("CSV dengan kolom `content`", type="csv", key="batch")
    if batch is not None:
        bdf = pd.read_csv(batch)
        if "content" not in bdf.columns:
            st.error("Kolom `content` tidak ditemukan.")
        else:
            bdf["text_clean"] = bdf["content"].apply(preprocess)
            bdf["prediksi"] = nb.predict(tfidf.transform(bdf["text_clean"]))
            st.dataframe(bdf, use_container_width=True)
            st.download_button("Unduh hasil", bdf.to_csv(index=False).encode("utf-8"),
                               "hasil_prediksi.csv", "text/csv")

# --------------------------------------------------------- Kalimat sering
with tabs[5]:
    top_n = st.slider("Jumlah baris", 5, 30, 10)
    for sent in ["positif", "negatif"]:
        if sent not in df["sentimen"].values:
            continue
        st.subheader(f"Kalimat {sent} yang sering muncul")
        st.dataframe(tabel_kalimat_sering(df, sent, top_n), use_container_width=True)
        st.subheader(f"Frasa 2 kata {sent}")
        st.dataframe(top_bigram(df, sent, top_n), use_container_width=True)

# ---------------------------------------------------------- Nilai bisnis
with tabs[6]:
    ct = pd.crosstab(df["kategori"], df["sentimen"])[labels_aktif]
    ct["Total"] = ct.sum(axis=1)
    if "negatif" in ct.columns:
        ct["% Negatif"] = (ct["negatif"] / ct["Total"] * 100).round(1)
    if "positif" in ct.columns:
        ct["% Positif"] = (ct["positif"] / ct["Total"] * 100).round(1)

    st.subheader("Sebaran sentimen per kategori")
    st.dataframe(ct.sort_values("% Negatif", ascending=False), use_container_width=True)

    pct = pd.crosstab(df["kategori"], df["sentimen"], normalize="index")[labels_aktif] * 100
    pct = pct.loc[ct.sort_values("% Negatif").index]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    pct.plot(kind="barh", stacked=True, ax=ax, color=[PALET[l] for l in labels_aktif])
    ax.set_xlabel("Persentase (%)"); ax.set_ylabel("")
    ax.legend(title="Sentimen", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    st.pyplot(fig)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Kata dominan: negatif")
        st.dataframe(kata_dominan(df, "negatif"), use_container_width=True)
    with c2:
        st.subheader("Kata dominan: positif")
        st.dataframe(kata_dominan(df, "positif"), use_container_width=True)

    st.subheader("Rekomendasi")
    st.markdown("""
**Sentimen positif: pertahankan**
1. **Amplifikasi konten populer.** Gameplay & Konten adalah penopang volume terbesar; beri panggung pada kreator dan map yang sering dipuji.
2. **Manfaatkan kepuasan umum.** Picu permintaan ulasan setelah sesi bermain yang menyenangkan.
3. **Transformasi monetisasi.** Sentimen positif harga rendah, jadi pertimbangkan paket ekonomis dan regional pricing Indonesia.

**Sentimen negatif: perbaiki (urut prioritas)**
1. **Performa untuk HP spek rendah:** mode grafis hemat, batasi objek/pemain per server, perbaiki crash saat loading.
2. **Verifikasi umur dan chat:** panduan Bahasa Indonesia, alternatif verifikasi, pesan error jelas, jalur banding cepat.
3. **Keamanan akun dan anti-cheat:** captcha tidak berulang, pemulihan akun cepat, banding ban transparan.
4. **Harga lebih adil per wilayah:** paket Robux lokal, promo berkala, komunikasi jelas saat harga berubah.
5. **Komunikasi update:** catatan rilis Bahasa Indonesia, rollout bertahap, balas ulasan negatif dengan solusi.
6. **Pantau sentimen berkala** memakai model ini pada ulasan baru.
""")
