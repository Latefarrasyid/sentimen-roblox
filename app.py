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
# HELPER PENJELASAN OTOMATIS
# ======================================================================
def top_kata(df, sentimen, kategori=None, n=5):
    d = df[df["sentimen"] == sentimen]
    if kategori is not None:
        d = d[d["kategori"] == kategori]
    c = Counter(w for t in d["text_clean"] for w in t.split() if w not in NEGASI)
    return [w for w, _ in c.most_common(n)]


def fmt_kata(kata):
    return ", ".join(f"**{w}**" for w in kata) if kata else "-"


def pct(x):
    return f"{x:.1%}"


def tingkat(v):
    if v >= 0.85:
        return "sangat baik"
    if v >= 0.70:
        return "baik"
    if v >= 0.55:
        return "cukup"
    return "masih kurang"


def jelaskan(md, judul="📖 Penjelasan untuk pemula", expanded=True):
    """Tampilkan kotak penjelasan jika toggle di sidebar aktif."""
    if st.session_state.get("show_help", True):
        with st.expander(judul, expanded=expanded):
            st.markdown(md)


# ======================================================================
# SIDEBAR
# ======================================================================
DEFAULT_PARAM = dict(p_test=0.2, p_alpha=0.5, p_ngram=2, p_mindf=2, p_balance=True)
for _k, _v in DEFAULT_PARAM.items():
    st.session_state.setdefault(_k, _v)
st.session_state.setdefault("show_help", True)


def reset_param():
    for k, v in DEFAULT_PARAM.items():
        st.session_state[k] = v


with st.sidebar:
    st.title("⚙️ Pengaturan")

    # ---- 1. Data
    st.markdown("### 1️⃣ Data")
    upload = st.file_uploader("Upload CSV hasil pelabelan leksikon", type="csv",
                              help="File harus punya kolom: content (teks ulasan), "
                                   "kategori (topik ulasan), sentimen (positif/netral/negatif).")
    st.caption("Kolom wajib: `content`, `kategori`, `sentimen`")

    raw = None
    if upload is not None:
        raw = pd.read_csv(upload)
        st.success(f"✅ File diupload: {upload.name}")
    else:
        try:
            raw = pd.read_csv(DEFAULT_CSV)
            st.success(f"✅ Memakai file lokal: {DEFAULT_CSV}")
        except FileNotFoundError:
            st.warning("Belum ada data. Upload CSV di atas.")

    st.divider()

    # ---- 2. Parameter model
    st.markdown("### 2️⃣ Parameter model")
    with st.expander("Atur parameter (opsional)", expanded=False):
        st.caption("Nilai bawaan sudah cukup baik. Ubah hanya jika ingin bereksperimen.")
        test_size = st.slider("Porsi data testing", 0.1, 0.4, step=0.05, key="p_test",
                              help="Bagian data yang disimpan untuk menguji model. "
                                   "0.2 artinya 20% data untuk ujian, 80% untuk belajar.")
        alpha = st.slider("Alpha (smoothing)", 0.1, 2.0, step=0.1, key="p_alpha",
                          help="Penghalus agar model tidak terlalu yakin pada kata yang jarang muncul. "
                               "Makin besar = makin 'hati-hati'.")
        ngram_max = st.selectbox("N-gram maksimum", [1, 2, 3], key="p_ngram",
                                 help="1 = hanya kata tunggal. 2 = kata tunggal + pasangan dua kata "
                                      "(mis. 'tidak bagus'). 3 = sampai tiga kata.")
        min_df = st.slider("Minimal kemunculan kata", 1, 5, key="p_mindf",
                           help="Kata yang muncul di kurang dari N ulasan diabaikan "
                                "(biasanya salah ketik atau terlalu langka).")
        balance = st.checkbox("Seimbangkan data training", key="p_balance",
                              help="Memperbanyak contoh kelas yang sedikit supaya model tidak "
                                   "hanya hafal kelas yang paling banyak.")
        st.button("↩️ Kembalikan ke nilai bawaan", on_click=reset_param, use_container_width=True)

    st.divider()

    # ---- 3. Tampilan
    st.markdown("### 3️⃣ Tampilan")
    st.toggle("Tampilkan penjelasan untuk pemula", key="show_help",
              help="Matikan jika Anda sudah paham dan ingin tampilan lebih ringkas.")

    st.divider()
    st.caption("TF-IDF + Multinomial Naive Bayes")

# ======================================================================
# HEADER
# ======================================================================
st.title("🎮 Klasifikasi Sentimen Ulasan Roblox")
st.caption("TF-IDF + Multinomial Naive Bayes · pelabelan leksikon per kategori")

if raw is None:
    st.info("Upload file `roblox_ulasan_dominan.csv` di sidebar untuk memulai "
            "(atau taruh file itu satu folder dengan `app.py`).")
    jelaskan("""
**Dashboard ini untuk apa?** Membaca ribuan ulasan pemain Roblox lalu menyimpulkan apakah
ulasan itu **positif**, **netral**, atau **negatif**, dan topik apa yang paling banyak dikeluhkan.

**Cara pakai:** (1) upload file CSV di sidebar kiri → (2) buka tab satu per satu dari kiri ke kanan →
(3) baca kotak *Penjelasan* di tiap tab, isinya otomatis menyesuaikan dengan data Anda.
""", "📖 Cara pakai")
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

jelaskan(f"""
**Dashboard ini untuk apa?** Membaca **{len(df):,} ulasan** Roblox lalu menyimpulkan apakah tiap ulasan
**positif**, **netral**, atau **negatif**, dan topik apa yang paling banyak dikeluhkan.

**Urutan membaca:** 📄 Data → 🧹 Preprocessing → ☁️ Word Cloud → 📊 Evaluasi → 🔮 Prediksi →
💬 Kalimat Sering → 💼 Nilai Bisnis. Di setiap tab ada kotak *Penjelasan* yang otomatis
menyesuaikan dengan data dan pengaturan Anda saat ini.
""", "📖 Cara membaca dashboard ini", expanded=False)

tabs = st.tabs(["📄 Data", "🧹 Preprocessing", "☁️ Word Cloud", "📊 Evaluasi",
                "🔮 Prediksi", "💬 Kalimat Sering", "💼 Nilai Bisnis"])

# ---------------------------------------------------------------- Data
with tabs[0]:
    vc = df["sentimen"].value_counts()
    n_dibuang = len(raw) - len(df)
    n_fitur = len(tfidf.get_feature_names_out())

    # --- penjelasan otomatis
    dom, minor = vc.index[0], vc.index[-1]
    rasio = vc.iloc[0] / max(vc.iloc[-1], 1)
    kalimat_seimbang = (
        f"Sebaran sentimen cukup seimbang (rasio terbesar:terkecil ≈ {rasio:.1f}:1)."
        if rasio < 2 else
        f"Sebaran **tidak seimbang**: kelas **{dom}** {rasio:.1f}× lebih banyak dari kelas **{minor}**. "
        + ("Pengaturan *Seimbangkan data training* sedang **aktif**, jadi model tetap diberi porsi belajar "
           "yang adil untuk kelas yang sedikit."
           if balance else
           "Pengaturan *Seimbangkan data training* sedang **mati**, sehingga model cenderung lebih pandai "
           "menebak kelas yang banyak. Coba aktifkan di sidebar.")
    )
    jelaskan(f"""
- File Anda berisi **{len(raw):,} ulasan**. {f'Sebanyak **{n_dibuang:,}** ulasan dibuang karena kosong setelah dibersihkan (misalnya hanya berisi emoji atau simbol).' if n_dibuang else 'Tidak ada ulasan yang dibuang.'} Tersisa **{len(df):,}** ulasan yang dipakai.
- Ulasan terbagi ke **{df['kategori'].nunique()} kategori topik**; yang paling banyak adalah **{df['kategori'].value_counts().index[0]}** ({df['kategori'].value_counts().iloc[0]:,} ulasan).
- Sentimen terbanyak adalah **{dom}** ({pct(vc.iloc[0] / len(df))}), paling sedikit **{minor}** ({pct(vc.iloc[-1] / len(df))}). {kalimat_seimbang}
- Data dibagi dua: **{len(train_df):,} ulasan untuk melatih** model (seperti bahan belajar) dan **{len(test_df):,} ulasan untuk menguji** (seperti soal ujian yang belum pernah dilihat).
- Model mengenali **{n_fitur:,} "fitur"**, yaitu kata atau pasangan kata yang dijadikan petunjuk untuk menebak sentimen.
""")

    c1, c2, c3 = st.columns(3)
    c1.metric("Baris data mentah", len(raw))
    c2.metric("Setelah buang teks kosong", len(df))
    c3.metric("Jumlah fitur TF-IDF", n_fitur)

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
    wc_asli = df["content"].astype(str).str.split().str.len().mean()
    wc_bersih = df["text_clean"].str.split().str.len().mean()
    turun = 1 - wc_bersih / wc_asli if wc_asli else 0
    vocab_unik = len({w for t in df["text_clean"] for w in t.split()})
    stem_info = ("memakai **Sastrawi** (stemmer bahasa Indonesia yang akurat)"
                 if get_stemmer() is not None else
                 "memakai **stemmer sederhana** karena Sastrawi belum terpasang")

    jelaskan(f"""
**Preprocessing = membersihkan teks** supaya komputer tidak bingung dengan variasi penulisan.
Contohnya "BAGUUUS!!!" dan "bagus" seharusnya dianggap kata yang sama.

- Rata-rata ulasan Anda awalnya **{wc_asli:.1f} kata**, setelah dibersihkan menjadi **{wc_bersih:.1f} kata** (berkurang ≈ **{turun:.0%}**). Yang dibuang adalah kata sambung, simbol, dan kata tidak bermakna.
- Setelah dibersihkan, ada **{vocab_unik:,} kata unik** di seluruh data.
- Kata penolak seperti *tidak, bukan, belum, jangan, kurang* **sengaja dipertahankan**, karena "tidak bagus" artinya kebalikan dari "bagus".
- Tahap stemming (mengembalikan kata ke bentuk dasar, mis. "memperbaiki" → "baik") saat ini {stem_info}.
- Di tabel bawah, baca dari atas ke bawah untuk melihat perubahan teks di tiap tahap.
""")

    st.markdown("Delapan tahap preprocessing: case folding → hapus URL/mention → hapus simbol → "
                "huruf berulang → tokenisasi → stopword (negasi dipertahankan) → stemming "
                "(Sastrawi) → filter token pendek.")
    if get_stemmer() is None:
        st.warning("Sastrawi tidak terpasang, memakai stemmer sederhana. "
                   "Jalankan `pip install Sastrawi`.")
    n = st.number_input("Jumlah contoh", 1, 10, 4)
    contoh = df["content"].sample(int(n), random_state=RANDOM_STATE).tolist()
    tabel = pd.DataFrame([preprocess(t, True) for t in contoh], columns=STAGE_NAMES).T
    tabel.columns = [f"Contoh {i + 1}" for i in range(len(contoh))]
    st.dataframe(tabel, use_container_width=True)

# ----------------------------------------------------------- Word Cloud
with tabs[2]:
    kata_pos = top_kata(df, "positif", n=15) if "positif" in labels_aktif else []
    kata_neg = top_kata(df, "negatif", n=15) if "negatif" in labels_aktif else []
    irisan = [w for w in kata_pos if w in kata_neg]

    teks_wc = "- **Word cloud** = gambar kumpulan kata; **makin besar ukuran kata, makin sering muncul**.\n"
    if kata_pos:
        teks_wc += f"- Kata paling sering di ulasan **positif**: {fmt_kata(kata_pos[:5])}.\n"
    if kata_neg:
        teks_wc += f"- Kata paling sering di ulasan **negatif**: {fmt_kata(kata_neg[:5])}.\n"
    if irisan:
        teks_wc += (f"- Kata yang muncul besar di **kedua** gambar: {fmt_kata(irisan[:5])}. "
                    "Artinya kata itu topik umum yang dibicarakan semua orang, bukan penentu "
                    "positif/negatif.\n")
    teks_wc += ("- Gunakan gambar ini untuk **menangkap tema** dengan cepat, tapi jangan dipakai sebagai "
                "angka pasti. Angka lebih rinci ada di tab *Kalimat Sering* dan *Nilai Bisnis*.")
    jelaskan(teks_wc)

    cols = st.columns(2)
    for col, (sent, cmap, judul) in zip(cols, [("positif", "Greens", "POSITIF"),
                                                ("negatif", "Reds", "NEGATIF")]):
        with col:
            st.subheader(f"Sentimen {judul}")
            teks = " ".join(df.loc[df["sentimen"] == sent, "text_clean"])
            if teks.strip():
                st.pyplot(wordcloud_fig(teks, cmap))
            else:
                st.write("Tidak ada data.")

# -------------------------------------------------------------- Evaluasi
with tabs[3]:
    y_train, y_test = train_df["sentimen"], test_df["sentimen"]
    hasil = pd.DataFrame({"Training": hitung(y_train, m["pred_train"]),
                          "Testing": hitung(y_test, m["pred_test"])})
    rep = classification_report(y_test, m["pred_test"], labels=labels_aktif,
                                output_dict=True, zero_division=0)
    cm_te = confusion_matrix(y_test, m["pred_test"], labels=labels_aktif)

    acc_te = hasil.loc["Akurasi", "Testing"]
    acc_tr = hasil.loc["Akurasi", "Training"]
    gap = acc_tr - acc_te
    f1_kelas = {k: rep[k]["f1-score"] for k in labels_aktif}
    terbaik = max(f1_kelas, key=f1_kelas.get)
    terburuk = min(f1_kelas, key=f1_kelas.get)

    cm_off = cm_te.copy()
    np.fill_diagonal(cm_off, 0)
    if cm_off.sum() > 0:
        i, j = np.unravel_index(cm_off.argmax(), cm_off.shape)
        kalimat_salah = (f"Kesalahan paling sering: ulasan yang sebenarnya **{labels_aktif[i]}** "
                         f"ditebak **{labels_aktif[j]}** sebanyak **{cm_off[i, j]:,}** kali.")
    else:
        kalimat_salah = "Model tidak membuat kesalahan sama sekali pada data testing."

    if gap > 0.10:
        kalimat_gap = (f"⚠️ Nilai training ({pct(acc_tr)}) jauh lebih tinggi dari testing ({pct(acc_te)}). "
                       "Ini tanda model **terlalu menghafal** data latihnya (*overfitting*). "
                       "Coba naikkan Alpha atau Minimal kemunculan kata di sidebar.")
    elif gap < -0.02:
        kalimat_gap = (f"Nilai testing ({pct(acc_te)}) sedikit lebih tinggi dari training ({pct(acc_tr)}). "
                       "Ini wajar saat data training diseimbangkan atau data kecil.")
    else:
        kalimat_gap = (f"Nilai training ({pct(acc_tr)}) dan testing ({pct(acc_te)}) berdekatan, "
                       "tanda model **stabil** dan tidak sekadar menghafal.")

    jelaskan(f"""
**Evaluasi = nilai ujian model.** Model diuji dengan **{len(test_df):,} ulasan** yang belum pernah dilihat saat belajar.

- **Akurasi {pct(acc_te)}** → dari 100 ulasan baru, sekitar **{round(acc_te * 100)}** ditebak dengan benar. Kategori: **{tingkat(acc_te)}**.
- {kalimat_gap}
- Kelas yang **paling mudah** ditebak: **{terbaik}** (F1 {pct(f1_kelas[terbaik])}). Yang **paling sulit**: **{terburuk}** (F1 {pct(f1_kelas[terburuk])}).
- {kalimat_salah}
- ℹ️ Label "benar" di sini berasal dari **pelabelan leksikon** (kamus kata), bukan penilaian manusia. Jadi angka ini mengukur seberapa baik model meniru aturan kamus tersebut.
""")

    jelaskan("""
- **Akurasi**: persentase tebakan yang benar dari seluruh data.
- **Precision**: saat model bilang "negatif", seberapa sering itu benar?
- **Recall**: dari semua ulasan yang benar-benar negatif, berapa yang berhasil ditemukan model?
- **F1-score**: gabungan precision dan recall; makin dekat 100% makin baik.
- **Macro average**: rata-rata biasa dari semua kelas, jadi kelas kecil dihitung sama pentingnya.
- **Confusion matrix**: tabel tebakan vs kenyataan. Angka di **diagonal** (kiri-atas ke kanan-bawah) adalah tebakan yang benar; angka di luar diagonal adalah kesalahan.
""", "📚 Kamus istilah", expanded=False)

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
        st.dataframe(pd.DataFrame(rep).T.round(4), use_container_width=True)

    st.subheader("Confusion matrix")
    cm_tr = confusion_matrix(y_train, m["pred_train"], labels=labels_aktif)
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

    st.subheader("Kata paling berpengaruh per kelas (Naive Bayes)")
    fitur = np.array(tfidf.get_feature_names_out())
    top_fitur = {k: fitur[np.argsort(nb.feature_log_prob_[i])[::-1][:12]]
                 for i, k in enumerate(nb.classes_)}
    jelaskan("Tabel ini menunjukkan **12 kata/frasa yang paling kuat mengarahkan model** ke tiap sentimen. "
             "Kata yang muncul di semua kolom (mis. 'game') hanya kata umum; yang khas di satu kolom "
             "adalah petunjuk utama sentimen itu.", "📖 Cara membaca tabel kata berpengaruh",
             expanded=False)
    st.dataframe(pd.DataFrame(top_fitur).rename_axis("Peringkat").rename(index=lambda i: i + 1),
                 use_container_width=True)

# -------------------------------------------------------------- Prediksi
with tabs[4]:
    jelaskan(f"""
Di sini Anda bisa **mencoba model secara langsung**. Tulis satu ulasan (Bahasa Indonesia) lalu klik *Prediksi*.

- Model akan memberi **tebakan sentimen** dan **tingkat keyakinan** (probabilitas). Makin tinggi persentase, makin yakin.
- Model hanya mengenali kata yang ada di data latihnya (**{n_fitur:,} fitur**). Kata baru atau slang yang belum pernah muncul akan diabaikan.
- Untuk banyak ulasan sekaligus, pakai bagian **Prediksi massal**: upload CSV berkolom `content`, lalu unduh hasilnya.
""")

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

            # --- penjelasan otomatis hasil prediksi
            urut = np.argsort(proba)[::-1]
            kedua = nb.classes_[urut[1]] if len(urut) > 1 else None
            yakin = proba.max()
            if yakin >= 0.70:
                tingkat_yakin = "**cukup yakin**"
            elif yakin >= 0.50:
                tingkat_yakin = "**agak ragu**"
            else:
                tingkat_yakin = "**sangat ragu** (ulasan mungkin bernada campuran)"
            token = bersih.split()
            kenal = [t for t in token if t in tfidf.vocabulary_]
            asing = [t for t in token if t not in tfidf.vocabulary_]
            teks_hasil = (f"- Model menebak **{pred}** dan {tingkat_yakin} ({pct(yakin)}).\n")
            if kedua is not None:
                teks_hasil += (f"- Alternatif terdekat: **{kedua}** ({pct(proba[urut[1]])}).\n")
            teks_hasil += (f"- Dari **{len(token)} kata** setelah dibersihkan, **{len(kenal)}** dikenali model"
                           + (f" dan **{len(asing)}** tidak dikenali ({fmt_kata(asing[:6])}), "
                              "jadi tidak ikut memengaruhi tebakan." if asing else ". Semua kata dikenali."))
            jelaskan(teks_hasil, "📖 Mengapa hasilnya begini?")

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
            vb = bdf["prediksi"].value_counts()
            ringkas = ", ".join(f"**{k}** {v:,} ({pct(v / len(bdf))})" for k, v in vb.items())
            kosong = int((bdf["text_clean"].str.strip() == "").sum())
            jelaskan(f"File Anda berisi **{len(bdf):,} ulasan**. Hasil prediksi: {ringkas}."
                     + (f" Ada **{kosong:,}** ulasan yang kosong setelah dibersihkan, tebakannya kurang "
                        "bisa dipercaya." if kosong else "")
                     + " Tombol di bawah tabel mengunduh hasil lengkap dalam format CSV.",
                     "📖 Ringkasan hasil prediksi massal")
            st.dataframe(bdf, use_container_width=True)
            st.download_button("Unduh hasil", bdf.to_csv(index=False).encode("utf-8"),
                               "hasil_prediksi.csv", "text/csv")

# --------------------------------------------------------- Kalimat sering
with tabs[5]:
    top_n = st.slider("Jumlah baris", 5, 30, 10)

    ringkas_kal = ("Tab ini menampilkan **kalimat dan frasa yang paling sering diulang** pemain, "
                   "jadi Anda bisa melihat keluhan atau pujian *dengan kata-kata pemain sendiri*.\n")
    tabel_cache = {}
    for sent in ["positif", "negatif"]:
        if sent not in df["sentimen"].values:
            continue
        tk = tabel_kalimat_sering(df, sent, top_n)
        tb = top_bigram(df, sent, top_n)
        tabel_cache[sent] = (tk, tb)
        if len(tk):
            ringkas_kal += (f"- Kalimat **{sent}** terbanyak: \"{tk.iloc[0]['Kalimat']}\" "
                            f"(**{int(tk.iloc[0]['Frekuensi'])}×**, kategori {tk.iloc[0]['Kategori dominan']}).\n")
        if len(tb):
            ringkas_kal += (f"- Frasa dua kata **{sent}** terbanyak: **{tb.iloc[0]['Frasa']}** "
                            f"({int(tb.iloc[0]['Frekuensi'])}×).\n")
    ringkas_kal += ("- *Frekuensi* = berapa kali muncul. Angka besar berarti banyak pemain merasakan hal yang sama.\n"
                    "- Frasa dua kata diambil dari teks yang sudah dibersihkan, jadi bentuknya bisa terlihat "
                    "terpotong atau berupa kata dasar.")
    jelaskan(ringkas_kal)

    for sent in ["positif", "negatif"]:
        if sent not in tabel_cache:
            continue
        tk, tb = tabel_cache[sent]
        st.subheader(f"Kalimat {sent} yang sering muncul")
        st.dataframe(tk, use_container_width=True)
        st.subheader(f"Frasa 2 kata {sent}")
        st.dataframe(tb, use_container_width=True)

# ---------------------------------------------------------- Nilai bisnis
with tabs[6]:
    ct = pd.crosstab(df["kategori"], df["sentimen"])[labels_aktif]
    ct["Total"] = ct[labels_aktif].sum(axis=1)
    if "negatif" in ct.columns:
        ct["% Negatif"] = (ct["negatif"] / ct["Total"] * 100).round(1)
    if "positif" in ct.columns:
        ct["% Positif"] = (ct["positif"] / ct["Total"] * 100).round(1)

    urut_neg = ct.sort_values("% Negatif", ascending=False) if "% Negatif" in ct.columns else ct
    urut_pos = ct.sort_values("% Positif", ascending=False) if "% Positif" in ct.columns else ct
    kat_terbesar = ct["Total"].idxmax()

    teks_bis = "Tab ini menerjemahkan hasil analisis menjadi **prioritas kerja** untuk tim produk.\n"
    teks_bis += (f"- Topik dengan ulasan **terbanyak**: **{kat_terbesar}** "
                 f"({int(ct.loc[kat_terbesar, 'Total']):,} ulasan).\n")
    if "% Negatif" in ct.columns:
        k = urut_neg.index[0]
        teks_bis += (f"- Topik dengan **keluhan paling tinggi (proporsi)**: **{k}** "
                     f"({urut_neg.loc[k, '% Negatif']}% negatif). Kata yang sering muncul: "
                     f"{fmt_kata(top_kata(df, 'negatif', k, 4))}.\n")
    if "% Positif" in ct.columns:
        k = urut_pos.index[0]
        teks_bis += (f"- Topik **paling disukai**: **{k}** ({urut_pos.loc[k, '% Positif']}% positif). "
                     f"Kata yang sering muncul: {fmt_kata(top_kata(df, 'positif', k, 4))}.\n")
    teks_bis += ("- Pada grafik batang, **semakin panjang warna merah, semakin banyak keluhan** di topik itu. "
                 "Urutan tabel dan grafik dimulai dari topik paling bermasalah.")
    jelaskan(teks_bis)

    st.subheader("Sebaran sentimen per kategori")
    st.dataframe(urut_neg, use_container_width=True)

    pct_df = pd.crosstab(df["kategori"], df["sentimen"], normalize="index")[labels_aktif] * 100
    if "% Negatif" in ct.columns:
        pct_df = pct_df.loc[ct.sort_values("% Negatif").index]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    pct_df.plot(kind="barh", stacked=True, ax=ax, color=[PALET[l] for l in labels_aktif])
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

    st.subheader("Prioritas perbaikan (otomatis dari data)")
    if "% Negatif" in ct.columns:
        prioritas = ""
        for no, (kat, row) in enumerate(urut_neg.head(3).iterrows(), start=1):
            prioritas += (f"{no}. **{kat}**: {int(row['negatif']):,} ulasan negatif dari "
                          f"{int(row['Total']):,} ({row['% Negatif']}%). "
                          f"Fokuskan pada isu: {fmt_kata(top_kata(df, 'negatif', kat, 4))}.\n")
        st.markdown(prioritas)
        st.caption("Tiga topik dengan proporsi keluhan tertinggi. Kata-kata di atas adalah petunjuk "
                   "isu yang dibahas, bukan kesimpulan final; baca ulasan aslinya untuk memastikan.")
    else:
        st.info("Data tidak memiliki sentimen negatif, jadi prioritas perbaikan tidak dapat dihitung.")

    with st.expander("Rekomendasi umum (berdasarkan analisis dataset awal)", expanded=False):
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
