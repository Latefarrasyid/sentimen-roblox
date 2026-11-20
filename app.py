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

# Kata kunci untuk koreksi di tab Prediksi (bentuk sudah dibersihkan/stem)
NEG_CUES = {"lag", "ngelag", "lemot", "lelet", "lambat", "mahal", "bug", "error", "eror",
            "crash", "macet", "gagal", "rusak", "jelek", "buruk", "parah", "kecewa",
            "bosan", "bosen", "ribet", "susah", "sulit", "hack", "hacker", "cheat",
            "scam", "tipu", "ban", "banned", "blokir", "hilang", "sampah", "toxic",
            "kesal", "sebel", "ganggu", "nyebelin", "disconnect", "kick"}
POS_CUES = {"seru", "bagus", "keren", "mantap", "mantul", "asik", "asyik", "suka",
            "senang", "puas", "menarik", "terbaik", "murah", "lancar", "mudah",
            "cinta", "top", "recommended", "worth", "puas", "hibur"}
KONTRAS = {"tapi", "tetapi", "namun", "cuma", "hanya", "sayang", "sayangnya", "sayangny"}


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
# PREDIKSI (Naive Bayes + koreksi kata kunci)
# ======================================================================
def skor_kata_kunci(tokens):
    """Hitung skor kata bernada positif/negatif. Memperhitungkan negasi
    ('tidak bagus' -> negatif) dan kata kontras ('tapi' -> kata sesudahnya lebih penting)."""
    pos = neg = 0.0
    kp, kn = [], []
    kontras = False
    for i, w in enumerate(tokens):
        if w in KONTRAS:
            kontras = True
            continue
        if w in NEG_CUES:
            pol = -1
        elif w in POS_CUES:
            pol = 1
        else:
            continue
        if i > 0 and tokens[i - 1] in NEGASI:
            pol = -pol
        bobot = 1.5 if kontras else 1.0
        if pol > 0:
            pos += bobot
            kp.append(w)
        else:
            neg += bobot
            kn.append(w)
    return pos, neg, kp, kn


def prediksi_teks(bersih, tfidf, nb, pakai_koreksi=True):
    """Kembalikan (proba_akhir, proba_nb_murni, kata_pos, kata_neg, dikoreksi)."""
    p_nb = nb.predict_proba(tfidf.transform([bersih]))[0]
    pos, neg, kp, kn = skor_kata_kunci(bersih.split())
    if not pakai_koreksi or (pos + neg) == 0:
        return p_nb, p_nb, kp, kn, False
    kelas = list(nb.classes_)
    p_lex = np.zeros(len(kelas))
    if abs(pos - neg) < 1e-9 and "netral" in kelas:
        p_lex[kelas.index("netral")] = 0.6
        if "positif" in kelas:
            p_lex[kelas.index("positif")] = 0.2
        if "negatif" in kelas:
            p_lex[kelas.index("negatif")] = 0.2
    else:
        if "positif" in kelas:
            p_lex[kelas.index("positif")] = pos / (pos + neg)
        if "negatif" in kelas:
            p_lex[kelas.index("negatif")] = neg / (pos + neg)
    if p_lex.sum() == 0:
        return p_nb, p_nb, kp, kn, False
    p_lex = p_lex / p_lex.sum()
    p = 0.3 * p_nb + 0.7 * p_lex
    return p / p.sum(), p_nb, kp, kn, True


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


def narasi_cm_jumlah(cm, labels, nama):
    total = int(cm.sum())
    benar = int(np.trace(cm))
    teks = (f"- Pada data **{nama}** ada **{total:,}** ulasan. Sebanyak **{benar:,}** ditebak benar "
            f"(**{pct(benar / max(total, 1))}**), yaitu angka di diagonal (kiri-atas ke kanan-bawah).\n")
    for i, l in enumerate(labels):
        n = int(cm[i].sum())
        if n == 0:
            continue
        d = int(cm[i, i])
        salah = n - d
        if salah > 0:
            off = cm[i].copy()
            off[i] = -1
            j = int(off.argmax())
            tambah = f" Salahnya paling banyak ditebak sebagai **{labels[j]}** ({int(cm[i, j]):,} ulasan)."
        else:
            tambah = " Tidak ada yang salah."
        teks += (f"- Ulasan yang sebenarnya **{l}** ada {n:,}: **{d:,}** benar ({pct(d / n)}), "
                 f"{salah:,} salah.{tambah}\n")
    return teks


def narasi_cm_persen(cm, labels):
    teks = ("- Setiap **baris** dijumlahkan menjadi 100%, jadi angkanya adil dibandingkan antar kelas "
            "walau jumlah ulasan tiap kelas berbeda.\n")
    recall_baris = {}
    for i, l in enumerate(labels):
        n = cm[i].sum()
        if n == 0:
            continue
        recall_baris[l] = cm[i, i] / n
        sisa = [(labels[j], cm[i, j] / n) for j in range(len(labels)) if j != i and cm[i, j] > 0]
        sisa_txt = ("; sisanya ditebak " + ", ".join(f"**{k}** {pct(v)}" for k, v in sisa)) if sisa else ""
        teks += f"- Dari semua ulasan **{l}**, **{pct(recall_baris[l])}** berhasil dikenali{sisa_txt}.\n"
    if recall_baris:
        best = max(recall_baris, key=recall_baris.get)
        worst = min(recall_baris, key=recall_baris.get)
        teks += (f"- Kelas yang **paling mudah dikenali**: **{best}** ({pct(recall_baris[best])}). "
                 f"Yang **paling sulit**: **{worst}** ({pct(recall_baris[worst])}).")
    return teks


# ======================================================================
# SIDEBAR
# ======================================================================
st.markdown("""
<style>
[data-testid="stSidebarUserContent"] { padding-top: 1rem; }
.sb-title { font-size: 1.3rem; font-weight: 700; line-height: 1.25; padding: .7rem .9rem;
            border-radius: .6rem; background: rgba(128,128,128,.14); margin-bottom: .6rem; }
.sb-title small { display: block; font-weight: 400; font-size: .78rem; opacity: .7; margin-top: .15rem; }
.sb-section { font-size: .78rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase;
              opacity: .75; margin: .3rem 0 .3rem 0; }
</style>
""", unsafe_allow_html=True)

DEFAULT_PARAM = dict(p_test=0.2, p_alpha=0.5, p_ngram=2, p_mindf=2, p_balance=True)
for _k, _v in DEFAULT_PARAM.items():
    st.session_state.setdefault(_k, _v)
st.session_state.setdefault("show_help", True)
st.session_state.setdefault("pakai_koreksi", True)


def reset_param():
    for k, v in DEFAULT_PARAM.items():
        st.session_state[k] = v


def sb_section(teks):
    st.sidebar.markdown(f'<div class="sb-section">{teks}</div>', unsafe_allow_html=True)


st.sidebar.markdown('<div class="sb-title">⚙️ Pengaturan'
                    '<small>Atur data dan model di sini</small></div>', unsafe_allow_html=True)

# ---- 1. Data
sb_section("1 · Data")
upload = st.sidebar.file_uploader("Upload CSV hasil pelabelan leksikon", type="csv",
                                  help="File harus punya kolom: content (teks ulasan), "
                                       "kategori (topik ulasan), sentimen (positif/netral/negatif).")
st.sidebar.caption("Kolom wajib: `content`, `kategori`, `sentimen`")

raw = None
if upload is not None:
    raw = pd.read_csv(upload)
    st.sidebar.success(f"✅ File diupload: {upload.name}")
else:
    try:
        raw = pd.read_csv(DEFAULT_CSV)
        st.sidebar.success(f"✅ Memakai file lokal: {DEFAULT_CSV}")
    except FileNotFoundError:
        st.sidebar.warning("Belum ada data. Upload CSV di atas.")

st.sidebar.divider()

# ---- 2. Parameter model
sb_section("2 · Parameter model")
with st.sidebar.expander("Atur parameter (opsional)", expanded=False):
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

st.sidebar.divider()

# ---- 3. Tampilan
sb_section("3 · Tampilan")
st.sidebar.toggle("Tampilkan penjelasan untuk pemula", key="show_help",
                  help="Matikan jika Anda sudah paham dan ingin tampilan lebih ringkas.")

st.sidebar.divider()
st.sidebar.caption("TF-IDF + Multinomial Naive Bayes")

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
n_fitur = len(tfidf.get_feature_names_out())

# ---------------------------------------------------------------- Data
with tabs[0]:
    vc = df["sentimen"].value_counts()
    n_dibuang = len(raw) - len(df)

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
    teks_dibuang = (f"Sebanyak **{n_dibuang:,}** ulasan dibuang karena kosong setelah dibersihkan "
                    "(misalnya hanya berisi emoji atau simbol)."
                    if n_dibuang else "Tidak ada ulasan yang dibuang.")
    kat_vc = df["kategori"].value_counts()
    jelaskan(f"""
- File Anda berisi **{len(raw):,} ulasan**. {teks_dibuang} Tersisa **{len(df):,}** ulasan yang dipakai.
- Ulasan terbagi ke **{df["kategori"].nunique()} kategori topik**; yang paling banyak adalah **{kat_vc.index[0]}** ({kat_vc.iloc[0]:,} ulasan).
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
    cm_tr = confusion_matrix(y_train, m["pred_train"], labels=labels_aktif)
    cm_te = confusion_matrix(y_test, m["pred_test"], labels=labels_aktif)

    tr, te = hasil["Training"], hasil["Testing"]
    acc_te, acc_tr = te["Akurasi"], tr["Akurasi"]
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
- ℹ️ Semua angka di tab ini mengukur **model Naive Bayes murni**. Koreksi kata kunci di tab Prediksi tidak ikut dihitung.
""")

    jelaskan("""
- **Akurasi**: persentase tebakan yang benar dari seluruh data.
- **Precision**: saat model bilang "negatif", seberapa sering itu benar?
- **Recall**: dari semua ulasan yang benar-benar negatif, berapa yang berhasil ditemukan model?
- **F1-score**: gabungan precision dan recall; makin dekat 100% makin baik.
- **Macro average**: rata-rata biasa dari semua kelas, jadi kelas kecil dihitung sama pentingnya.
- **Confusion matrix**: tabel tebakan vs kenyataan. Angka di **diagonal** (kiri-atas ke kanan-bawah) adalah tebakan yang benar; angka di luar diagonal adalah kesalahan.
""", "📚 Kamus istilah", expanded=False)

    # ---------------- Metrik (kartu)
    st.subheader("Metrik (macro average)")
    mc = st.columns(4)
    for c, (nama, val) in zip(mc, te.items()):
        c.metric(f"{nama} (testing)", f"{val:.2%}",
                 delta=f"{val - tr[nama]:+.2%} vs training")

    prec, rec = te["Precision"], te["Recall"]
    if prec - rec > 0.05:
        catatan_pr = (f"Precision ({pct(prec)}) lebih tinggi dari recall ({pct(rec)}): model cenderung "
                      "**hati-hati**. Kalau ia memberi suatu label biasanya benar, tetapi masih ada "
                      "ulasan yang terlewat.")
    elif rec - prec > 0.05:
        catatan_pr = (f"Recall ({pct(rec)}) lebih tinggi dari precision ({pct(prec)}): model cenderung "
                      "**mudah memberi label**. Hampir semua ulasan terjaring, tetapi ada tebakan yang meleset.")
    else:
        catatan_pr = (f"Precision ({pct(prec)}) dan recall ({pct(rec)}) berdekatan: model "
                      "**seimbang** antara ketepatan dan kelengkapan.")
    rata_delta = float((te - tr).mean())
    arah = ("lebih rendah dari" if rata_delta < -0.005 else
            "lebih tinggi dari" if rata_delta > 0.005 else "hampir sama dengan")
    jelaskan(f"""
Empat kartu di atas adalah **nilai rapor model pada data ujian ({len(test_df):,} ulasan)**.

- **Akurasi {pct(te['Akurasi'])}**: sekitar {round(te['Akurasi'] * 100)} dari 100 ulasan ditebak benar → **{tingkat(te['Akurasi'])}**.
- **Precision {pct(prec)}**: rata-rata, saat model menyebut sebuah sentimen, tebakannya benar {round(prec * 100)} dari 100 kali.
- **Recall {pct(rec)}**: rata-rata, dari 100 ulasan yang sungguh punya suatu sentimen, model berhasil menemukan {round(rec * 100)}.
- **F1-score {pct(te['F1-score'])}**: nilai gabungan precision dan recall → **{tingkat(te['F1-score'])}**.
- {catatan_pr}
- Angka kecil berwarna di bawah tiap kartu = selisih nilai testing dibanding training. Rata-rata nilai testing **{arah}** training ({rata_delta:+.1%}). Tanda minus berarti performa pada soal baru sedikit turun dari saat belajar, dan itu normal selama tidak terlalu besar.
""", "📖 Cara membaca kartu metrik")

    # ---------------- Tabel metrik
    st.subheader("Tabel metrik: training vs testing")
    st.dataframe(hasil.round(4), use_container_width=True)
    selisih = (tr - te)
    m_gap = selisih.idxmax()
    jelaskan(f"""
- Kolom **Training** = nilai pada **{len(train_df):,} ulasan** yang dipakai belajar. Kolom **Testing** = nilai pada **{len(test_df):,} ulasan** ujian.
- Selisih terbesar ada pada metrik **{m_gap}**: training {pct(tr[m_gap])} vs testing {pct(te[m_gap])} (beda {abs(selisih[m_gap]) * 100:.1f} poin).
- Metrik testing tertinggi: **{te.idxmax()}** ({pct(te.max())}); terendah: **{te.idxmin()}** ({pct(te.min())}).
- {'Selisih yang kecil menandakan model **konsisten**: ia tidak hanya hafal data latih.' if gap <= 0.10 else 'Selisih yang besar menandakan model **terlalu hafal** data latih dan kurang luwes pada ulasan baru.'}
- Angka ditampilkan dalam desimal (0.85 = 85%).
""", "📖 Cara membaca tabel metrik")

    # ---------------- Grafik metrik
    st.subheader("Grafik metrik")
    fig, ax = plt.subplots(figsize=(8, 4))
    hasil.plot(kind="bar", ax=ax, color=["#3182bd", "#fd8d3c"], width=0.7)
    for c in ax.containers:
        ax.bar_label(c, labels=[f"{v:.1%}" for v in c.datavalues], fontsize=8, padding=2)
    ax.set_ylim(0, 1.15); plt.xticks(rotation=0); plt.tight_layout()
    st.pyplot(fig)
    rata_gap_bar = float(selisih.mean())
    if rata_gap_bar > 0.10:
        bentuk = "Batang **oranye jauh lebih pendek** dari biru, tanda model kurang bisa mengeneralisasi."
    elif rata_gap_bar > 0.03:
        bentuk = "Batang oranye **sedikit lebih pendek** dari biru, wajar untuk model yang diuji pada data baru."
    else:
        bentuk = "Batang biru dan oranye **hampir sama tinggi**, tanda performa stabil."
    jelaskan(f"""
- **Batang biru** = nilai training, **batang oranye** = nilai testing. Makin tinggi batang, makin baik; batas atas grafik adalah 100%.
- {bentuk}
- Pada data testing, metrik **{te.idxmax()}** paling tinggi ({pct(te.max())}) dan **{te.idxmin()}** paling rendah ({pct(te.min())}).
- Fokus utama ke batang **oranye**, karena itu gambaran kinerja model pada ulasan yang belum pernah dilihat.
""", "📖 Cara membaca grafik")

    # ---------------- Classification report
    st.subheader("Classification report (testing)")
    st.dataframe(pd.DataFrame(rep).T.round(4), use_container_width=True)
    baris_kelas = ""
    for k in labels_aktif:
        r = rep[k]
        baris_kelas += (f"- **{k}** ({int(r['support']):,} ulasan uji): precision {pct(r['precision'])}, "
                        f"recall {pct(r['recall'])}, F1 {pct(r['f1-score'])}.\n")
    f1_macro = rep["macro avg"]["f1-score"]
    f1_w = rep["weighted avg"]["f1-score"]
    cat_w = ("F1 *weighted* lebih tinggi dari *macro*, artinya kelas yang jumlahnya besar dikenali lebih baik "
             "daripada kelas yang kecil." if f1_w - f1_macro > 0.03 else
             "F1 *weighted* dan *macro* berdekatan, artinya kualitas relatif merata di semua kelas.")
    jelaskan(f"""
Tabel ini merinci nilai **per kelas** (bukan rata-rata):

{baris_kelas}
- **support** = jumlah ulasan uji di kelas itu. **macro avg** = rata-rata biasa antar kelas ({pct(f1_macro)}), **weighted avg** = rata-rata dengan memperhitungkan jumlah ulasan ({pct(f1_w)}). {cat_w}
""", "📖 Cara membaca classification report")

    # ---------------- Confusion matrix (vertikal)
    st.subheader("Confusion matrix")
    contoh_c = (f"Contoh membaca: baris **{labels_aktif[0]}** × kolom **{labels_aktif[-1]}** = "
                f"**{int(cm_te[0, -1]):,}**, artinya {int(cm_te[0, -1]):,} ulasan yang sebenarnya "
                f"**{labels_aktif[0]}** ditebak model sebagai **{labels_aktif[-1]}**.")
    jelaskan(f"""
Confusion matrix adalah tabel **"tebakan vs kenyataan"**. **Baris = sentimen sebenarnya**, **kolom = tebakan model**.
Kotak di diagonal (kiri-atas ke kanan-bawah) adalah tebakan yang benar; kotak lain adalah kesalahan. Warna makin gelap = angka makin besar.

{contoh_c}
""", "📖 Cara membaca confusion matrix")

    def gambar_cm(cm, judul, fmt):
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", cbar=False, ax=ax,
                    xticklabels=labels_aktif, yticklabels=labels_aktif)
        ax.set_title(judul, fontweight="bold")
        ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual")
        plt.tight_layout()
        return fig

    st.markdown("##### 1) Data training")
    st.pyplot(gambar_cm(cm_tr, "Training", "d"))
    jelaskan(narasi_cm_jumlah(cm_tr, labels_aktif, "training") +
             f"- Sebagai pembanding, akurasi training **{pct(acc_tr)}** vs testing **{pct(acc_te)}**.",
             "📖 Penjelasan confusion matrix training")

    st.markdown("##### 2) Data testing")
    st.pyplot(gambar_cm(cm_te, "Testing", "d"))
    jelaskan(narasi_cm_jumlah(cm_te, labels_aktif, "testing") +
             "- Inilah gambaran paling jujur, karena data testing belum pernah dilihat model saat belajar.",
             "📖 Penjelasan confusion matrix testing")

    st.markdown("##### 3) Data testing (persentase per baris)")
    cm_te_n = cm_te / np.maximum(cm_te.sum(axis=1, keepdims=True), 1)
    st.pyplot(gambar_cm(cm_te_n, "Testing (per baris)", ".1%"))
    jelaskan(narasi_cm_persen(cm_te, labels_aktif),
             "📖 Penjelasan confusion matrix persentase")

    # ---------------- Kata berpengaruh
    st.subheader("Kata paling berpengaruh per kelas (Naive Bayes)")
    fitur = np.array(tfidf.get_feature_names_out())
    top_fitur = {k: fitur[np.argsort(nb.feature_log_prob_[i])[::-1][:12]]
                 for i, k in enumerate(nb.classes_)}
    umum = set.intersection(*[set(v) for v in top_fitur.values()]) if top_fitur else set()
    ket_umum = (f"Kata yang muncul di **semua** kolom ({fmt_kata(sorted(umum)[:5])}) adalah kata umum "
                "yang kurang membedakan sentimen." if umum else
                "Tidak ada kata yang muncul di semua kolom, tanda tiap sentimen punya ciri kata yang jelas.")
    jelaskan(f"Tabel ini menunjukkan **12 kata/frasa yang paling kuat mengarahkan model** ke tiap sentimen. "
             f"{ket_umum} Kata yang khas di satu kolom adalah petunjuk utama sentimen itu.",
             "📖 Cara membaca tabel kata berpengaruh", expanded=False)
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

    st.checkbox("Aktifkan koreksi kata kunci (disarankan)", key="pakai_koreksi",
                help="Naive Bayes menghitung kata satu per satu dan tidak paham kata kontras seperti "
                     "'tapi'. Koreksi ini menambahkan pengecekan kata bernada positif/negatif, "
                     "negasi ('tidak bagus'), dan kata kontras, lalu menggabungkannya dengan tebakan model.")
    pakai_koreksi = st.session_state["pakai_koreksi"]

    st.subheader("Coba prediksi ulasan baru")
    teks = st.text_area("Tulis ulasan:", "game nya seru banget tapi sering lag dan harga robux mahal",
                        height=100)
    if st.button("Prediksi", type="primary"):
        bersih = preprocess(teks)
        if not bersih:
            st.warning("Teks kosong setelah preprocessing. Coba kalimat lain.")
        else:
            proba, p_nb, kp, kn, dikoreksi = prediksi_teks(bersih, tfidf, nb, pakai_koreksi)
            pred = nb.classes_[proba.argmax()]
            pred_nb = nb.classes_[p_nb.argmax()]
            warna = {"positif": "success", "negatif": "error", "netral": "warning"}[pred]
            getattr(st, warna)(f"Prediksi: **{pred.upper()}** ({proba.max():.1%})")
            if dikoreksi and pred_nb != pred:
                st.caption(f"Tanpa koreksi kata kunci, model murni menebak **{pred_nb}** "
                           f"({p_nb.max():.1%}).")
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
            teks_hasil = f"- Model menebak **{pred}** dan {tingkat_yakin} ({pct(yakin)}).\n"
            if kedua is not None:
                teks_hasil += f"- Alternatif terdekat: **{kedua}** ({pct(proba[urut[1]])}).\n"
            if kp or kn:
                teks_hasil += (f"- Kata bernada **negatif** yang terdeteksi: {fmt_kata(kn)}. "
                               f"Kata bernada **positif**: {fmt_kata(kp)}.\n")
                if any(w in KONTRAS for w in token):
                    teks_hasil += ("- Ada kata kontras (mis. *tapi*). Kata setelahnya diberi bobot lebih besar "
                                   "karena biasanya itulah inti penilaian.\n")
            if dikoreksi:
                if pred_nb != pred:
                    teks_hasil += (f"- Koreksi kata kunci **mengubah** hasil dari **{pred_nb}** menjadi **{pred}**, "
                                   "karena jumlah kata bernada negatif/positif tidak sejalan dengan tebakan "
                                   "model murni. Naive Bayes menghitung kata satu per satu sehingga "
                                   "kata umum seperti *game* atau *seru* bisa mengalahkan keluhan.\n")
                else:
                    teks_hasil += "- Koreksi kata kunci **sejalan** dengan tebakan model murni.\n"
            elif pakai_koreksi:
                teks_hasil += ("- Tidak ada kata kunci positif/negatif yang dikenali, jadi hasil murni "
                               "dari model Naive Bayes.\n")
            else:
                teks_hasil += "- Koreksi kata kunci **dimatikan**: hasil murni dari model Naive Bayes.\n"
            teks_hasil += (f"- Dari **{len(token)} kata** setelah dibersihkan, **{len(kenal)}** dikenali model"
                           + (f" dan **{len(asing)}** tidak dikenali ({fmt_kata(asing[:6])}), "
                              "jadi tidak ikut memengaruhi tebakan model." if asing
                              else ". Semua kata dikenali."))
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
            preds = []
            for t in bdf["text_clean"]:
                if t.strip():
                    p, _, _, _, _ = prediksi_teks(t, tfidf, nb, pakai_koreksi)
                    preds.append(nb.classes_[p.argmax()])
                else:
                    preds.append(None)
            bdf["prediksi"] = preds
            vb = bdf["prediksi"].value_counts()
            ringkas = ", ".join(f"**{k}** {v:,} ({pct(v / len(bdf))})" for k, v in vb.items())
            kosong = int((bdf["text_clean"].str.strip() == "").sum())
            jelaskan(f"File Anda berisi **{len(bdf):,} ulasan**. Hasil prediksi: {ringkas}."
                     + (f" Ada **{kosong:,}** ulasan yang kosong setelah dibersihkan sehingga tidak "
                        "diprediksi." if kosong else "")
                     + f" Koreksi kata kunci saat ini **{'aktif' if pakai_koreksi else 'mati'}**."
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
