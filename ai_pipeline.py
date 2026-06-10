import asyncio
import json
import os
import re
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import nltk
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client

# Mengunduh dependensi NLTK secara diam-diam
nltk.download('punkt_tab', quiet=True)

# Memuat variabel lingkungan
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not all([SUPABASE_URL, SUPABASE_KEY, GROQ_API_KEY]):
    raise ValueError("⚠️ Kredensial tidak lengkap! Pastikan SUPABASE_URL, SUPABASE_KEY, dan GROQ_API_KEY ada di file .env")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
client = Groq(api_key=GROQ_API_KEY)

print("[INIT] Memuat model embedding...")
embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


# ==========================================
# FUNGSI PENDUKUNG (HYBRID LOGIC)
# ==========================================
def bersihkan_teks_struktural(teks):
    return re.sub(r'\s+', ' ', str(teks)).strip()

def ekstrak_entitas_cepat(teks):
    """Mengekstrak entitas dari teks (menghindari awal kalimat & stop words)."""
    entitas = re.findall(r'\b[A-Z][a-zA-Z0-9]*\b', str(teks))
    stop_words = {
        'di', 'ke', 'dari', 'pada', 'dalam', 'yaitu', 'untuk', 'yang', 'dan', 'ini', 'itu',
        'telah', 'sebuah', 'puluhan', 'ratusan', 'insiden', 'skuad', 'pasukan', 'tim',
        'berkat', 'setelah', 'menyusul', 'pecahan', 'sejumlah', 'kepala', 'beberapa',
        'banyak', 'surat', 'para', 'warga', 'calon', 'getaran', 'bencana', 'masyarakat',
        'korban', 'aksi', 'gelar', 'pawai', 'ri', 'the', 'akan', 'istana', 'presiden'
    }
    return set([e.lower() for e in entitas if e.lower() not in stop_words])

def jaccard_similarity(set1, set2):
    if not set1 and not set2: return 1.0
    irisan = set1.intersection(set2)
    gabungan = set1.union(set2)
    return len(irisan) / len(gabungan) if len(gabungan) > 0 else 0

def vektorisasi_berbobot(judul, isi):
    """Bobot: Judul 70%, Isi (Lead) 30%."""
    teks_judul = bersihkan_teks_struktural(judul)
    teks_isi = bersihkan_teks_struktural(" ".join(str(isi).split()[:150]))
    vec_judul = embed_model.encode([teks_judul])[0]
    vec_isi = embed_model.encode([teks_isi])[0] if teks_isi else vec_judul
    vektor_kombinasi = (vec_judul * 0.7) + (vec_isi * 0.3)
    return vektor_kombinasi / np.linalg.norm(vektor_kombinasi)

def bersihkan_markdown_json(teks_kotor):
    teks = re.sub(r'^```json\s*', '', teks_kotor, flags=re.MULTILINE)
    teks = re.sub(r'^
```\s*', '', teks, flags=re.MULTILINE)
    return teks.strip()

def map_intensitas_ke_persen(intensitas):
    intensitas = str(intensitas).strip().lower()
    if "tinggi" in intensitas: return 100
    elif "sedang" in intensitas: return 50
    else: return 20


# ==========================================
# WORKER 3: CLUSTERING
# ==========================================
def worker_3_clustering_supabase():
    print("\n[WORKER 3] Memulai Smart Clustering (Lock-Centroid Mode)...")
    res_berita = supabase.table("tabel_berita").select("id_berita, judul, isi_teks").eq("status_proses", 0).execute()
    berita_baru = res_berita.data

    if not berita_baru:
        print("[WORKER 3] Tidak ada antrean berita baru.")
        return

    waktu_batas = (datetime.now(ZoneInfo("Asia/Jakarta")) - timedelta(hours=24)).isoformat()
    res_klaster = supabase.table("tabel_cluster").select("id_cluster, summary_text, judul_summary").gte("waktu_terbentuk", waktu_batas).execute()
    klaster_aktif = res_klaster.data

    memori_klaster = {}
    for c in klaster_aktif:
        id_c = c["id_cluster"]
        teks_referensi = c.get("judul_summary") or c.get("summary_text") or ""
        
        if teks_referensi:
            vec_k = embed_model.encode([bersihkan_teks_struktural(teks_referensi)])[0]
            entitas_k = ekstrak_entitas_cepat(teks_referensi)
        else:
            res_first = supabase.table("tabel_berita").select("judul, isi_teks").eq("id_cluster", id_c).limit(1).execute()
            if res_first.data:
                a = res_first.data[0]
                vec_k = vektorisasi_berbobot(a['judul'], a['isi_teks'])
                entitas_k = ekstrak_entitas_cepat(a['judul'] + " " + a['isi_teks'])
            else: continue
            
        memori_klaster[id_c] = {"vektor": vec_k, "entitas": entitas_k}

    pembaruan_klaster, klaster_direvisi, klaster_baru_count = {}, set(), 0
    threshold_atas, threshold_bawah = 0.70, 0.45

    for b in berita_baru:
        id_berita = b["id_berita"]
        vektor_artikel = vektorisasi_berbobot(b["judul"], b["isi_teks"])
        entitas_artikel = ekstrak_entitas_cepat(b["isi_teks"])
        id_terbaik = None

        if memori_klaster:
            daftar_id = list(memori_klaster.keys())
            matriks_klaster = np.array([m["vektor"] for m in memori_klaster.values()])
            skor_semantik = cosine_similarity([vektor_artikel], matriks_klaster)[0]
            indeks_max = np.argmax(skor_semantik)
            skor_max = skor_semantik[indeks_max]
            kandidat_id = daftar_id[indeks_max]

            if skor_max >= threshold_atas:
                id_terbaik = kandidat_id
            elif skor_max >= threshold_bawah:
                if jaccard_similarity(entitas_artikel, memori_klaster[kandidat_id]["entitas"]) >= 0.10:
                    id_terbaik = kandidat_id

        if id_terbaik is not None:
            target_id = id_terbaik
            klaster_direvisi.add(id_terbaik)
            memori_klaster[id_terbaik]["entitas"].update(entitas_artikel)
        else:
            res_new = supabase.table("tabel_cluster").insert({"status_summary": 0, "status_prediksi": 0}).execute()
            target_id = res_new.data[0]["id_cluster"]
            memori_klaster[target_id] = {"vektor": vektor_artikel, "entitas": entitas_artikel}
            klaster_baru_count += 1

        pembaruan_klaster.setdefault(target_id, []).append(id_berita)

    for id_c, list_id_b in pembaruan_klaster.items():
        supabase.table("tabel_berita").update({"id_cluster": id_c, "status_proses": 1}).in_("id_berita", list_id_b).execute()

    if klaster_direvisi:
        daftar_id_revisi = list(klaster_direvisi)
        supabase.table("tabel_sentimen_aktor").delete().in_("id_cluster", daftar_id_revisi).execute()
        supabase.table("tabel_sektor").delete().in_("id_cluster", daftar_id_revisi).execute()
        supabase.table("tabel_cluster").update({"status_summary": 0, "status_prediksi": 0}).in_("id_cluster", daftar_id_revisi).execute()

    for id_c in pembaruan_klaster.keys():
        res_count = supabase.table("tabel_berita").select("id_berita", count="exact").eq("id_cluster", id_c).execute()
        if res_count.count is not None:
            supabase.table("tabel_cluster").update({"jumlah_berita": res_count.count}).eq("id_cluster", id_c).execute()

    print(f"[WORKER 3] Selesai. Klaster Baru: {klaster_baru_count}, Klaster Direvisi: {len(klaster_direvisi)}")


# ==========================================
# WORKER 4: SUMMARIZE & SENTIMENT
# ==========================================
def worker_4_summarize_and_sentiment():
    print("\n[WORKER 4] Memulai Summarization & Sentiment Analysis...")
    waktu_batas = (datetime.now(ZoneInfo("Asia/Jakarta")) - timedelta(hours=24)).isoformat()
    antrean_klaster = supabase.table("tabel_cluster").select("id_cluster").eq("status_summary", 0).gte("jumlah_berita", 5).gte("waktu_terbentuk", waktu_batas).execute().data

    if not antrean_klaster:
        print("[WORKER 4] Tidak ada klaster matang (>= 5 berita) dalam 24 jam terakhir untuk dirangkum.")
        return

    print(f"[WORKER 4] Ditemukan {len(antrean_klaster)} klaster untuk diproses.\n")

    for baris in antrean_klaster:
        id_cluster = baris["id_cluster"]
        res_awal = supabase.table("tabel_berita").select("judul, isi_teks").eq("id_cluster", id_cluster).order("id_berita", desc=False).limit(15).execute().data
        res_akhir = supabase.table("tabel_berita").select("judul, isi_teks").eq("id_cluster", id_cluster).order("id_berita", desc=True).limit(15).execute().data
        
        berita_unik = {b["judul"]: b for b in (res_awal + res_akhir)}.values()
        batas_kata = 150 if len(berita_unik) > 15 else (200 if len(berita_unik) > 5 else 300)

        teks_potongan = [f"Judul: {b['judul']} | Isi: {' '.join(str(b['isi_teks']).split()[:batas_kata])}." for b in berita_unik if b['isi_teks']]
        teks_final = " ".join(" ".join(teks_potongan).split()[:5000])

        prompt = f"""
        Berikut adalah kumpulan potongan berita awal (Lead) terkait suatu peristiwa:
        "{teks_final}"

        Tugas Anda:
        1. JUDUL: Buat 1 kalimat "Judul Utama" bergaya jurnalistik (maksimal 10 kata).
        2. RANGKUMAN: Buat rangkuman bergaya jurnalistik (3-5 kalimat).
        3. AKTOR UTAMA: Identifikasi MAKSIMAL 3 aktor utama.
        4. SENTIMEN: Tentukan sentimen (Positif/Negatif/Netral) dan intensitasnya (Tinggi/Sedang/Rendah).

        OUTPUT WAJIB JSON MURNI DENGAN FORMAT:
        {{
            "judul": "...",
            "rangkuman": "...",
            "aktor": [{{"nama": "...", "sentimen": "...", "intensitas": "..."}}]
        }}
        """

        try:
            waktu_mulai = time.time()
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            hasil_json = json.loads(bersihkan_markdown_json(response.choices[0].message.content))
            
            supabase.table("tabel_cluster").update({
                "judul_summary": hasil_json.get("judul", "Tanpa Judul"),
                "summary_text": hasil_json.get("rangkuman", "Gagal menghasilkan rangkuman."),
                "status_summary": 1
            }).eq("id_cluster", id_cluster).execute()

            supabase.table("tabel_sentimen_aktor").delete().eq("id_cluster", id_cluster).execute()

            if hasil_json.get("aktor"):
                data_aktor = [{"id_cluster": id_cluster, "nama_aktor": str(a.get("nama"))[:100], "sentimen": str(a.get("sentimen")), "persentase": map_intensitas_ke_persen(a.get("intensitas"))} for a in hasil_json["aktor"]]
                supabase.table("tabel_sentimen_aktor").insert(data_aktor).execute()

            print(f"  [V] Klaster {id_cluster} Sukses | Waktu: {round(time.time() - waktu_mulai, 2)}s")
            time.sleep(6)

        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                print("     [!] Rate limit Groq. Menunggu 60 detik...")
                time.sleep(60)
            else:
                supabase.table("tabel_cluster").update({"status_summary": 2}).eq("id_cluster", id_cluster).execute()
                time.sleep(3)


# ==========================================
# WORKER 5: PREDIKSI DAMPAK
# ==========================================
def worker_5_prediksi():
    print("\n[WORKER 5] Memulai Prediksi Dampak...")
    antrean_klaster = supabase.table("tabel_cluster").select("id_cluster, judul_summary, summary_text").eq("status_prediksi", 0).execute().data

    if not antrean_klaster:
        print("[WORKER 5] Tidak ada cluster yang perlu diprediksi.")
        return

    for baris in antrean_klaster:
        id_cluster = baris["id_cluster"]
        if not baris.get("summary_text"):
            supabase.table("tabel_cluster").update({"status_prediksi": 2}).eq("id_cluster", id_cluster).execute()
            continue

        res_aktor = supabase.table("tabel_sentimen_aktor").select("nama_aktor, sentimen, persentase").eq("id_cluster", id_cluster).execute()
        konteks_aktor = "\n".join([f"- {a['nama_aktor']}: {a['sentimen']} ({a['persentase']}%)" for a in res_aktor.data]) if res_aktor.data else "Tidak ada data aktor."

        prompt_prediksi = f"""
        Kamu adalah analis kebijakan dan ekonomi senior Indonesia.
        JUDUL: {baris.get("judul_summary")}
        RINGKASAN: {baris.get("summary_text")}
        AKTOR: {konteks_aktor}

        Tugas: Pilih 1-3 sektor terdampak (Ekonomi & Bisnis, Politik & Pemerintahan, Hukum & Keamanan, Sosial & Masyarakat, Kesehatan, Pendidikan, Energi & Lingkungan, Teknologi, Olahraga & Hiburan, Hubungan Internasional) beserta prediksi dampak dan risiko.

        OUTPUT WAJIB JSON (ARRAY OF OBJECTS):
        {{
            "analisis_sektor": [{{"nama_sektor": "...", "prediksi_dampak": "...", "tingkat_risiko": "Tinggi/Sedang/Rendah"}}]
        }}
        """

        try:
            response_prediksi = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt_prediksi}],
                model="llama-3.1-8b-instant",
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            daftar_sektor = json.loads(bersihkan_markdown_json(response_prediksi.choices[0].message.content)).get("analisis_sektor", [])
            
            if daftar_sektor:
                data_insert = [{"id_cluster": id_cluster, "nama_sektor": s.get("nama_sektor", "Lainnya"), "prediksi_dampak": s.get("prediksi_dampak", "-"), "tingkat_risiko": s.get("tingkat_risiko", "Sedang").strip().capitalize()} for s in daftar_sektor]
                supabase.table("tabel_sektor").insert(data_insert).execute()

            supabase.table("tabel_cluster").update({"status_prediksi": 1}).eq("id_cluster", id_cluster).execute()
            print(f"  [V] Prediksi Cluster {id_cluster} tersimpan.")
            time.sleep(6)

        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                print("     [!] Rate limit Groq. Menunggu 60 detik...")
                time.sleep(60)
            else:
                supabase.table("tabel_cluster").update({"status_prediksi": 2}).eq("id_cluster", id_cluster).execute()
                time.sleep(2)


# ==========================================
# MAIN LOOP
# ==========================================
async def run_ai_pipeline_periodically():
    interval_detik = 2700 # 45 Menit
    while True:
        waktu_mulai = time.time()
        print(f"\n{'='*60}\n🚀 [SIKLUS AI DIMULAI: {time.strftime('%H:%M:%S')}]\n{'='*60}")
        try:
            await asyncio.to_thread(worker_3_clustering_supabase)
            await asyncio.to_thread(worker_4_summarize_and_sentiment)
            await asyncio.to_thread(worker_5_prediksi)
        except Exception as e:
            print(f"\n❌ [ERROR] Terjadi kegagalan pada siklus ini: {e}")

        durasi = time.time() - waktu_mulai
        waktu_tidur = interval_detik - durasi
        print(f"\n{'-'*60}")
        
        if waktu_tidur > 0:
            print(f"✅ [SIKLUS SELESAI] Durasi: {durasi:.2f} detik.\n💤 Tidur selama {waktu_tidur/60:.2f} menit.")
            await asyncio.sleep(waktu_tidur)
        else:
            print(f"⚠️ [OVERLOAD] Durasi melampaui interval!\nLangsung memulai siklus baru...")
            await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        asyncio.run(run_ai_pipeline_periodically())
    except KeyboardInterrupt:
        print("\n👋 Pipeline AI dihentikan oleh pengguna.")