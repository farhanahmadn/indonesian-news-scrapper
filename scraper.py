import asyncio
import os
import random
import re
import time
import urllib.parse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import aiohttp
import feedparser
import trafilatura
from dotenv import load_dotenv
from googlenewsdecoder import gnewsdecoder
from supabase import create_client

# Memuat variabel lingkungan dari file .env
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("⚠️ Waduh! SUPABASE_URL atau SUPABASE_KEY belum diatur di file .env")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

PORTAL_TIER_1 = [
    "kompas.com", "detik.com", "tempo.co", "cnnindonesia.com", 
    "cnbcindonesia.com", "antaranews.com", "liputan6.com", 
    "republika.co.id", "suara.com", "tirto.id"
]

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Mobile/15E148 Safari/604.1"
]

POLA_MUTLAK = [
    "berita dan informasi", "kabar akurat terpercaya", "timeline berita terbaru", "berita harian",
    "kumpulan berita", "kumpulan artikel", "top 3 berita", "indeks berita", "berita terpopuler", 
    "top news", "jadwal sholat", "prakiraan cuaca", "kurs valas"
]

POLA_AWALAN = [
    "berita terkini", "berita terbaru", "terkini dan terbaru", "berita hari ini", 
    "kabar terbaru", "headline hari ini", "fokus berita", "kabar harian"
]

KATA_HARAM = ["zodiak", "ramalan", "promo", "diskon", "lirik lagu"]
POLA_URL_SAMPAH = ["/tag/", "/tags/", "/indeks/", "/index/"]


def hapus_watermark_portal(judul_bawaan_google, nama_portal):
    suffix_google = f" - {nama_portal}"
    if judul_bawaan_google.endswith(suffix_google):
        judul_asli = judul_bawaan_google[:-len(suffix_google)]
    else:
        judul_asli = re.sub(r'\s*-\s*[^-]+$', '', judul_bawaan_google)
    return judul_asli.strip()


def apakah_berita_sampah(judul, url):
    if not judul or not judul.strip():
        return True
    if len(judul.strip().split()) < 3:
        return True
    if any(p in url for p in pola_url_sampah):
        return True
    if any(kata in judul for kata in kata_haram):
        return True
    if any(pola in judul for pola in POLA_MUTLAK):
        return True
    for pola in POLA_AWALAN:
        if judul.startswith(pola):
            return True
    return False


async def ekstrak_isi_artikel_async(session, url, sem):
    """Mengekstrak teks artikel secara asinkron dengan batasan konkurensi."""
    async with sem:
        await asyncio.sleep(random.uniform(0.5, 1.5))
        headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
            "Referer": "https://www.google.com/"
        }
        try:
            async with session.get(url, headers=headers, timeout=15) as response:
                if response.status == 200:
                    html = await response.text()
                    text = await asyncio.to_thread(trafilatura.extract, html)
                    return text if text else "Gagal mengekstrak teks."
                elif response.status == 403:
                    return "Terblokir (403 Forbidden)."
                return f"Gagal (Status: {response.status})"
        except asyncio.TimeoutError:
            return "Error: Timeout."
        except Exception as e:
            return f"Error: {e}"


async def agregator_seluruh_berita_tier1_async():
    """Mengumpulkan URL dari RSS lalu mengekstrak teks secara paralel."""
    print(f"\n{'='*40}")
    print(f"🔄 BATCH: {datetime.now(ZoneInfo('Asia/Jakarta')).strftime('%H:%M:%S WIB')} | ASYNC SCRAPER")
    print(f"{'='*40}")

    semua_berita_mentah = []

    for portal in PORTAL_TIER_1:
        query_lengkap = f'site:{portal} when:1h'
        query_encoded = urllib.parse.quote_plus(query_lengkap)
        url_rss = f"https://news.google.com/rss/search?q={query_encoded}&hl=id&gl=ID&ceid=ID:id"

        feed = feedparser.parse(url_rss)

        for entry in feed.entries:
            try:
                nama_portal = entry.source.title if 'source' in entry else "Tidak diketahui"
                judul = hapus_watermark_portal(entry.title, nama_portal)

                decode_result = gnewsdecoder(entry.link)
                url_asli = decode_result['decoded_url'] if decode_result.get('status') else entry.link

                if apakah_berita_sampah(judul.lower(), url_asli.lower()):
                    continue

                waktu_utc = datetime(*entry.published_parsed[:6])
                waktu_wib = waktu_utc + timedelta(hours=7)

                semua_berita_mentah.append({
                    "Judul": judul,
                    "Portal": nama_portal,
                    "Waktu_Rilis": waktu_wib.strftime("%Y-%m-%d %H:%M:%S"),
                    "URL": url_asli,
                    "Isi_Berita": ""
                })
            except:
                continue

    if not semua_berita_mentah:
        print("INFO: Tidak ada berita baru.")
        return []

    sem = asyncio.Semaphore(15) 
    async with aiohttp.ClientSession() as session:
        tasks = [ekstrak_isi_artikel_async(session, b["URLStream"], sem) for b in semua_berita_mentah] # Perbaikan penamaan b["URL"]
        tasks = [ekstrak_isi_artikel_async(session, b["URL"], sem) for b in semua_berita_mentah]
        hasil_ekstraksi = await asyncio.gather(*tasks)

    semua_berita_valid = []
    for i, teks in enumerate(hasil_ekstraksi):
        if len(teks) >= 50 and "Gagal" not in teks and "Error" not in teks:
            semua_berita_mentah[i]["Isi_Berita"] = teks
            semua_berita_valid.append(semua_berita_mentah[i])

    semua_berita_valid.sort(key=lambda x: x["Waktu_Rilis"], reverse=True)
    print(f"✅ SUKSES: Menemukan {len(semua_berita_valid)} berita yang valid.")
    return semua_berita_valid


async def simpan_berita_ke_db_async(array_berita_bersih):
    """Menyimpan berita ke Supabase secara batch & asinkron."""
    if not array_berita_bersih:
        return

    print(f"[INGESTION] Menyuntikkan {len(array_berita_bersih)} berita ke Supabase...")

    data_untuk_dimasukkan = [
        {
            "judul": b["Judul"],
            "isi_teks": b["Isi_Berita"],
            "portal_sumber": b["Portal"],
            "url_asli": b["URL"],
            "waktu_rilis": b["Waktu_Rilis"],
            "status_proses": 0
        }
        for b in array_berita_bersih
    ]

    def jalankan_upsert():
        return supabase.table("tabel_berita").upsert(
            data_untuk_dimasukkan,
            on_conflict="url_asli",
            ignore_duplicates=True 
        ).execute()

    try:
        hasil = await asyncio.to_thread(jalankan_upsert)
        print(f"[INGESTION] Sukses! {len(hasil.data)} berita baru masuk ke Supabase.")
    except Exception as e:
        print(f"⚠️ Gagal melakukan Batch Upsert: {e}")


async def run_worker_1_periodically():
    interval_detik = 1800 # 30 Menit

    while True:
        waktu_mulai = time.time()
        data_berita = await agregator_seluruh_berita_tier1_async()

        for i, b in enumerate(data_berita[:5], start=1):
            print(f"{i:02d}. {b['Judul']} [{b['Waktu_Rilis']}]")

        if data_berita:
            await simpan_berita_ke_db_async(data_berita)

        durasi_eksekusi = time.time() - waktu_mulai
        waktu_tidur = interval_detik - durasi_eksekusi

        if waktu_tidur > 0:
            print(f"\n[Siklus Selesai ({durasi_eksekusi:.2f} detik). Tidur selama {waktu_tidur:.2f} detik...]")
            await asyncio.sleep(waktu_tidur)
        else:
            print(f"\n[⚠️ Peringatan: Eksekusi melampaui batas waktu. Langsung lanjut siklus baru!]")


if __name__ == "__main__":
    # Menjalankan event loop asinkron secara native di Python standar
    try:
        asyncio.run(run_worker_1_periodically())
    except KeyboardInterrupt:
        print("\n👋 Scraper dihentikan oleh pengguna.")