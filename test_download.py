import urllib.request
import sys

file_id = "1-caam_dahYOf2eorq4mez04Om6DD5d_3"
url = f"https://drive.google.com/uc?export=download&id={file_id}"
headers = {"User-Agent": "Mozilla/5.0"}

print(f"Tentando download: {url}")
req = urllib.request.Request(url, headers=headers)
try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        content_type = resp.headers.get("Content-Type", "")
        print(f"Content-Type: {content_type}")
        content = resp.read(500)
        print(f"Primeiros bytes: {content[:300]}")
except Exception as e:
    print(f"Erro: {e}")
