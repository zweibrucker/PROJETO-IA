"""
Script de download dos dados PRF 2023.

Fonte: Portal de Dados Abertos da PRF
URL da página: https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf
Arquivo: acidentes2023_todas_causas_tipos.csv (compactado em ZIP)
Data de acesso: 2026-06-02
Licença: Dados Abertos do Governo Federal (dados.gov.br)
"""

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

# URL de download direto do Google Drive (arquivo ZIP com o CSV da PRF 2023)
FILE_ID = "1-caam_dahYOf2eorq4mez04Om6DD5d_3"
DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
OUTPUT_PATH = Path(__file__).parent / "data" / "raw" / "acidentes2023.csv"


def download_prf_2023():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        print(f"[OK] Arquivo já existe: {OUTPUT_PATH}")
        return

    print(f"[PRF] Baixando dados de 2023...")
    print(f"[PRF] URL: {DOWNLOAD_URL}")

    headers = {
        "User-Agent": "Mozilla/5.0 (projeto-ml educacional; acesso 2026-06-02)"
    }
    req = urllib.request.Request(DOWNLOAD_URL, headers=headers)

    with urllib.request.urlopen(req, timeout=300) as resp:
        total = resp.headers.get("Content-Length")
        if total:
            total = int(total)
            print(f"[PRF] Tamanho: {total / 1_048_576:.1f} MB")
        else:
            print("[PRF] Tamanho desconhecido, baixando...")

        data = bytearray()
        chunk_size = 1024 * 256  # 256 KB
        downloaded = 0
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            data.extend(chunk)
            downloaded += len(chunk)
            mb = downloaded / 1_048_576
            if total:
                pct = downloaded / total * 100
                print(f"\r[PRF] {mb:.1f} MB / {total/1_048_576:.1f} MB ({pct:.1f}%)", end="", flush=True)
            else:
                print(f"\r[PRF] {mb:.1f} MB baixados...", end="", flush=True)
        print()

    print(f"[PRF] Download concluído: {len(data) / 1_048_576:.1f} MB")

    # Verifica se é ZIP (magic bytes PK)
    if data[:2] == b'PK':
        print("[PRF] Arquivo ZIP detectado. Extraindo CSV...")
        with zipfile.ZipFile(io.BytesIO(bytes(data))) as zf:
            csv_files = [f for f in zf.namelist() if f.endswith(".csv")]
            print(f"[PRF] Arquivos no ZIP: {zf.namelist()}")
            if not csv_files:
                raise RuntimeError("Nenhum CSV encontrado no ZIP.")
            csv_name = csv_files[0]
            print(f"[PRF] Extraindo: {csv_name}")
            with zf.open(csv_name) as src, open(OUTPUT_PATH, "wb") as dst:
                dst.write(src.read())
    else:
        # Já é CSV direto
        OUTPUT_PATH.write_bytes(bytes(data))

    size_mb = OUTPUT_PATH.stat().st_size / 1_048_576
    print(f"[PRF] Salvo em: {OUTPUT_PATH} ({size_mb:.1f} MB)")
    print("[PRF] Download finalizado com sucesso.")


if __name__ == "__main__":
    download_prf_2023()
