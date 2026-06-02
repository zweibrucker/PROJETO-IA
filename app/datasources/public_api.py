"""
Fonte de dados: APIs e bases públicas brasileiras (Grupo 2).
=============================================================

Este adapter consome dados de fontes públicas como IBGE, PRF, INEP,
dados.gov.br, Portal da Transparência, etc. É o adapter usado pelas
equipes que NÃO desenvolvem aplicação web própria.

Padrão de uso:
    1. Definir a variável de ambiente que escolhe a fonte:
        export ML_PUBLIC_SOURCE="ibge_municipios"

    2. Em `app/config.py`:
        DATASOURCE_KIND = "api"

    3. Adapte (ou adicione) o método `_fetch_<fonte>()` correspondente.

Fontes pré-configuradas neste módulo
------------------------------------
- ibge_municipios     -> Lista de municípios BR (Bloco D, busca semântica)
- ibge_pib_municipal  -> PIB municipal (Bloco B, regressão/classif.)
- prf_acidentes       -> Acidentes em rodovias federais (Bloco B)
- inep_microdados     -> Microdados educacionais (Bloco A ou B)
- dados_gov_generico  -> Endpoint CKAN de dados.gov.br

Cache local
-----------
APIs públicas são lentas e podem cair. Toda chamada é cacheada em
`data/_cache/` como CSV. A segunda execução já lê do disco. Use
`force_refresh=True` para invalidar.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import DATA_DIR

CACHE_DIR: Path = DATA_DIR / "_cache"
CACHE_DIR.mkdir(exist_ok=True)


# =============================================================
# Helpers de HTTP + cache
# =============================================================
def _cached_get(url: str, cache_key: str, force_refresh: bool = False) -> Any:
    """GET HTTP com cache em disco.

    Cacheia o JSON de retorno em `data/_cache/<cache_key>.json`. Em
    caso de timeout ou erro de rede, levanta a exceção original — a
    equipe deve decidir o fallback (re-treinar com cache antigo, etc.).
    """
    cache_path = CACHE_DIR / f"{cache_key}.json"
    if cache_path.exists() and not force_refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    # Import tardio: só exige `requests` quando a API é realmente usada.
    import requests

    # Algumas APIs públicas (incluindo a do IBGE) bloqueiam clientes
    # sem User-Agent. Mande um cabeçalho identificando o projeto.
    headers = {"User-Agent": "ProjetoIntegradorML/0.2 (educacional)"}

    print(f"[API] GET {url}")
    resp = requests.get(url, timeout=30, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def _cached_csv(url: str, cache_key: str, force_refresh: bool = False, **read_csv_kwargs: Any) -> pd.DataFrame:
    """Mesmo conceito, mas para CSV (PRF, INEP)."""
    cache_path = CACHE_DIR / f"{cache_key}.csv"
    if cache_path.exists() and not force_refresh:
        return pd.read_csv(cache_path, **read_csv_kwargs)

    print(f"[API] Baixando CSV: {url}")
    df = pd.read_csv(url, **read_csv_kwargs)
    df.to_csv(cache_path, index=False)
    return df


# =============================================================
# Adapter principal
# =============================================================
class PublicApiSource:
    """Lê dados de bases públicas. A fonte específica é escolhida
    pela variável de ambiente `ML_PUBLIC_SOURCE`.

    Cada método `fetch_*` consulta a env var e roteia para o
    helper privado correspondente. Se a fonte for incompatível com
    o método (p. ex. pedir interactions de um dataset que só tem
    municípios), levanta NotImplementedError com mensagem clara.
    """

    def __init__(self) -> None:
        self.source = os.environ.get("ML_PUBLIC_SOURCE", "ibge_municipios")
        print(f"[API] fonte ativa: {self.source}")

    # ---------------- Bloco A ----------------
    def fetch_interactions(self) -> pd.DataFrame:
        if self.source == "inep_microdados":
            return self._fetch_inep_interactions()
        raise NotImplementedError(
            f"A fonte '{self.source}' não fornece dados de interação "
            f"(user_id, item_id, rating). Para Bloco A, use uma fonte "
            f"que tenha relação muitos-para-muitos."
        )

    # ---------------- Bloco B ----------------
    def fetch_dataset(self) -> pd.DataFrame:
        if self.source == "ibge_pib_municipal":
            return self._fetch_ibge_pib()
        if self.source == "prf_acidentes":
            return self._fetch_prf_acidentes()
        raise NotImplementedError(
            f"A fonte '{self.source}' não está configurada como dataset "
            f"tabular para Bloco B. Adicione um método _fetch_*() neste arquivo."
        )

    # ---------------- Bloco C ----------------
    def fetch_texts(self) -> pd.DataFrame:
        if self.source == "prf_acidentes":
            return self._fetch_prf_textos()
        raise NotImplementedError(
            f"A fonte '{self.source}' não fornece textos livres."
        )

    # ---------------- Bloco D ----------------
    def fetch_corpus(self) -> pd.DataFrame:
        if self.source == "ibge_municipios":
            return self._fetch_ibge_municipios()
        if self.source == "dados_gov_generico":
            return self._fetch_dados_gov_corpus()
        raise NotImplementedError(
            f"A fonte '{self.source}' não está configurada como corpus para Bloco D."
        )

    # =============================================================
    # IBGE — Municípios (Bloco D)
    # =============================================================
    @staticmethod
    def _fetch_ibge_municipios() -> pd.DataFrame:
        """Lista todos os municípios brasileiros (~5570).

        API documentada em:
            https://servicodados.ibge.gov.br/api/docs/localidades
        """
        url = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios"
        data = _cached_get(url, "ibge_municipios")

        rows = []
        for m in data:
            rows.append({
                "item_id": str(m["id"]),
                "titulo": m["nome"],
                "texto": (
                    f"Município de {m['nome']}, "
                    f"microrregião {m['microrregiao']['nome']}, "
                    f"mesorregião {m['microrregiao']['mesorregiao']['nome']}, "
                    f"estado {m['microrregiao']['mesorregiao']['UF']['nome']} "
                    f"({m['microrregiao']['mesorregiao']['UF']['sigla']}), "
                    f"região {m['microrregiao']['mesorregiao']['UF']['regiao']['nome']}."
                ),
            })
        return pd.DataFrame(rows)

    # =============================================================
    # IBGE — PIB municipal (Bloco B)
    # =============================================================
    @staticmethod
    def _fetch_ibge_pib() -> pd.DataFrame:
        """PIB per capita por município, ano mais recente disponível.

        NOTA: A equipe deve adaptar para a tabela específica (SIDRA).
        Este é apenas um stub instrucional.
        """
        # Stub: a equipe deve trocar pela URL real de SIDRA.
        # Exemplo (PIB 2021 por município):
        # https://apisidra.ibge.gov.br/values/t/5938/n6/all/v/all/p/2021
        raise NotImplementedError(
            "Implementar consulta à API SIDRA do IBGE. "
            "Documentação: https://apisidra.ibge.gov.br/"
        )

    # =============================================================
    # PRF — Acidentes em rodovias federais (Bloco B / C)
    # =============================================================
    @staticmethod
    def _fetch_prf_acidentes() -> pd.DataFrame:
        """Acidentes registrados pela PRF em rodovias federais — ano 2023.

        Fonte oficial:
            https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf
        Arquivo utilizado: acidentes2023_todas_causas_tipos.csv
        Data de acesso: 2026-06-02
        Licença: Dados Abertos do Governo Federal (dados.gov.br)

        O arquivo local é carregado de data/raw/acidentes2023.csv.
        Separador: ponto-e-vírgula (;). Encoding: latin-1.
        """
        csv_local = DATA_DIR / "raw" / "acidentes2023.csv"
        if csv_local.exists():
            print(f"[PRF] Carregando CSV local: {csv_local}")
            df = pd.read_csv(csv_local, sep=";", encoding="latin-1", low_memory=False)
        else:
            # Fallback: download direto do Google Drive (arquivo ZIP)
            url = os.environ.get(
                "ML_PRF_CSV_URL",
                "https://drive.google.com/uc?export=download&id=1-caam_dahYOf2eorq4mez04Om6DD5d_3"
            )
            print(f"[PRF] CSV local não encontrado. Baixando de: {url}")
            import urllib.request
            import io
            import zipfile

            req = urllib.request.Request(url, headers={"User-Agent": "ProjetoIntegradorML/0.2"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = resp.read()

            if data[:2] == b'PK':
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    csv_name = [f for f in zf.namelist() if f.endswith(".csv")][0]
                    with zf.open(csv_name) as f:
                        raw = f.read()
            else:
                raw = data

            csv_local.parent.mkdir(parents=True, exist_ok=True)
            csv_local.write_bytes(raw)
            df = pd.read_csv(io.BytesIO(raw), sep=";", encoding="latin-1", low_memory=False)
            print(f"[PRF] Salvo em {csv_local}")

        print(f"[PRF] Dataset carregado: {df.shape[0]:,} linhas × {df.shape[1]} colunas")
        return df

    @staticmethod
    def _fetch_prf_textos() -> pd.DataFrame:
        """Textos da coluna de causa do acidente — útil para Bloco C."""
        df = PublicApiSource._fetch_prf_acidentes()
        text_col = next((c for c in ["causa_acidente", "descricao", "observacao"] if c in df.columns), None)
        if text_col is None:
            raise RuntimeError("Nenhuma coluna textual reconhecida no CSV da PRF.")
        return pd.DataFrame({
            "id": df.index.astype(str),
            "texto": df[text_col].fillna("").astype(str),
        })

    # =============================================================
    # INEP — Microdados educacionais (Bloco A — exemplo simplificado)
    # =============================================================
    @staticmethod
    def _fetch_inep_interactions() -> pd.DataFrame:
        """Interações sintéticas a partir de microdados INEP.

        Os microdados reais do INEP exigem download manual (ZIPs grandes).
        A equipe deve substituir esta implementação após baixar e
        processar os arquivos.
        """
        raise NotImplementedError(
            "Microdados INEP exigem download manual em "
            "https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados. "
            "Após baixar, exporte um CSV simplificado para data/interactions.csv "
            "e mude DATASOURCE_KIND='csv'."
        )

    # =============================================================
    # dados.gov.br — Corpus genérico via CKAN
    # =============================================================
    @staticmethod
    def _fetch_dados_gov_corpus() -> pd.DataFrame:
        """Lista de datasets públicos do portal dados.gov.br.

        Permite indexar todos os conjuntos de dados disponíveis e
        fornecer busca semântica sobre eles ("quero datasets sobre saúde").
        """
        url = "https://dados.gov.br/api/3/action/package_list"
        data = _cached_get(url, "dados_gov_packages")
        # `data` é {"success": True, "result": ["nome1", "nome2", ...]}
        names = data.get("result", [])

        # Limita para não estourar a API. Em produção, paginar.
        names = names[:200]
        rows = []
        for n in names:
            detail = _cached_get(
                f"https://dados.gov.br/api/3/action/package_show?id={n}",
                f"dados_gov_pkg_{n}",
            )
            r = detail.get("result", {})
            rows.append({
                "item_id": r.get("id", n),
                "titulo": r.get("title", n),
                "texto": r.get("notes", "") or "",
            })
        return pd.DataFrame(rows)
