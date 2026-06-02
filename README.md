## Projeto IA — Classificação de Acidentes em Rodovias Federais

Trabalho da disciplina de Inteligência Artificial. O objetivo é prever a gravidade de um acidente
na rodovia federal (sem vítimas, com feridos ou com mortos) a partir das características do ocorrido,
usando dados abertos da PRF de 2023.

## Dados

Baixados do portal de dados abertos da PRF:
https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf

O arquivo `acidentes2023.csv` tem 571 mil registros e não está no repositório por ser grande demais (206 MB).
Para baixar, rode:
python download_prf.py

## Instalação
git clone https://github.com/zweibrucker/PROJETO-IA.git
cd PROJETO-IA
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

## Como treinar o modelo
python app/ml/train.py

Isso vai pré-processar os dados, treinar a Regressão Logística (baseline) e depois o Random Forest.
As métricas ficam salvas em `models/metrics.json`.

## Como subir a API
uvicorn app.main:app --reload --port 9000

Para testar se está funcionando: http://localhost:9000/health

## Como fazer uma previsão
curl -X POST http://localhost:9000/predict 
-H "Content-Type: application/json" 
-d '{
"features": {
"uf": "MG",
"br": 40,
"causa_acidente": "Velocidade incompativel",
"tipo_acidente": "Colisao frontal",
"fase_dia": "Plena Noite",
"condicao_metereologica": "Chuva",
"tipo_pista": "Dupla",
"tracado_via": "Reta",
"uso_solo": "Rural"
}
}'

Resposta:

```json
{
  "prediction": "Com Vítimas Feridas",
  "proba": {
    "Sem Vítimas": 0.08,
    "Com Vítimas Feridas": 0.71,
    "Com Vítimas Fatais": 0.21
  }
}
```

## Notebooks

Com a API no ar, abra o Jupyter:
jupyter notebook

Rodar na ordem: `01_eda` → `02_baseline` → `04_cliente_analitico`

O notebook `04_cliente_analitico` precisa da API rodando para funcionar.

## Resultados

| Modelo | Accuracy | F1 Macro |
|---|---|---|
| Regressão Logística (baseline) | 41,4% | 0,354 |
| Random Forest | 92,1% | 0,872 |