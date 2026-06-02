# Estudo de Viabilidade — Sistema de Classificação de Acidentes PRF

**Disciplina:** Inteligência Artificial  
**Bloco:** B — Classificação/Predição  
**Sistema-alvo:** PRF — Acidentes em Rodovias Federais  
**Data:** 2026-06-02

---

## 1. Contexto e Público-Alvo

### 1.1 Quem Usaria o Sistema

O componente de classificação automática de gravidade de acidentes seria utilizado por três perfis dentro da PRF:

**Agentes de Campo (operacionais):**  
Ao registrar um boletim de acidente via tablet/celular, o agente preencheria os campos disponíveis (UF, rodovia, causa, condições) e receberia automaticamente uma sugestão de classificação. Isso reduz o tempo de preenchimento manual e padroniza o processo entre regiões.

**Supervisores de Delegacia:**  
Uso de dashboards alimentados pela API para monitorar a distribuição de gravidade em tempo real, identificando trechos com alta incidência de acidentes fatais e priorizando patrulhamento preventivo.

**Gestores Estratégicos (DENATRAN / Ministério dos Transportes):**  
Relatórios automáticos gerados a partir das classificações para embasar políticas públicas de segurança viária, definição de trechos críticos e alocação de orçamento para obras de engenharia rodoviária.

### 1.2 Cenários de Uso

| Cenário | Ator | Benefício |
|---------|------|-----------|
| Triagem inicial de acidente | Agente PRF | Classificação em <1s vs. 10-20min manual |
| Despacho de recursos | Centro de Controle | Ambulância enviada conforme gravidade prevista |
| Relatório mensal | Gestor | Gerado automaticamente, sem consolidação manual |
| Análise preventiva | Engenharia de tráfego | Identificação de trechos de risco |

---

## 2. Ganhos Potenciais

### 2.1 Redução de Tempo de Triagem

Atualmente, a classificação de gravidade de um acidente depende do julgamento do agente no campo, que pode levar entre 10 e 30 minutos para registrar e classificar corretamente. O modelo proposto realiza a classificação em menos de **100 milissegundos** por ocorrência após o registro das features básicas.

**Estimativa de ganho:**
- Volume médio: ~70.000 acidentes/ano no Brasil (PRF 2023)
- Tempo economizado por acidente: ~15 minutos de triagem administrativa
- Total: ~17.500 horas/ano de trabalho liberado para atividades operacionais

### 2.2 Padronização e Redução de Erro Humano

A classificação manual apresenta variabilidade entre agentes, turnos e regiões. Um estudo interno da PRF (hipotético, para fins acadêmicos) estimaria ~12% de discordância entre agentes distintos na classificação de severidade. O modelo elimina essa variabilidade ao aplicar critérios consistentes baseados nos dados históricos de 2023.

### 2.3 Alocação Mais Eficiente de Recursos de Emergência

Com a classificação automática disponível antes da chegada das equipes, é possível:
- Pré-posicionar ambulâncias em trechos de alta probabilidade de acidentes fatais
- Acionar médicos especializados somente quando a probabilidade de "Com Vítimas Fatais" superar um limiar configurável (ex: 60%)
- Reduzir tempo médio de atendimento (indicador crítico para sobrevivência em acidentes graves)

### 2.4 Geração de Inteligência Preventiva

A API permite integração com sistemas de videomonitoramento e radares de velocidade. Com a predição em tempo real, é possível correlacionar condições ambientais (chuva, neblina) com aumento de risco e emitir alertas preventivos via painéis eletrônicos na rodovia.

---

## 3. Riscos Éticos

### 3.1 Viés Regional nos Dados

O dataset da PRF 2023 apresenta forte concentração em estados do Sul e Sudeste (SP, MG, PR, RS concentram >50% dos registros), enquanto estados do Norte e Centro-Oeste são sub-representados. Isso pode gerar:

- **Subdesempenho regional:** o modelo pode ter menor acurácia em estados com poucos dados históricos (AM, AP, RR), onde condições de estrada e padrões de acidente diferem significativamente.
- **Risco de alocação desigual:** se o sistema subestimar a gravidade em regiões sub-representadas, pode haver menor despacho de recursos emergenciais nessas áreas.

**Mitigação recomendada:** avaliar métricas (F1 por classe) estratificadas por UF antes de implantação. Considerar retreino com datasets regionalizados para estados com menor representação.

### 3.2 Viés de Causa de Acidente

A variável `causa_acidente` é preenchida pelo agente de campo e pode conter vieses de percepção. Por exemplo, "ingestão de álcool" pode ser registrada com maior frequência em alguns estados ou horários, não necessariamente porque é mais prevalente, mas porque é mais verificada. O modelo aprende esses vieses do passado.

### 3.3 Conformidade com a LGPD (Lei nº 13.709/2018)

O dataset utilizado para treinamento **não contém dados pessoais identificáveis**:
- Não há CPF, nome, RG ou documento de identidade
- Não há endereço residencial das vítimas
- As colunas `idade` e `sexo` são atributos estatísticos agregados, não vinculados a identidades individuais

O sistema classifica o **acidente**, não o **indivíduo**. Portanto, está em conformidade com o Artigo 7º da LGPD (tratamento de dados para execução de políticas públicas).

**Recomendação:** documentar no contrato com a PRF que nenhum dado pessoal é armazenado pelo microsserviço de ML.

### 3.4 Responsabilidade de Decisão

O sistema é uma ferramenta de apoio, não de decisão autônoma. Nenhuma decisão de despacho de recursos de emergência deve ser tomada exclusivamente com base na predição do modelo. Um agente humano deve sempre confirmar presencialmente.

---

## 4. Manutenção e Ciclo de Vida

### 4.1 Retreino Anual

Os dados da PRF são publicados anualmente. Recomenda-se:

| Evento | Ação |
|--------|------|
| Janeiro | Download do CSV do ano anterior |
| Fevereiro | Reexecutar `app/ml/train.py` com dados novos |
| Março | Comparar métricas do novo modelo com o atual |
| Março | Deploy do novo modelo (se métricas superiores) |

### 4.2 Monitoramento de Data Drift

Com o tempo, a distribuição dos dados de produção pode divergir dos dados de treino (concept drift). Indicadores a monitorar:

- **Distribuição das classes previstas:** se "Com Vítimas Fatais" começar a cair consistentemente, pode indicar drift ou mudança real no padrão de acidentes.
- **Confiança média das predições:** queda na probabilidade média da classe vencedora indica que o modelo está incerto.

Ferramentas recomendadas: `evidently`, `alibi-detect`, ou monitoramento customizado com KL-divergence.

### 4.3 Responsável Técnico

O modelo deve ter um responsável técnico nomeado na PRF (ou no órgão gestor) com:
- Acesso às métricas de produção
- Capacidade de acionar o processo de retreino
- Autoridade para desativar o sistema em caso de degradação grave

---

## 5. Limitações Técnicas

### 5.1 Dados de Treinamento Limitados a 2023

O modelo foi treinado exclusivamente com dados de 2023. Eventos que alterem padrões de acidente (obras em rodovias federais, mudanças climáticas, novos veículos) não são capturados até o próximo retreino.

### 5.2 Sazonalidade Não Modelada

A variável `data_inversa` não foi incluída no modelo para evitar overfitting temporal. Porém, acidentes variam sazonalmente (Carnaval, férias, períodos chuvosos). Um modelo mais sofisticado poderia incluir features derivadas do calendário (mês, feriado, fim de semana).

### 5.3 Subnotificação Regional

Dados da PRF dependem da capacidade de registro das delegacias regionais. Em regiões com menor infraestrutura, acidentes podem não ser registrados ou classificados incorretamente, contaminando o dataset de treino.

### 5.4 Dependência do Pré-processamento

O modelo depende do `OrdinalEncoder` treinado na fase de pré-processamento. Qualquer mudança nos valores categóricos (nova UF criada, novo tipo de acidente catalogado) exige retreino completo do encoder e do modelo.

### 5.5 Escopo Geográfico

O modelo é válido apenas para rodovias federais (PRF). Rodovias estaduais e municipais têm padrões distintos de acidente e requereriam modelos específicos.

---

## 6. Recomendação Final

O sistema é **viável para implantação piloto** em uma região de tamanho médio (ex: 3 estados do Sul), com:
- Período de observação de 6 meses antes de expansão nacional
- Avaliação de equidade regional (métricas por UF)
- Protocolo claro de "não substituição de decisão humana"
- Plano de manutenção documentado e responsável técnico nomeado

O potencial de ganho é significativo (redução de triagem, padronização, prevenção), mas a implantação responsável requer cuidado com viés regional e LGPD.
