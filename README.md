# Avaliação Automática de Redações do Enem com um Transformer Multicompetência

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.46-FFD21E)
![MLflow](https://img.shields.io/badge/MLflow-tracking-0194E2?logo=mlflow&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

Sistema de **Avaliação Automática de Redações (AES, *Automatic Essay Scoring*)** para o **Exame Nacional do Ensino Médio (Enem)**. Um único codificador, como o **BERTimbau Base**, é compartilhado por **cinco cabeças de predição**, uma para cada competência do exame, e estima as cinco notas (0 a 200, em passos de 40) em **uma só passagem direta**, seguindo o paradigma de Aprendizado Multi-Tarefa (MTL).

O repositório reúne o código-fonte, os arquivos de configuração e os registros completos dos experimentos do Trabalho de Conclusão de Curso descrito em [Contexto acadêmico](#contexto-acadêmico).

---

## Sumário

- [Contexto acadêmico](#contexto-acadêmico)
- [Motivação e objetivos](#motivação-e-objetivos)
- [Visão geral da solução](#visão-geral-da-solução)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Instalação](#instalação)
- [Como executar](#como-executar)
- [Possibilidades de execução](#possibilidades-de-execução)
- [Referência do `aesconfig.json`](#referência-do-aesconfigjson)
- [Saídas geradas](#saídas-geradas)
- [Resultados](#resultados)
- [Resultados incluídos no repositório](#resultados-incluídos-no-repositório)
- [Limitações e trabalhos futuros](#limitações-e-trabalhos-futuros)
- [Como citar](#como-citar)
- [Referências principais](#referências-principais)
- [Licença](#licença)

---

## Contexto acadêmico

Este projeto é o produto prático do Trabalho de Conclusão de Curso (TCC) intitulado **"Avaliação Automática de Redações: Desenvolvimento e Análise Computacional de um Modelo Transformer Multicompetência para o Enem"**, do curso de **Bacharelado em Ciência da Computação** e apresentado ao **Departamento de Informática da Universidade Estadual de Maringá (UEM)**.

| | |
|---|---|
| **Autor** | Vitor da Rocha Machado |
| **Orientadora** | Valéria Delisandra Feltrim |
| **Banca** | Lucas de Oliveira Teixeira · Yandre Maldonado e Gomes da Costa |
| **Instituição** | Universidade Estadual de Maringá, Centro de Tecnologia, Departamento de Informática |
| **Curso** | Bacharelado em Ciência da Computação |
| **Ano** | 2026 (Maringá, PR) |
| **Repositório** | <https://github.com/hito-boo/enem-essay-scoring> |

A monografia detalha a fundamentação teórica, a revisão da literatura, a metodologia e a análise dos resultados. Este README resume o que é necessário para entender, executar e estender o código.

---

## Motivação e objetivos

A correção manual de redações em larga escala é cara, lenta e sujeita à subjetividade e à fadiga humana. A AES surge como ferramenta de **apoio** ao avaliador, sem substituí-lo, e como meio de oferecer retorno rápido a estudantes. Os Grandes Modelos de Linguagem (LLMs) têm bom desempenho, mas seu custo computacional inviabiliza o uso em infraestruturas restritas. Já treinar **cinco modelos independentes**, um por competência, multiplica memória e tempo de inferência.

**Objetivo geral:** modelar, implementar e avaliar o desempenho e o custo computacional de um modelo Transformer multicompetência para a AES do Enem.

**Objetivos específicos**

1. Levantar o estado da arte de Transformers e modelos pré-treinados para o português brasileiro em AES.
2. Desenvolver e treinar o modelo de múltiplas saídas, comparando formulações de **regressão contínua**, **classificação multiclasse** e **regressão ordinal**, com ajuste de hiperparâmetros.
3. Investigar e aplicar técnicas de eficiência computacional (**LoRA** e **Quantização Pós-Treinamento**).
4. Quantificar o consumo de recursos (tempo, CPU, RAM, VRAM, disco) do modelo padrão e das versões otimizadas.
5. Avaliar o desempenho com métricas padrão da área e compará-lo com a literatura.

---

## Visão geral da solução

### Dados utilizados na pesquisa

- **Corpus:** subconjunto **JBCS2025** do **AES-Enem** (Silveira, Barbosa e Mauá, 2024), carregado do Hugging Face (`kamel-usp/aes_enem_dataset`, configuração `JBCS2025`).
- **Conteúdo:** 385 redações do portal Educação UOL (38 temas), cada uma avaliada por **dois especialistas independentes**, totalizando **770 instâncias anotadas**.
- **Partições:** 500 para treino, 132 para validação e 138 para teste, divididas **por tema (*cross-prompt*)**. Nenhum tema aparece em duas partições, o que evita vazamento de dados.
- **Pré-processamento:** tokenização com o tokenizador do BERTimbau, truncamento e *padding* em **512 tokens**. As notas `{0, 40, …, 200}` são divididas por 40, resultando em níveis `{0, …, 5}`. Na avaliação, as predições voltam à escala 0–200.
- **Notebooks de exploração** (`corpus/`) documentam por que o JBCS2025 foi escolhido em vez do corpus de redações oficiais (157 amostras, com forte viés para notas altas) e do Essay-BR (ruído de formatação e vazamento de comentários de avaliadores).

### Arquitetura

```text
                          Redação (texto bruto)
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                         Tokenizador BERTimbau                          │
│                  input_ids + attention_mask (B × 512)                  │
└────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 Encoder BERTimbau Base (compartilhado)                 │
│                       12 camadas · d_model = 768                       │
└────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  Representação global [CLS] / pooler                   │
│                          (B × 768) + Dropout                           │
└────────────────────────────────────────────────────────────────────────┘
                                     │
       ┌──────────────┬──────────────┼──────────────┬──────────────┐
       ▼              ▼              ▼              ▼              ▼
┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐
│ Cabeça C1  │ │ Cabeça C2  │ │ Cabeça C3  │ │ Cabeça C4  │ │ Cabeça C5  │
│   Norma    │ │   Tema e   │ │Argumentação│ │   Coesão   │ │Proposta de │
│   padrão   │ │ repertório │ │            │ │            │ │intervenção │
└────────────┘ └────────────┘ └────────────┘ └────────────┘ └────────────┘
       │              │              │              │              │
       └──────────────┴──────────────┬──────────────┴──────────────┘
                                     │
                                     ▼
                   Perda global = Σ wc · Lc   (c = 1…5)
```

Cada cabeça pode ser **linear** (`Linear(768 → d_out)`) ou **multicamadas** (`768 → 256 → ReLU → d_out`). A dimensão de saída `d_out` depende da formulação da tarefa.

### Formulações da tarefa

| `task_type` | `d_out` | Perda | Decodificação |
|---|:---:|---|---|
| `regression` | 1 | MSE (ponderado por classe) | `round(clip(z, 0, 5))` |
| `classification` | 6 | Entropia cruzada (ponderada por classe) | `argmax` |
| `ordinal` + `extended_binary` (**padrão**) | 5 | BCE nos 5 limiares `y ≥ k` (ponderada por classe) | nº de limiares com `P(y ≥ k) > 0,5` |
| `ordinal` + `corn` | 5 | CORN loss (`coral-pytorch`) | `corn_label_from_logits` |

### Outros componentes

- **Perda sensível a classes:** frequência inversa suavizada por um expoente `power` (γ), para compensar a escassez de notas extremas. Desativada automaticamente no CORN.
- **Perda multicompetência ponderada:** pesos `loss_weights` por competência, normalizados internamente.
- **Treinamento:** AdamW com *weight decay* seletivo (sem decaimento em *bias* e `LayerNorm`), acumulação de gradientes, *schedulers* opcionais (`linear_warmup`, `cosine_warmup`, `reduce_on_plateau`), *gradient clipping* opcional e *early stopping* com restauração dos melhores pesos.
- **LoRA** (via `peft`): congela o codificador e injeta adaptadores de baixa classificação. As cinco cabeças permanecem totalmente treináveis.
- **Quantização pós-treinamento (PTQ):** quantização dinâmica INT8 das camadas `Linear`, para inferência em CPU.
- **Profiling:** *thread* em segundo plano coleta tempo, CPU, RAM e pico de VRAM.
- **Rastreamento de experimentos:** cada execução é registrada em CSV e no **MLflow**, junto com as matrizes de confusão.

---

## Estrutura do projeto

```text
enem-essay-scoring/
├── LICENSE                         # Licença MIT
├── aesconfig.json                  # Configuração central de todos os experimentos
├── config.py                       # Carrega o JSON, define DEVICE e set_seed()
├── task_strategy.py                # Alvos e decodificação por tipo de tarefa
├── requirements.txt                # Dependências (PyTorch com CUDA 11.8)
│
├── corpus/                         # Análise exploratória dos corpora
│   ├── explore_aesenem.ipynb       #   AES-Enem / JBCS2025 (corpus adotado)
│   └── explore_official.ipynb      #   Redações oficiais
│
├── dataset/                        # Dados
│   ├── tokenizer.py                #   Carrega o dataset (HF), tokeniza e normaliza notas
│   └── dataloader.py               #   DataLoaders de treino, validação e teste
│
├── models/                         # Arquitetura
│   ├── bert_models.py              #   BaseModel: encoder compartilhado + 5 cabeças
│   ├── competence_models.py        #   LinearHead e MultiLayerHead
│   ├── lora.py                     #   Injeção dos adaptadores LoRA (peft)
│   └── model_picker.py             #   Monta o modelo a partir do aesconfig.json
│
├── scoring/                        # Treinamento, avaliação e otimização
│   ├── main.py                     #   Ponto de entrada: treino + teste + registro
│   ├── engine.py                   #   Laços de treino/validação/teste
│   ├── losses.py                   #   Perdas (BCE/MSE/CE ponderadas, CORN)
│   ├── metrics.py                  #   QWK, RMSE, F1, acurácias e matriz de confusão
│   ├── optimizer.py                #   AdamW com weight decay seletivo
│   ├── scheduler.py                #   Escalonadores de taxa de aprendizado
│   ├── early_stopping.py           #   Parada antecipada + checkpoint
│   ├── training_config.py          #   Pesos da perda, acumulação, early stopping
│   ├── mapping.py                  #   Mapeia nomes de config para classes (optimizer, loss)
│   └── quantize.py                 #   Quantização pós-treinamento (INT8) e avaliação em CPU
│
├── profiling/
│   └── profiler.py                 # ResourceProfiler (tempo, CPU, RAM, VRAM)
│
├── tuning/                         # Busca sequencial de hiperparâmetros
│   ├── _common.py                  #   Gera configs por trial e dispara scoring.main
│   ├── stage1_bertimbau_baseline.py#   Etapa 1: baseline (BERTimbau Base, ajuste fino completo)
│   └── stage2_bertimbau_lora.py    #   Etapa 2: ajuste do LoRA
│
└── results/
    ├── results_logger.py           # Grava CSV e MLflow
    ├── training_history.py         # Histórico por época/passo e gráficos
    ├── best/                       # Resultados finais reportados na monografia
    │   ├── experiments_log.csv     #   Baseline (linha 1) e LoRA (linha 2)
    │   └── quantization.csv        #   Modelo quantizado (INT8)
    └── empiric/                    # Experimentos de validação estrutural
        ├── ordinal_baseline_seeds.csv
        ├── regression_experiments_log.csv
        ├── classification_experiments_log.csv
        ├── cornloss_experiments_log.csv
        └── heads_bertimbau.csv
```

Os diretórios abaixo são **gerados durante a execução** e não fazem parte do código-fonte:
`checkpoints/`, `mlruns/`, `results/experiments_log.csv`, `results/training_history/` e `tuning/configs/`.

---

## Instalação

### Requisitos

- **Python 3.11 ou superior** (recomendado, para compatibilidade com as versões fixadas no `requirements.txt`).
- **GPU NVIDIA com CUDA** para treinar em tempo razoável. O `requirements.txt` aponta para os *wheels* do PyTorch com **CUDA 11.8**. Sem GPU o código roda na CPU (`DEVICE` é escolhido automaticamente), porém muito mais devagar.
- **Acesso à internet na primeira execução**, para baixar o dataset e os pesos do BERTimbau do Hugging Face.

Referência de hardware usado nos experimentos: AMD Ryzen 5 2400G, 16 GB de RAM e NVIDIA GeForce RTX 3050 de 8 GB (Windows 10).

| Cenário | Consumo observado |
|---|---|
| Ajuste fino completo (batch físico 8, 512 tokens) | ~4,2 GB de VRAM · ~7 min para o treino |
| Ajuste fino com LoRA | ~2,4 GB de VRAM · ~6 min |
| Inferência quantizada (INT8, CPU) | ~2 GB de RAM · modelo de ~171 MB em disco |

### Passo a passo

```bash
# 1. Clonar o repositório
git clone https://github.com/hito-boo/enem-essay-scoring.git
cd enem-essay-scoring

# 2. Criar e ativar um ambiente virtual
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows (cmd/PowerShell)

# 3. Instalar as dependências
pip install -r requirements.txt
```

---

## Como executar

> Todos os comandos devem ser executados **a partir da raiz do repositório**, no formato de módulo (`python -m ...`), pois o código usa *imports* absolutos entre pacotes.

### 1. Treinar e avaliar o modelo (baseline)

```bash
python -m scoring.main
```

Com o `aesconfig.json` fornecido, isso treina o **baseline** da monografia: BERTimbau Base, cabeças lineares, regressão ordinal por Classificação Binária Estendida, `seed = 21`. O fluxo é:

1. fixa a semente (`set_seed`);
2. baixa/carrega e tokeniza o corpus, e cria os `DataLoaders`;
3. monta o modelo e o otimizador;
4. treina com validação por época, *early stopping* e *profiling*;
5. restaura os melhores pesos e avalia no **conjunto de teste**;
6. salva histórico, gráficos, métricas e consumo de recursos (CSV + MLflow).

O terminal exibe, por época, as perdas, o QWK e o RMSE de validação por competência. Ao final exibe uma tabela de métricas de teste e as matrizes de confusão de cada competência.

### 2. Quantizar um modelo treinado

```bash
python -m scoring.quantize
```

Carrega o *checkpoint* indicado em `quantization.checkpoint_path`, aplica quantização dinâmica INT8 nas camadas `Linear`, compara o tamanho em disco antes e depois e avalia o modelo quantizado **na CPU**.

### 3. Buscar hiperparâmetros (opcional)

```bash
python -m tuning.stage1_bertimbau_baseline   # Etapa 1: baseline
python -m tuning.stage2_bertimbau_lora       # Etapa 2: LoRA
```

A busca é **sequencial por coordenadas**: cada *round* varia um grupo de hiperparâmetros mantendo os demais fixos. Para cada *trial*, `tuning/_common.py` gera um JSON em `tuning/configs/<etapa>/`, define `AES_CONFIG_PATH` apontando para ele e executa `scoring.main` em um subprocesso. O resultado de cada *trial* é acrescentado a `results/experiments_log.csv`.

Para escolher o *round* (`ACTIVE_ROUND`), a tarefa (`ACTIVE_TASK_TYPE`, `ACTIVE_ORDINAL_METHOD`), o conjunto de sementes e o `search_space`, edite as constantes e a chamada `run_stage(...)` no final de cada script. Os *rounds* definidos em `SEARCH_SPACE_ROUNDS` correspondem à busca descrita na monografia:

| Round | Hiperparâmetros investigados (Etapa 1) |
|:---:|---|
| 1 | Taxa de aprendizado |
| 2 | Épocas, *scheduler* e acumulação de gradientes |
| 3 | Épocas e passos de acumulação |
| 4 | *Weight decay* e *dropout* do corpo |
| 5 | Expoente `power` (γ) do balanceamento de classes |
| 6 | Balanceamento ligado/desligado e pesos uniformes entre competências |
| 7 | Distribuições de pesos da perda entre competências |

### 4. Acompanhar os experimentos no MLflow

```bash
mlflow ui
```

Abra <http://localhost:5000> e selecione o experimento **`Enem-Essay-Scoring`**. Cada execução registra parâmetros achatados, métricas de teste e validação, e as matrizes de confusão como artefatos de texto.

### 5. Explorar os corpora

```bash
jupyter notebook corpus/
```

---

## Possibilidades de execução

Praticamente tudo é controlado pelo `aesconfig.json`. É possível manter vários arquivos de configuração e escolher qual usar com a variável de ambiente `AES_CONFIG_PATH`:

```bash
# Linux/macOS
AES_CONFIG_PATH=minha_config.json python -m scoring.main

# Windows (PowerShell)
$env:AES_CONFIG_PATH = "minha_config.json"; python -m scoring.main
```

### Variações suportadas

| Objetivo | O que alterar no `aesconfig.json` |
|---|---|
| **Regressão contínua** | `"task_type": "regression"` |
| **Classificação multiclasse** | `"task_type": "classification"` |
| **Ordinal via EBC** (padrão) | `"task_type": "ordinal"`, `"ordinal": {"method": "extended_binary"}` |
| **Ordinal via CORN** | `"task_type": "ordinal"`, `"ordinal": {"method": "corn"}` (o balanceamento de classes é desativado automaticamente) |
| **Cabeças multicamadas** (por competência) | `"transformers": {"c2": "multi", "c3": "multi", ...}` (valores: `"linear"` ou `"multi"`) |
| **Ajuste fino eficiente (LoRA)** | `"lora": {"enabled": true, ...}`, com taxa de aprendizado maior e `gradient_clipping` ligado |
| **Trocar o codificador** | `"transformers": {"bert": "..."}` e `"tokenizer": "..."` (o codificador precisa ter *hidden size* 768) |
| **Desligar o balanceamento de classes** | `"class_balanced_loss": {"enabled": false}` |
| **Pesos por competência na perda** | `"params": {"loss_weights": [...]}` (5 valores não negativos) |
| **Escalonador de taxa de aprendizado** | `"lr_scheduler": {"enabled": true, "type": "linear_warmup" \| "cosine_warmup" \| "reduce_on_plateau"}` |
| **Sem acumulação de gradientes** | `"gradient_accumulation": {"enabled": false}` |
| **Desligar o profiling** | `"profiling": false` |

### Para treinar com LoRA

Configuração usada na monografia (parte do `aesconfig.json`):

```json
"lora": {
    "enabled": true,
    "r": 8,
    "alpha": 16,
    "dropout": 0.1,
    "target_modules": ["query", "value"]
},
"gradient_clipping": { "enabled": true, "max_norm": 1.0 },
"params": { "learning_rate": 4e-4 }
```

Reproduz o modelo com **294.912 parâmetros treináveis no corpo** (0,27% do total).

### Para usar outro codificador

```json
"tokenizer": "PORTULAN/albertina-100m-portuguese-ptbr-encoder",
"transformers": { "bert": "PORTULAN/albertina-100m-portuguese-ptbr-encoder", "body": "base", ... }
```

A infraestrutura já foi validada de ponta a ponta com o **Albertina 100M PT-BR**, mas os hiperparâmetros do baseline foram ajustados apenas para o BERTimbau Base. Para o Albertina é necessária uma nova busca.

### Notas de uso

- Cada execução **acrescenta** uma linha a `results/experiments_log.csv` (nada é sobrescrito).
- O *checkpoint* `checkpoints/best_model.pt` é **sobrescrito** a cada treinamento. Copie-o antes de iniciar outro experimento se quiser preservá-lo.
- O *profiler* é finalizado dentro do bloco do *early stopping* (`scoring/engine.py`). Mantenha `early_stopping.enabled = true` e `restore_best_weights = true` quando `profiling` estiver ligado.
- O parâmetro `drop_out` tem **6 posições**: a primeira é o *dropout* do corpo compartilhado e as cinco seguintes são as cabeças C1 a C5. Já `loss_weights` tem 5 posições (C1 a C5).

---

## Referência do `aesconfig.json`

| Chave | Descrição | Valor padrão |
|---|---|---|
| `dataset_corpus.link` / `.name` | Dataset no Hugging Face e configuração | `kamel-usp/aes_enem_dataset` / `JBCS2025` |
| `tokenizer` | Tokenizador do Hugging Face | `neuralmind/bert-base-portuguese-cased` |
| `transformers.bert` | Codificador pré-treinado | `neuralmind/bert-base-portuguese-cased` |
| `transformers.body` | Tipo de corpo (`base`) | `base` |
| `transformers.c1` … `c5` | Topologia da cabeça de cada competência (`linear` ou `multi`) | `linear` |
| `profiling` | Liga a coleta de recursos de hardware | `true` |
| `optimizer` | Otimizador (`adamw`) | `adamw` |
| `task_type` | `regression`, `classification` ou `ordinal` | `ordinal` |
| `ordinal.method` | `extended_binary` ou `corn` | `extended_binary` |
| `params.seed` | Semente global | `21` |
| `params.learning_rate` | Taxa de aprendizado | `5e-5` |
| `params.weight_decay` | Decaimento de pesos | `0.01` |
| `params.epochs` | Máximo de épocas | `10` |
| `params.drop_out` | *Dropout* `[corpo, C1, C2, C3, C4, C5]` | `[0.1, 0, 0, 0, 0, 0]` |
| `params.loss_weights` | Pesos da perda por competência `[C1…C5]` | `[0.18, 0.21, 0.21, 0.18, 0.22]` |
| `params.batch_size` | Tamanho do lote físico | `8` |
| `lr_scheduler.enabled/type/warmup_ratio` | Escalonador de taxa de aprendizado | desativado |
| `gradient_accumulation.enabled/steps` | Acumulação de gradientes (lote efetivo = `batch_size × steps`) | ativado, `4` |
| `class_balanced_loss.enabled/power` | Balanceamento por frequência inversa (γ) | ativado, `1.0` |
| `early_stopping.*` | `patience`, `min_delta`, `checkpoint_dir`, `checkpoint_name`, `restore_best_weights` | `3`, `1e-4`, `checkpoints`, `best_model.pt`, `true` |
| `lora.*` | `enabled`, `r`, `alpha`, `dropout`, `target_modules` | desativado, `8`, `16`, `0.1`, `["query","value"]` |
| `gradient_clipping.enabled/max_norm` | Recorte de gradiente pela norma global | desativado, `1.0` |
| `quantization.checkpoint_path` | *Checkpoint* a ser quantizado | `checkpoints/bertimbau-baseline.pt` |
| `training_evaluation.output_dir` | Pasta do histórico de treino | `results/training_history` |

---

## Saídas geradas

| Arquivo / pasta | Conteúdo |
|---|---|
| `checkpoints/best_model.pt` | Melhor modelo segundo a perda de validação |
| `results/experiments_log.csv` | Uma linha por execução com configuração, métricas de validação e teste por competência, matrizes de confusão e consumo de hardware |
| `results/training_history/history.csv` e `history.png` | Perdas, RMSE e QWK por época e competência, e taxa de aprendizado |
| `results/training_history/train_loss_steps.csv` e `eval_loss_steps.csv` | Curva de aprendizado por passo |
| `results/training_history/learning_curve.png` | Perda de treino × validação |
| `mlruns/` | Registros locais do MLflow |

### Métricas reportadas

| Métrica | Significado |
|---|---|
| **QWK** | *Quadratic Weighted Kappa*, a concordância ordinal com o avaliador humano (métrica principal) |
| **RMSE** | Erro quadrático médio, na escala oficial de 0 a 200 pontos |
| **F1-Macro / F1-Ponderado** | F1 por competência, sem e com peso pelo suporte de cada nota |
| **Acc** | Acurácia exata |
| **AccR** (Acurácia Relaxada) | Fração de predições com erro de até **±40** pontos (um nível) |
| **AccE** (Acurácia Padrão Enem) | Fração de predições com erro de até **±80** pontos, a tolerância oficial do exame |

---

## Resultados

Todos os resultados abaixo são do **conjunto de teste** do JBCS2025 (partição *cross-prompt*), obtidos com o baseline da monografia (`seed = 21`), salvo quando indicado.

### Baseline: BERTimbau Base + 5 cabeças lineares + regressão ordinal (EBC)

| Competência | QWK | RMSE | F1-Macro | Acc | AccR | AccE |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| C1, Norma padrão | 0,642 | 30,83 | 0,301 | 0,507 | 0,978 | 0,993 |
| C2, Tema e repertório | 0,300 | 56,36 | 0,175 | 0,239 | 0,717 | 0,935 |
| C3, Argumentação | 0,406 | 50,62 | 0,227 | 0,304 | 0,775 | 0,964 |
| C4, Coesão | 0,556 | 35,06 | 0,328 | 0,486 | 0,928 | 0,993 |
| C5, Proposta de intervenção | 0,616 | 45,68 | 0,337 | 0,399 | 0,790 | 0,986 |
| Média | 0,504 | 43,71 | 0,273 | 0,387 | 0,838 | 0,974 |

- **Estabilidade (3 sementes: 21, 42 e 7):** QWK médio de **0,488 ± 0,016**, RMSE de 41,85 ± 1,67, AccR de 86,0 ± 1,9% e AccE de 97,7 ± 0,9%. A hierarquia entre as competências se manteve em todas as sementes.
- **Padrão por competência:** as competências formais e coesivas (C1 e C4) e a de proposta de intervenção (C5) são as mais bem previstas. As competências C2 (tema e repertório) e C3 (argumentação) são as mais difíceis, o que a monografia atribui à abstração semântica exigida e à ausência dos textos motivadores na entrada.
- **Erros próximos:** na maioria dos casos em que o modelo erra a nota exata, ele erra por apenas um nível. Em média, 97,4% das predições ficam dentro da tolerância oficial do Enem.

### Otimização computacional

| | Baseline (FP32) | LoRA (r = 8, α = 16) | Quantizado (INT8) |
|---|:---:|:---:|:---:|
| **QWK médio** | 0,504 | 0,426 (84,5% do baseline) | 0,435 (86,3% do baseline) |
| RMSE médio | 43,71 | 45,89 | 42,51 |
| AccR / AccE | 83,8% / 97,4% | 84,3% / 95,1% | 84,6% / 97,8% |
| Parâmetros treináveis | 109.218.048 | **294.912 (0,27%)** | n/a (pós-treino) |
| Pico de VRAM | 4.231,65 MB | **2.427,73 MB (−42,6%)** | 0 MB (inferência em CPU) |
| Tempo de treino | 430,20 s | **359,92 s (−16,3%)** | n/a |
| Tamanho em disco | 415,58 MB | 416,78 MB | **170,98 MB (−58,9%)** |
| Pico de RAM | 1.382 MB | 1.400 MB | 2.026 MB |
| Dispositivo | GPU | GPU | CPU |

- **LoRA** permite ajustar o modelo em GPUs domésticas de entrada, ao custo de uma queda de QWK concentrada principalmente na C5.
- **PTQ INT8** elimina a dependência de GPU. Avaliar todo o conjunto de teste em CPU levou 127,72 s (o tempo é de inferência, e não comparável ao de treino). Em compensação, o modelo quantizado emite menos notas extremas, o que reduz o QWK em C5.

## Resultados incluídos no repositório

| Arquivo | Conteúdo |
|---|---|
| `results/best/experiments_log.csv` | **Baseline** (linha 1) e **LoRA** (linha 2) |
| `results/best/quantization.csv` | Modelo **quantizado INT8** |
| `results/empiric/ordinal_baseline_seeds.csv` | Baseline com as sementes 21, 42 e 7 |
| `results/empiric/regression_experiments_log.csv` | Regressão contínua (variações de taxa de aprendizado) |
| `results/empiric/classification_experiments_log.csv` | Classificação multiclasse (variações de taxa de aprendizado) |
| `results/empiric/cornloss_experiments_log.csv` | Regressão ordinal condicional (CORN) |
| `results/empiric/heads_bertimbau.csv` | Cabeças lineares × configuração híbrida (C2 e C3 em MLP) |

Os CSVs seguem o mesmo esquema (configuração achatada, métricas `C{i}_QWK`, `C{i}_RMSE`, `C{i}_F1_*`, `C{i}_Acc*`, matrizes de confusão, métricas de validação e consumo de hardware) e podem ser lidos diretamente com `pandas`:

```python
import pandas as pd

df = pd.read_csv("results/best/experiments_log.csv")
print(df[[f"C{i}_QWK" for i in range(1, 6)]].mean(axis=1))   # QWK médio por execução
```

---

## Limitações e trabalhos futuros

**Limitações**

- Corpus pequeno e desbalanceado: poucas redações nas notas extremas, o que limita cabeças maiores e o método CORN.
- A janela de **512 tokens** impede incluir a proposta temática e os textos motivadores, informações relevantes para C2 e C3 em avaliação *cross-prompt*.
- Subjetividade inerente à tarefa: a concordância entre humanos no benchmark fica entre 0,450 e 0,640 de QWK.
- Competição entre gradientes no MTL: critérios abstratos (C2, C3, C5) e superficiais (C1, C4) convergem em velocidades diferentes, dificultando uma hiperparametrização única.
- A busca de hiperparâmetros e as otimizações concentraram-se no BERTimbau Base. O Albertina serviu apenas para validar a modularidade da infraestrutura.

**Trabalhos futuros**

- Transferência em múltiplos estágios, combinando redações simuladas e o corpus de redações oficiais, com aumento de dados para as notas extremas.
- Outros codificadores para o português (Albertina, BERTuguês) e modelos de janela estendida, para incluir os textos motivadores na entrada.
- Modelos híbridos que fundam o vetor contextual do Transformer com características linguísticas explícitas (TF-IDF, NILC-Metrix).

---

## Referências principais

- **AES-Enem / JBCS2025:** Silveira, I. C.; Barbosa, A.; Mauá, D. D. (2024). Corpus e benchmark de AES para redações do Enem. Disponível em [`kamel-usp/aes_enem_dataset`](https://huggingface.co/datasets/kamel-usp/aes_enem_dataset).
- **BERTimbau:** Souza, F.; Nogueira, R.; Lotufo, R. (2020). *BERTimbau: pretrained BERT models for Brazilian Portuguese*. Modelo: [`neuralmind/bert-base-portuguese-cased`](https://huggingface.co/neuralmind/bert-base-portuguese-cased).
- **Classificação Binária Estendida (EBC):** Li, L.; Lin, H.-T. (2007). *Ordinal regression by extended binary classification*.
- **CORN:** Shi, X.; Cao, W.; Raschka, S. (2023). *Deep neural networks for rank-consistent ordinal regression based on conditional probabilities*. Biblioteca: [`coral-pytorch`](https://github.com/Raschka-research-group/coral-pytorch).
- **LoRA:** Hu, E. J. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. Biblioteca: [`peft`](https://github.com/huggingface/peft).
- **Albertina:** Santos, R. et al. (2024). Modelo: [`PORTULAN/albertina-100m-portuguese-ptbr-encoder`](https://huggingface.co/PORTULAN/albertina-100m-portuguese-ptbr-encoder).
- **Trabalhos comparados:** Barbosa, Silveira e Mauá (2025); Chalegre, Machado e Feltrim (2026); Carpi et al. (2026).

As referências completas estão na monografia.

---

## Licença

Este projeto é distribuído sob a **Licença MIT**. Copyright © 2026 Vitor da Rocha Machado. O texto completo está no arquivo [`LICENSE`](LICENSE).

Você pode usar, copiar, modificar e distribuir o código livremente, inclusive para fins comerciais, desde que mantenha o aviso de copyright e o texto da licença em todas as cópias ou partes substanciais do software. O software é fornecido "como está", sem garantias de qualquer tipo.

> A licença MIT cobre o **código deste repositório**. O corpus AES-Enem e os modelos pré-treinados (BERTimbau, Albertina) são obtidos do Hugging Face e possuem seus próprios termos de uso e licenças, que devem ser consultados nas respectivas páginas.