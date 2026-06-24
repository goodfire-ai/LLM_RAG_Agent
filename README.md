# Repository for: Autonomous AI Agents for Clinical Decision Making in Oncology
⚠️ This repository is currently under construction. Usage might change in the future. 

⚠️ The current agent implementation uses test functions for the image segmentation and genetic modeling tasks as the original implementation requires external repositories that are challenging to setup. We are working on a solution to simplify their setup in the very near future. The provided test functions (```agent_tools_dummy.py```) are implemented as agent tools without any changes to their original implementation (```agent_tools.py```) and have therefore *no* influence on the LLM-Agents behaviour.

![Local Image](./overview.png)

---

## `ferber_agent` — modernized, pip-installable package

`ferber_agent/` is a modern reimplementation of this agent's *method* on the current OpenAI
SDK (function-calling / Responses) + chromadb, with a frontier backbone (default `gpt-5.1`).
The original `RAGent/DSPY/` code (dspy + llama-index 0.9 + vendored agent + cohere) is preserved
above for reference but its dependencies are deprecated.

On top of the two pipeline modes (default / faithful), the agent exposes three orthogonal
switches:

- **Execution backend** (`backend`): `chat_completions` (default) · `responses_faithful` ·
  `native_agentic` — *how* the OpenAI plumbing runs the pipeline.
- **Web search** (`web_search`): an OpenAI-native web-search tool that replaces the
  discontinued Google Custom Search the paper used.
- **Retrieval engine** (`retrieval_engine`): `chroma_cosine` (default) · `chroma_cohere` ·
  `openai_filesearch_responses` · `openai_filesearch_chat` — *how* faithful Stage-2 guideline
  retrieval is done.

All three default to the prior behavior (faithful chat-completions over cosine Chroma, web
search off), so existing usage is unchanged. Install:

```bash
pip install -e .              # core text track (API/CPU, no torch)
pip install -e .[imaging]     # + radiology_report (vision) and medsam (torch + transformers)
pip install -e .[rerank]      # + Cohere reranking
```

```python
from ferber_agent import FerberAgent
agent = FerberAgent(chroma_dir="/path/to/chroma", rerank=True,
                    tools=("rag","oncokb","pubmed","calculate",
                           "radiology_report","medsam","histology_classifier"))
res = agent.answer(context, question, images={"September2023.png": "/abs/Xing_1.jpg"})
```

### Tool status
| Tool | Status | Notes |
|------|--------|-------|
| `rag` | ✓ | guideline retrieval from a Chroma index (OpenAI embeddings) |
| `rerank` | ✓ | real Cohere `rerank-english-v3.0`; **cosine-only fallback** without `COHERE_API_KEY` |
| `oncokb` | ✓ | prod endpoint with `ONCOKB_API_TOKEN`, else the public **demo** endpoint (common variants only) |
| `pubmed` | ✓ | NCBI E-utilities |
| `web_search` | ✓ | OpenAI-native web search (replaces the paper's discontinued Google Custom Search); off by default, enable with `web_search=True` |
| `calculate` | ✓ | safe arithmetic (e.g. lesion-area progression ratios) |
| `radiology_report` | ✓ | GPT-4V-style structured report from a patient image (vision backbone) |
| `medsam` | ✓ | MedSAM (transformers `SamModel`, Wang Lab HF mirror); bbox prompt → lesion area in px |
| `histology_classifier` | ✗ **hard gap** | the in-house KRAS/BRAF/MSI H&E classifier was never released; returns an explicit "unavailable, use the molecular report" message |

Imaging-tool schemas are only exposed when the call carries an `images` map, so the
text-only genomic track is unaffected. `resolve_image` matches the model-supplied filename
(vignette date-name *or* on-disk basename, extension-insensitive) to a path.

### Two modes: default vs faithful

`FerberAgent` runs in either of two modes (the constructor flag `faithful`):

- **Default** (`faithful=False`) — a compact two-stage loop: autonomous tool use, then a
  single RAG-grounded, cited answer. This is the lightweight track (e.g. for MTBBench).
- **Faithful** (`faithful=True`) — the **paper's full multi-stage pipeline**, with the
  **verbatim upstream prompt strings** (see `ferber_agent/faithful_prompts.py`):

  1. **Stage 1 — tool gathering**: the agent loop (verbatim agent + `chat_ext` prompts) calls
     the patient-evidence tools (`oncokb` / `pubmed` / `calculate`, plus the imaging tools when
     images are supplied), nudging once to use all useful tools. Guideline `rag` is **not** a
     callable tool here.
  2. **Stage 2 — mandatory guideline grounding**: `Search` expands the question into focused
     subqueries (up to `n_subqueries`, default 12) → retrieve top‑`retrieve_k` (default 40) per
     subquery → Cohere rerank top‑`rerank_top_n` (default 10) → dedup union → `AnswerStrategy`
     → `GenerateCitedResponse` (passages are prefixed `Source N:` so `[n]` citations resolve) →
     optional one-pass **citation self-evaluation** → `Suggestions`.

```python
from ferber_agent import FerberAgent
agent = FerberAgent(
    chroma_dir="/path/to/chroma", faithful=True, rerank=True,
    n_subqueries=12, retrieve_k=40, rerank_top_n=10, max_function_calls=10,
    citation_selfeval=True,
)
res = agent.answer(context, question,
                   images={"September2023.png": "/abs/Xing_1.jpg"},  # optional, enables imaging
                   case_key="Adams")                                  # keys the histology replay
print(res.answer_text)        # cited answer + suggestions
print(res.citations, len(res.retrieved))
```

### Configuration knobs (faithful mode)

| Knob (constructor / env) | Default | Effect |
|---|---|---|
| `backend` | `chat_completions` | execution backend: `chat_completions` / `responses_faithful` / `native_agentic` |
| `web_search` / `FERBER_WEB_SEARCH` | off | enable the OpenAI-native web-search tool |
| `retrieval_engine` | derived (`chroma_cosine`) | Stage‑2 retrieval engine (see below); unset → derived from `rerank` |
| `vector_store_id` / `OPENAI_VECTOR_STORE_ID` | unset | OpenAI vector store for the `openai_filesearch_*` engines |
| `n_subqueries` | 12 | max Stage‑2 subqueries fanned out for retrieval |
| `retrieve_k` | 20 (40 in the paper track) | passages retrieved per subquery |
| `rerank_top_n` | 10 | passages kept per subquery after Cohere rerank |
| `max_function_calls` | 10 | Stage‑1 tool-gathering rounds |
| `agent_temp` / `rag_temp` | 0.2 / 0.1 | sampling temps for the agent vs RAG stages (GPT‑4-era models only) |
| `citation_selfeval` / `FERBER_CITATION_SELFEVAL` | off | run the paper's one-pass faithfulness check + single revise |
| `rag_workers` / `RAG_WORKERS` | 12 | thread-pool width for subquery retrieval |
| `citation_workers` / `CITATION_WORKERS` | 12 | thread-pool width for citation checks |
| `EMBED_CACHE_DIR` | unset | disk cache for query embeddings + reranked results (cross-process) |
| `HISTOLOGY_LOOKUP` | bundled file | override the histology replay lookup JSON |

### Execution backends

`backend` selects *how* the OpenAI plumbing runs the pipeline. The faithful pipeline is
identical across the first two (same verbatim prompts, same explicit stages) — only the
transport differs:

- **`chat_completions`** (default) — chat.completions function-calling. Web search, when
  enabled, is a callable function tool (a nested Responses `web_search` call).
- **`responses_faithful`** — the SAME explicit faithful stages over the Responses API, carrying
  reasoning items across tool calls and attaching the hosted `web_search` tool directly.
- **`native_agentic`** — OpenAI's Responses runtime drives the loop itself (the faithful
  evidence tools + a callable `rag` tool + hosted web search), rather than the explicit staged
  pipeline. This is a *different* agent, not the paper-faithful pipeline.

```python
agent = FerberAgent(chroma_dir="/path/to/chroma", faithful=True,
                    backend="responses_faithful", web_search=True)
```

Per-rollout token usage, hosted-tool calls, latency, and an approximate USD cost are tracked on
`agent._usage` (see `ferber_agent/usage.py`); `agent._usage.summary()` returns a serializable
record after each `answer()`.

### Web search

The paper's agent had a `google_search` tool (a llama-hub `GoogleSearchToolSpec` agent keyed on
`GOOGLE_API_KEY` / `GOOGLE_SEARCH_ENGINE`). The Google Custom Search whole-web endpoint was
discontinued, so the capability is restored via OpenAI's native web search (`web_search` in
`ferber_agent/tools.py`): a nested Responses `web_search` call that returns a source-cited
summary plus the cited URLs. It is off by default; enable with `web_search=True` (or
`FERBER_WEB_SEARCH=1`). It degrades to an explicit message — never raises — on any search
failure.

### Retrieval engines

`retrieval_engine` selects how faithful Stage‑2 guideline retrieval is done (see
`ferber_agent/retrieval.py`). Every engine implements the same `retrieve(query, retrieve_k,
top_n)` contract and returns normalized passage dicts:

| Engine | What it does |
|---|---|
| `chroma_cosine` (default) | Chroma top‑`retrieve_k` cosine → top‑`rerank_top_n` (no rerank) |
| `chroma_cohere` | Chroma cosine → Cohere rerank → top‑`rerank_top_n` (cosine-only without `COHERE_API_KEY`) |
| `openai_filesearch_responses` | OpenAI Responses `file_search` over a vector store of the source docs |
| `openai_filesearch_chat` | the SAME vector store, retrieved via a chat.completions `file_search` function-tool wrapper |

When `retrieval_engine` is unset it is derived from the legacy `rerank` flag (`chroma_cohere`
if `rerank=True` and a Cohere key is present, else `chroma_cosine`), so existing callers are
unchanged. The two `chroma_*` engines delegate to the cached `RagTool.retrieve_reranked` path,
so they reuse the embedding/rerank cache and produce output identical to the inline Chroma
retrieval (the result-preserving speedup below applies to all engines via the same parallel
fan-out).

**Recommendation (from experiment #24):** the four engines were tied on answer quality on
ferber20, so `chroma_cosine` — the cheapest, with no Cohere dependency and no OpenAI vector
store to maintain — is the recommended default. The `chroma_cohere` and `openai_filesearch_*`
engines are available for comparison but did not improve the result.

#### Building the vector store (for the `openai_filesearch_*` engines)

The file_search engines query an OpenAI vector store of the guideline corpus. Build one with
`scripts/build_vector_store.py` (pure network I/O; needs `OPENAI_API_KEY`):

```bash
python scripts/build_vector_store.py --corpus-dir /path/to/corpus --out vector_store.json
# corpus = {source}.jsonl files with a clean_text field (the same corpus the Chroma index uses)
```

It uploads one file per document with `{source, title, doc_id}` attributes (so retrieved chunks
carry provenance), waits for indexing, and records the store id. Then point the agent at it:

```python
agent = FerberAgent(chroma_dir="/path/to/chroma", faithful=True,
                    retrieval_engine="openai_filesearch_responses",
                    vector_store_id="vs_...")  # or OPENAI_VECTOR_STORE_ID
```

The build is idempotent: an already-fully-indexed recorded store is reused unless `--force`.

### Result-preserving speedup

The two independent Stage‑2 workloads — per-subquery retrievals and per-statement citation
checks — fan out across a thread pool with **order-preserving reassembly**, so the parallel
run is byte-identical to the serial run (`tests/test_equivalence.py` proves this offline). The
query embedding (the slow OpenAI round-trip) is computed outside the Chroma lock and cached,
and reranked results are cached per `(query, k, top_n)`; since `text-embedding-3-large` is
deterministic this cannot change a result, and it additionally makes retrieval reproducible
run-to-run. In the source experiment this cut faithful-generation wall time ~3.7× (1968s →
526s) with no change to the outcome.

### Histology replay (replaces the unavailable in-house classifier)

The paper's `check_mutations` tool ran proprietary H&E image classifiers that were never
released, so they are **not reproducible**. Faithful mode instead **replays** the paper's
pre-extracted per-case MSI/KRAS/BRAF predictions via `histology_replay`, which reads a lookup
JSON keyed by case surname. The lookup bundled at `ferber_agent/data/histology_lookup.json`
(20 ferber20 cases) was built from the **public paper supplementary** by
`scripts/build_histology_lookup.py`; override it with `HISTOLOGY_LOOKUP`. A case with no
documented prediction returns an explicit gap message — predictions are never fabricated.

### Prompt fidelity

`ferber_agent/faithful_prompts.py` holds the **26 verbatim prompt blocks** copied
character-for-character from the original dspy source vendored in this repo (`RAGent/DSPY/`).
They are generated by `scripts/extract_prompts.py` (AST extraction, no retyping), and
`tests/test_prompt_fidelity.py` re-extracts and asserts byte-equality, so any drift fails CI.

### Faithfulness caveats (what this is NOT)
- **Backbone**: default `gpt-5.1`, not the paper's GPT‑4 — so the paper's absolute scores are a
  reference, not a reproduction target.
- **Histology**: predictions are *replayed* from the supplementary, not produced by a live
  classifier (the in-house ViTs are unreleased).
- **DSPy machinery**: the prompts are verbatim, but the dspy/llama-index orchestration is
  reimplemented on the OpenAI SDK; there is no DSPy compilation/optimization or backtracking
  loop (the citation self-eval is a single pass, matching the paper's description).
- **KRAS/BRAF/MSI in default mode**: surfaced as an explicit gap, never faked.
- **OncoKB demo endpoint**: full annotations for common alterations (e.g. BRAF V600E → 30
  treatments); rarer variants may return "Unknown" until a prod `ONCOKB_API_TOKEN` is set.
- **Reranking** degrades to cosine order without a Cohere key (flagged on the agent).

### Tests & scripts
- `tests/test_tools.py`, `tests/test_faithful_pipeline.py`, `tests/test_equivalence.py`,
  `tests/test_prompt_fidelity.py`, `tests/test_backends.py`, `tests/test_web_search.py`,
  `tests/test_retrieval.py` — all **hermetic** (no OpenAI / Chroma / Cohere / GPU; OpenAI
  clients are stubbed). Run with `PYTHONPATH=. pytest tests/` (skipping `test_smoke.py`, which
  is a live end-to-end check needing `OPENAI_API_KEY`). The backend / web-search /
  retrieval-engine switches each have unit tests covering construction, the exposed tool set,
  and a mocked file_search path.
- `scripts/extract_prompts.py` — regenerate the verbatim prompt module from `RAGent/DSPY`.
- `scripts/build_histology_lookup.py` — rebuild the histology replay lookup from a
  supplementary text (needs `ANTHROPIC_API_KEY`; `pip install -e .[histology]`).
- `scripts/build_vector_store.py` — build the OpenAI vector store for the `openai_filesearch_*`
  retrieval engines (needs `OPENAI_API_KEY`).

### Dependencies & provenance
All runtime dependencies are **public** (openai, chromadb, requests, tiktoken; cohere, torch +
transformers + the `wanglab/medsam-vit-base` weights, and anthropic only for the optional
extras). There are no internal/private package dependencies.

This package synthesizes the validated code from several research experiments: a paper-faithful
rebuild (faithful 79.4% vs bare 64.0% completeness, +15.3pp, bootstrap CI [+6.2, +24.5]) and a
validated result-preserving speedup (3.7× faster faithful generation; +17.4pp, Wilcoxon
p=0.0035, 16/20 cases favoring faithful); the **execution-backend switch + web-search tool**
(experiment #21, which decomposed the native-plumbing options) and the **retrieval-engine
switch + file_search engines + vector-store builder** (experiment #24, which found the four
retrieval engines tied on ferber20 — hence `chroma_cosine` as the cheapest tied-best default).
The full ferber20 numeric evaluation lives with those experiments and is not re-run here.

---

## Software Requirements
All experiments were run on an Apple MacBook Pro M2 Max 96GB 2023.
No special hardware is required, if you wish to run certain models with hardware acceleration, it is recommended to have a CUDA-compatible GPU to speed up the process.

## General Setup Instructions

Please follow the steps below:

#### 1. **Python Installation**:
Install Python from source. We used Python 3.11.6 throughout this project. 
#### 2. **Dependency Installation**: 

Clone this repository:
  ```
  git clone https://github.com/Dyke-F/LLM_RAG_Agent.git
  ```

This process might take around 1 minute.

Set up a clean python3 virtual environment, i.e. 

  ```
  python3 -m venv medvenv
  source medvenv/bin/activate
  ```

Install necessary dependencies:
  ```bash
  pip install -r requirements.txt
  ```

3. **Repository Structure**:
```
.env
RAGent/DSPY.
├── agent_tools_dummy.py                 # dummy implementation of agent tools returning defaults for fast debugging and demonstrations
├── agent_tools.py                       # Actual implementation of the agent tools. 
├── chroma_db_retriever.py               # Retriever Class for RAG, modified from DSPY's implementation to run via HTTP Client.
├── citation_utils.py                    # Utility function for Citation Checking in the Agent's output.
├── deduplicate_data.py                  # Remove duplicated files (if exist).
├── embed.py                             # Core script to generate text embeddings from medical texts and create a permanent Chroma db storage.
├── filter_data_sources.py               # Script to preprocess and clean data to relevant topics.
├── loguru_logger.py                     # Implementation of the main logger.
├── med_agent.py                         # Implementation of the MedAgent class from LLama-Indexes OpenAI Agent class.
├── patient_cases.py                     # Patient cases for the experiments.
├── preprocess_logger.py                 # Implementation of the preprocessing logger.
├── preprocess_sources.py                # Unify data and add IDs.
├── rag_config.py                        # Configuration file with defaults for the embedding and db creation.
├── rag_logger.py                        # Logger for retrieval.
├── rag_utils.py                         # Utility functions for RAG metadata etc.
├── rag.py                               # Main implementation of embeddings and RAG class and loaders.
├── run_experiment.ipynb                 # Main notebook to run an experiment.
├── scrape_meditron.py                   # Download and convert meditron guidelines data.
├── signatures.py                        # DSPY signatures (Prompts).
└── utils.py                             # Utility functions for display etc.
```

## Setup

This repository requires access to the following APIs: GPT-4 and GPT-4V, Cohere Reranking, Google Search and Querying the OncoKB. If you do not have one, create an account and generate an API key for each. While OpenAI and Cohere require a paid tier, the Google Search API is free. For OncoKB an academic license can be requiested for research purposes. Check for further information here:
- https://openai.com/blog/openai-api 
- https://dashboard.cohere.com/welcome/register
- https://developers.google.com/custom-search/v1/introduction?hl=de
- https://www.oncokb.org/api-access


After generating an API key, copy it and place it in a **.env** file in the main directory of this repository.
The ```.env``` file should look like this:

```
OPENAI_API_KEY="sk-******************" # Place your API key here
COHERE_API_KEY="*********************" # Place your API key here
GOOGLE_API_KEY="*********************" # Place your API key here
GOOGLE_SEARCH_ENGINE="***************" # Place your backend here
```

## Experiments
#### 1. Download medical guidelines.

For instance, meditron guidelines are available at: https://huggingface.co/datasets/epfl-llm/guidelines. You can use the ```scrape_meditron.py``` file for this. Please define your download directory.

#### 2. Data Cleaning (Optional):

Given your data, you might want to perform optional data cleaning or pre-processing. This step is highlighy dependant on your data source and can vary a lot. 
    Examples for data cleaning can be found here: ```https://github.com/epfLLM/meditron/blob/main/gap-replay/guidelines/clean.py```. We have used modifications and own implementations for data cleaning.

  ⚠️ The only requirement is that your data is stored as ```.jsonl``` file with at least one document that has a field ```clean_data``` and eventually contains metadata fields.


#### 3. Preprocess the data: 
- I. Run the ```filter_data_sources.py``` file to filter the data for a specific topic (based on keywords) by either modifying the file or setting the ```--keywords``` argument. Define each data source as ```--to_filter``` to apply filtering or as ```--to_copy``` to ignore filtering if you have multiple .jsonl data files in the "data/" directory.
- II. Run ```deduplicate_data.py``` by seetting an ```--in_directory``` and ```--out_directory``` and the data files in the respective directory you want to apply deduplication to.
- III. Run ```preprocess_sources.py``` to add IDs and prepare the metadata for embedding by setting ```--directory```.

#### 4. Generate text embeddings and storage.
  - I. Eventually modify ```rag_config.py``` as desired. The ```RAGConfig``` class contains comments that explain each possible setting.
  - II. Set the metadata that shall be used during embedding in ```rag_utils.py``` in MetadataFields. The name shall be the file name for your data (i.e. if your data is called ```guidelines.jsonl```) then place ```GUIDELINES``` as a name and set all fields you want as your metadata as they are named in the dataset ```.jsonl``` file. Also your dataset ```.jsonl``` file must have a field named ```clean_text```, which is the main text for embedding. This field must be manually created beforehand or set during data cleaning. 
  - III. Via Terminal execute: ```chroma run --path ...``` where ```--path``` equals the default_client_path in rag_config.RAGConfig.
  - IV. Once a ChromaDB HTTP client is setup, in a new terminal run: ```python3 embed.py --to_embed ...``` where ```to_embed```lists all datafiles you want to generate embeddings for.

5. Define your test cases in ```patient_cases.py```. Upload any relevant data (like CT images) into a directory called ```Imaging```.
6. Execute and eventually modify the cells in ```run_experiment.ipynb``` to test the agent on the respective patient (by filename). This file provides a minimal working implementation of the agent calling test-tools. These tools do not actually run in the background, but provide the exact same interface to the model. We work on releasing a full-working solution in the coming weeks.


⚠️ DSPY naturally caches results, which we observe could lead to unexpected behaviour when composing modules. You can disable this behaviour by setting ```cache_turn_on = False``` in ```dsp/modules/cache_utils.py``` and force deletion of the cache directory through ```rm -rf cachedir_joblib``` (located in the home directory).