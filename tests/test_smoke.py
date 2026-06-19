"""Standalone smoke test for the modernized ferber_agent (real gpt-5.1 call, tiny index)."""
import os
import tempfile

import chromadb
from chromadb.utils import embedding_functions

from ferber_agent import FerberAgent, FerberResult

EMBED = "text-embedding-3-large"
DOCS = [
    ("braf_crc", "For BRAF V600E mutant metastatic colorectal cancer, guidelines recommend "
                 "encorafenib plus cetuximab after prior therapy; BRAF inhibitor monotherapy "
                 "is not effective in colorectal cancer."),
    ("braf_mel", "In BRAF V600E melanoma, combined BRAF/MEK inhibition (e.g. dabrafenib plus "
                 "trametinib) is a standard first-line targeted option."),
    ("msi_io", "Microsatellite instability-high tumors respond to immune checkpoint "
               "inhibitors such as pembrolizumab regardless of tissue of origin."),
]


def build_tiny_index(path: str):
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"], model_name=EMBED)
    client = chromadb.PersistentClient(path=path)
    coll = client.get_or_create_collection(
        name="oncology_db", embedding_function=ef, metadata={"hnsw:space": "cosine"})
    coll.add(ids=[d[0] for d in DOCS], documents=[d[1] for d in DOCS],
             metadatas=[{"title": d[0], "source": "test", "article_source": "test"} for d in DOCS])
    return coll.count()


def main():
    with tempfile.TemporaryDirectory() as tmp:
        n = build_tiny_index(tmp)
        print(f"tiny index docs: {n}")
        agent = FerberAgent(chroma_dir=tmp, llm_model=os.environ.get("FERBER_MODEL", "gpt-5.1"),
                            retrieve_k=5)
        ctx = ("65-year-old with metastatic colorectal adenocarcinoma. Molecular: BRAF V600E "
               "mutation detected; microsatellite stable.")
        q = ("What targeted therapy is recommended?\nA) Encorafenib + cetuximab\n"
             "B) Vemurafenib monotherapy")
        res: FerberResult = agent.answer(ctx, q)
        print("=== answer_text (head) ===")
        print(res.answer_text[:600])
        print("=== tool_calls ===", [(c["tool"], list(c["args"].values())[:1]) for c in res.tool_calls])
        print("=== n_retrieved ===", len(res.retrieved))
        assert res.answer_text.strip(), "empty answer"
        assert len(res.retrieved) >= 1, "no retrieval"
        assert isinstance(res.tool_calls, list)
        print("\nSTANDALONE FERBER SMOKE: PASS")


if __name__ == "__main__":
    main()
