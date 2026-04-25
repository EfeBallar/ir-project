import json, datasets
from pathlib import Path

R_NUMBER_SEED = 1072381
DOCS_TO_ADD = 2000

DATA_PATH = Path("./anthology_sample.parquet")
QUERY_DOCS_PATH = "./acl_anthology_queries.parquet"
FULL_DOCS_PATH = "./acl_anthology_full.parquet"
QUERIES_PATH = Path("./acl_anthology_queries.json")

def build_sample_if_missing(data_path=DATA_PATH):
    """Build anthology sample parquet once, then reuse it."""
    data_path = Path(data_path)

    if data_path.exists():
        print(f"[part2_data] using cached sample: {data_path}")
        return data_path

    print("[part2_data] building anthology sample parquet...")
    query_documents = datasets.load_dataset("parquet", data_files=QUERY_DOCS_PATH)[
        "train"
    ]
    all_documents = datasets.load_dataset("parquet", data_files=FULL_DOCS_PATH)["train"]

    random_documents = all_documents.shuffle(seed=R_NUMBER_SEED).take(DOCS_TO_ADD)
    anthology_sample = datasets.concatenate_datasets(
        [query_documents, random_documents]
    ).shuffle(seed=R_NUMBER_SEED)

    anthology_sample.to_parquet(str(data_path))
    print(f"[part2_data] wrote sample: {data_path} ({len(anthology_sample)} docs)")
    return data_path


def load_anthology(data_path=DATA_PATH):
    """Load the cached anthology sample as a datasets.Dataset."""
    data_path = Path(data_path)
    if not data_path.exists():
        build_sample_if_missing(data_path=data_path)
    return datasets.load_dataset("parquet", data_files=str(data_path))["train"]


def load_queries(queries_path=QUERIES_PATH):
    """Load Part 2 query set and return list of query entries."""
    queries_path = Path(queries_path)
    with open(queries_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("queries", [])


def answerable_queries(queries):
    """Filter out queries without any relevant documents."""
    if isinstance(queries, dict):
        queries = queries.get("queries", [])
    return [entry for entry in queries if len(entry.get("r", [])) > 0]
