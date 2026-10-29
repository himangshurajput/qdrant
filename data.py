from sklearn.datasets import fetch_20newsgroups

SAMPLE_SIZE = 6000
MIN_CHARS = 100
MAX_CHARS = 4000
RANDOM_STATE = 42


def load_documents(sample_size=SAMPLE_SIZE, random_state=RANDOM_STATE):
    dataset = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=random_state,
    )
    documents = []
    for text, label in zip(dataset.data, dataset.target):
        text = text.strip()
        if len(text) < MIN_CHARS:
            continue
        documents.append(
            {
                "id": len(documents),
                "category": dataset.target_names[label],
                "text": text[:MAX_CHARS],
            }
        )
        if len(documents) == sample_size:
            break
    return documents


if __name__ == "__main__":
    docs = load_documents()
    print(f"loaded {len(docs)} documents")
    print(f"categories: {len({d['category'] for d in docs})}")
    print(docs[0]["category"])
    print(docs[0]["text"][:300])
