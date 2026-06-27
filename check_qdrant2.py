import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchText

load_dotenv()

client = QdrantClient(
    host=os.getenv("QDRANT_HOST"), port=int(os.getenv("QDRANT_PORT", 6333))
)
collection = os.getenv("QDRANT_COLLECTION_NAME")

# Search by title keyword
records, _ = client.scroll(
    collection_name=collection,
    scroll_filter=Filter(
        must=[FieldCondition(key="title", match=MatchText(text="EXFO"))]
    ),
    limit=20,
    with_payload=True,
    with_vectors=False,
)
print(f"Chunks found by title keyword 'EXFO': {len(records)}")
for r in records[:5]:
    if r.payload:
        print(f"  document_id={r.payload.get('document_id')!r}")
        print(f"  title={r.payload.get('title')!r}")
        print()

# Also try REACH
records2, _ = client.scroll(
    collection_name=collection,
    scroll_filter=Filter(
        must=[FieldCondition(key="title", match=MatchText(text="REACH"))]
    ),
    limit=20,
    with_payload=True,
    with_vectors=False,
)
print(f"Chunks found by title keyword 'REACH': {len(records2)}")
for r in records2[:5]:
    if r.payload:
        print(f"  document_id={r.payload.get('document_id')!r}")
        print(f"  title={r.payload.get('title')!r}")
        print()
