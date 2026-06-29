from types import SimpleNamespace

from qdrant_client.models import Filter, MatchAny

from src.chat.retriever import QdrantRetriever


class _DummyRecord:
    def __init__(self, payload: dict):
        self.payload = payload


def _build_retriever(records: list[_DummyRecord]):
    retriever = object.__new__(QdrantRetriever)
    retriever.collection_name = "test_collection"
    retriever.client = SimpleNamespace(scroll=object())
    retriever._classify_source_kind = lambda payload: "unknown"

    captured_scroll_filters: list[Filter | None] = []

    def fake_with_retries(*_args, **kwargs):
        captured_scroll_filters.append(kwargs.get("scroll_filter"))
        return records, None

    retriever._with_retries = fake_with_retries
    return retriever, captured_scroll_filters


def test_retrieve_by_document_uses_all_known_legislation_id_fields():
    legislation_id = "d32a664d-1e27-4a79-8e89-bdef2311d6f5"
    retriever, captured_filters = _build_retriever(
        [_DummyRecord(payload={"text": "chunk", "chunk_id": "c1"})],
    )

    results, diagnostics = retriever._retrieve_by_document_impl(
        document_id=None,
        legislation_id=legislation_id,
        domain=None,
        title=None,
    )

    assert len(results) == 1
    assert diagnostics["retrieved_k"] == 1

    applied_filter = captured_filters[0]
    assert applied_filter is not None
    assert applied_filter.should is not None

    condition_keys = {condition.key for condition in applied_filter.should}
    assert condition_keys == {
        "legislation_id",
        "document_id",
        "document_metadata.id",
        "document_metadata.legislation_id",
    }

    for condition in applied_filter.should:
        assert condition.match.value == legislation_id


def test_retrieve_by_document_document_id_path_remains_direct_match():
    retriever, captured_filters = _build_retriever(
        [_DummyRecord(payload={"text": "chunk", "chunk_id": "c1"})],
    )

    retriever._retrieve_by_document_impl(
        document_id="42",
        legislation_id=None,
        domain=None,
        title=None,
    )

    applied_filter = captured_filters[0]
    assert applied_filter is not None
    assert applied_filter.must is not None
    assert len(applied_filter.must) == 1
    assert applied_filter.must[0].key == "document_id"
    assert applied_filter.must[0].match.value == "42"


def test_retrieve_by_document_applies_domain_as_must_condition():
    legislation_id = "d32a664d-1e27-4a79-8e89-bdef2311d6f5"
    retriever, captured_filters = _build_retriever(
        [_DummyRecord(payload={"text": "chunk", "chunk_id": "c1"})],
    )

    retriever._retrieve_by_document_impl(
        document_id=None,
        legislation_id=legislation_id,
        domain="eu",
        title=None,
    )

    applied_filter = captured_filters[0]
    assert applied_filter is not None
    assert applied_filter.must is not None
    assert len(applied_filter.must) == 1
    assert applied_filter.must[0].key == "domain"
    assert applied_filter.must[0].match.value == "eu"


def test_retrieve_by_document_sorts_mixed_chunk_id_types_without_crashing():
    retriever, _ = _build_retriever(
        [
            _DummyRecord(payload={"text": "a", "chunk_id": "10"}),
            _DummyRecord(payload={"text": "b", "chunk_id": 2}),
            _DummyRecord(payload={"text": "c", "chunk_id": "section-1"}),
            _DummyRecord(payload={"text": "d", "chunk_id": None}),
        ],
    )

    results, diagnostics = retriever._retrieve_by_document_impl(
        document_id="42",
        legislation_id=None,
        domain=None,
        title=None,
    )

    assert diagnostics["retrieved_k"] == 4
    assert [item["chunk_id"] for item in results] == [2, "10", "section-1", None]


def test_retrieve_by_document_multi_ids_uses_match_any_across_known_fields():
    ids = [
        "d32a664d-1e27-4a79-8e89-bdef2311d6f5",
        "a11b664d-2e27-4a79-8e89-bdef2311d999",
    ]
    retriever, captured_filters = _build_retriever(
        [_DummyRecord(payload={"text": "chunk", "chunk_id": "c1", "title": "Doc A"})],
    )

    results, diagnostics = retriever._retrieve_by_document_impl(
        document_id=None,
        legislation_id=None,
        domain=None,
        title=None,
        legislation_ids=ids,
        titles=None,
    )

    assert diagnostics["retrieved_k"] == 1
    applied_filter = captured_filters[0]
    assert applied_filter is not None
    assert applied_filter.should is not None
    condition_keys = {condition.key for condition in applied_filter.should}
    assert condition_keys == {
        "legislation_id",
        "document_id",
        "document_metadata.id",
        "document_metadata.legislation_id",
    }
    for condition in applied_filter.should:
        assert isinstance(condition.match, MatchAny)
        assert condition.match.any == ids
