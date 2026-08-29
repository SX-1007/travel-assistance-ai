from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import pytest
from pinecone.exceptions import PineconeException

from app.core import firebase_db, pinecone_db
from app.memory import extractor


def _firestore_document(*, exists: bool = True, data=None):
    document = Mock()
    snapshot = Mock(exists=exists)
    snapshot.to_dict.return_value = data
    document.get.return_value = snapshot
    database = Mock()
    database.collection.return_value.document.return_value = document
    return database, document


@pytest.mark.unit
class TestFirestoreCacheBoundaries:
    def test_retry_decorator_recovers_after_transient_failures(self):
        operation = Mock(side_effect=[RuntimeError("one"), RuntimeError("two"), "ok"])
        operation.__name__ = "operation"
        wrapped = firebase_db._with_retry(max_retries=2, base_delay=0)(operation)
        with (
            patch.object(firebase_db.random, "uniform", return_value=0),
            patch.object(firebase_db.time, "sleep") as sleep,
        ):
            assert wrapped() == "ok"
        assert operation.call_count == 3
        assert sleep.call_count == 2

    def test_retry_decorator_reraises_the_final_error(self):
        operation = Mock(side_effect=ValueError("permanent"))
        operation.__name__ = "operation"
        wrapped = firebase_db._with_retry(max_retries=1, base_delay=0)(operation)
        with patch.object(firebase_db.time, "sleep"):
            with pytest.raises(ValueError, match="permanent"):
                wrapped()
        assert operation.call_count == 2

    def test_cache_miss_returns_none(self):
        database, _ = _firestore_document(exists=False)
        with patch.object(firebase_db, "_get_db", return_value=database):
            assert firebase_db.get_cached_data("cache", "hotel", city="Tokyo") is None

    def test_cache_read_error_is_a_safe_miss(self):
        database, document = _firestore_document()
        document.get.side_effect = RuntimeError("offline")
        with patch.object(firebase_db, "_get_db", return_value=database):
            assert firebase_db.get_cached_data("cache", "hotel") is None

    @pytest.mark.parametrize(
        "expires_at", [None, datetime.now(timezone.utc) - timedelta(seconds=1)]
    )
    def test_expired_or_unversioned_cache_entries_are_deleted(self, expires_at):
        database, document = _firestore_document(
            data={"payload": {"x": 1}, "expires_at": expires_at}
        )
        with patch.object(firebase_db, "_get_db", return_value=database):
            assert firebase_db.get_cached_data("cache", "hotel") is None
        document.delete.assert_called_once_with()

    def test_valid_cache_hit_returns_payload(self):
        payload = {"price": 123}
        database, document = _firestore_document(
            data={
                "payload": payload,
                "expires_at": datetime.now(timezone.utc) + timedelta(hours=1),
            }
        )
        with patch.object(firebase_db, "_get_db", return_value=database):
            assert firebase_db.get_cached_data("cache", "hotel") == payload
        document.delete.assert_not_called()

    @pytest.mark.parametrize("ttl", [0, -1])
    def test_cache_write_rejects_nonpositive_ttl(self, ttl):
        with patch.object(firebase_db, "_get_db") as get_db:
            assert (
                firebase_db.set_cached_data("cache", "hotel", {}, ttl_hours=ttl)
                is False
            )
        get_db.assert_not_called()

    def test_cache_write_rejects_oversized_payload_before_database_access(self):
        with (
            patch.object(firebase_db, "FIRESTORE_DOC_MAX_BYTES", 1),
            patch.object(firebase_db, "_get_db") as get_db,
        ):
            assert (
                firebase_db.set_cached_data("cache", "hotel", {"large": "payload"})
                is False
            )
        get_db.assert_not_called()

    def test_cache_write_sets_payload_and_timestamps(self):
        database, document = _firestore_document()
        with patch.object(firebase_db, "_get_db", return_value=database):
            assert firebase_db.set_cached_data(
                "cache", "hotel", {"price": 20}, ttl_hours=2
            )
        written = document.set.call_args.args[0]
        assert written["payload"] == {"price": 20}
        assert written["expires_at"] - written["created_at"] == timedelta(hours=2)

    def test_cache_write_and_invalidation_fail_closed(self):
        database, document = _firestore_document()
        document.set.side_effect = RuntimeError("write failed")
        with patch.object(firebase_db, "_get_db", return_value=database):
            assert firebase_db.set_cached_data("cache", "hotel", {}) is False

        document.delete.side_effect = RuntimeError("delete failed")
        with patch.object(firebase_db, "_get_db", return_value=database):
            assert firebase_db.invalidate_cached_data("cache", "hotel") is False

    def test_cache_invalidation_success(self):
        database, document = _firestore_document()
        with patch.object(firebase_db, "_get_db", return_value=database):
            assert firebase_db.invalidate_cached_data("cache", "hotel", city="Tokyo")
        document.delete.assert_called_once_with()

    def test_bulk_clear_deletes_and_commits_a_page(self):
        database = Mock()
        documents = [Mock(reference="a"), Mock(reference="b")]
        database.collection.return_value.limit.return_value.stream.return_value = (
            documents
        )
        with patch.object(firebase_db, "_get_db", return_value=database):
            assert firebase_db.clear_trip_cache("cache") == 2
        assert database.batch.return_value.delete.call_count == 2
        database.batch.return_value.commit.assert_called_once_with()

    def test_bulk_clear_returns_progress_count_on_provider_error(self):
        database = Mock()
        database.collection.return_value.limit.return_value.stream.side_effect = (
            RuntimeError("offline")
        )
        with patch.object(firebase_db, "_get_db", return_value=database):
            assert firebase_db.clear_trip_cache("cache") == 0


@pytest.mark.unit
class TestPineconeStorageBoundaries:
    def test_client_is_lazy_and_cached(self):
        client = Mock()
        with (
            patch.object(pinecone_db, "_pc", None),
            patch.object(pinecone_db, "Pinecone", return_value=client) as constructor,
        ):
            assert pinecone_db._get_pinecone() is client
            assert pinecone_db._get_pinecone() is client
        constructor.assert_called_once()

    def test_existing_index_is_connected_without_creation(self):
        client = Mock()
        client.list_indexes.return_value = [{"name": pinecone_db.INDEX_NAME}]
        expected = Mock()
        client.Index.return_value = expected
        with (
            patch.object(pinecone_db, "_index", None),
            patch.object(pinecone_db, "_get_pinecone", return_value=client),
        ):
            assert pinecone_db._get_or_create_index() is expected
        client.create_index.assert_not_called()

    def test_missing_index_is_created_and_waited_until_ready(self):
        client = Mock()
        client.list_indexes.return_value = []
        client.describe_index.return_value = Mock(status=Mock(ready=True))
        expected = Mock()
        client.Index.return_value = expected
        with (
            patch.object(pinecone_db, "_index", None),
            patch.object(pinecone_db, "_get_pinecone", return_value=client),
        ):
            assert pinecone_db._get_or_create_index() is expected
        client.create_index.assert_called_once()

    def test_embedding_empty_input_avoids_provider(self):
        with patch.object(pinecone_db, "_get_pinecone") as get_client:
            assert pinecone_db._embed_texts([]) == []
        get_client.assert_not_called()

    def test_embedding_batches_and_flattens_provider_vectors(self):
        client = Mock()

        def embed(**kwargs):
            return [
                Mock(values=[float(index)]) for index, _ in enumerate(kwargs["inputs"])
            ]

        client.inference.embed.side_effect = embed
        with (
            patch.object(pinecone_db, "_get_pinecone", return_value=client),
            patch.object(pinecone_db, "EMBED_BATCH", 2),
        ):
            result = pinecone_db._embed_texts(["a", "b", "c"])
        assert result == [[0.0], [1.0], [0.0]]
        assert client.inference.embed.call_count == 2

    def test_embedding_retries_transient_pinecone_errors(self):
        client = Mock()
        client.inference.embed.side_effect = [
            PineconeException("temporary"),
            [Mock(values=[1.0, 2.0])],
        ]
        with (
            patch.object(pinecone_db, "_get_pinecone", return_value=client),
            patch.object(pinecone_db.time, "sleep") as sleep,
        ):
            assert pinecone_db._embed_texts(["query"]) == [[1.0, 2.0]]
        sleep.assert_called_once()

    @pytest.mark.parametrize(
        ("category", "item", "expected_text"),
        [
            ("hotel", {"hotel_name": "Inn", "amenities": ["WiFi"]}, "Hotel: Inn"),
            ("activity", {"name": "Museum", "type": "culture"}, "Activity: Museum"),
            ("flight", {"flight_number": "TA1", "airline": "Test"}, "Flight TA1"),
            ("other", {"name": "Custom"}, '"name": "Custom"'),
        ],
    )
    def test_upsert_builds_category_payload_metadata_and_namespace(
        self, category, item, expected_text
    ):
        index = Mock()
        captured_texts = []

        def embed(texts, input_type="passage"):
            captured_texts.extend(texts)
            return [[0.1, 0.2] for _ in texts]

        with (
            patch.object(pinecone_db, "_get_or_create_index", return_value=index),
            patch.object(pinecone_db, "_embed_texts", side_effect=embed),
        ):
            assert pinecone_db.upsert_travel_data("session", category, [item])

        assert expected_text in captured_texts[0]
        record = index.upsert.call_args.kwargs["vectors"][0]
        assert json.loads(record["metadata"]["raw_data"]) == item
        assert record["metadata"]["category"] == category
        assert index.upsert.call_args.kwargs["namespace"] == "session"

    def test_upsert_rejects_missing_index_empty_items_and_embedding_mismatch(self):
        with patch.object(pinecone_db, "_get_or_create_index", return_value=None):
            assert pinecone_db.upsert_travel_data("s", "hotel", [{}]) is False
        with patch.object(pinecone_db, "_get_or_create_index", return_value=Mock()):
            assert pinecone_db.upsert_travel_data("s", "hotel", []) is False
        with (
            patch.object(pinecone_db, "_get_or_create_index", return_value=Mock()),
            patch.object(pinecone_db, "_embed_texts", return_value=[]),
        ):
            assert (
                pinecone_db.upsert_travel_data("s", "hotel", [{"hotel_name": "Inn"}])
                is False
            )

    def test_personalised_search_reconstructs_ranked_items_and_skips_bad_json(self):
        index = Mock()
        index.query.return_value = {
            "matches": [
                {"score": 0.9876, "metadata": {"raw_data": '{"name":"Museum"}'}},
                {"score": 0.5, "metadata": {"raw_data": "not-json"}},
            ]
        }
        with (
            patch.object(pinecone_db, "_get_or_create_index", return_value=index),
            patch.object(pinecone_db, "_embed_texts", return_value=[[0.1]]),
        ):
            result = pinecone_db.search_personalised_options(
                "session", "culture", "activity", 3
            )
        assert result == [{"name": "Museum", "relevance_score": 0.988}]
        assert index.query.call_args.kwargs["namespace"] == "session"
        assert index.query.call_args.kwargs["top_k"] == 3

    @pytest.mark.parametrize("preference", ["", "   "])
    def test_personalised_search_rejects_blank_preferences(self, preference):
        with patch.object(pinecone_db, "_get_or_create_index", return_value=Mock()):
            assert (
                pinecone_db.search_personalised_options("s", preference, "hotel") == []
            )

    def test_session_clear_success_and_failure(self):
        index = Mock()
        with patch.object(pinecone_db, "_get_or_create_index", return_value=index):
            assert pinecone_db.clear_session_vectors("session")
        index.delete.assert_called_once_with(delete_all=True, namespace="session")

        index.delete.side_effect = RuntimeError("offline")
        with patch.object(pinecone_db, "_get_or_create_index", return_value=index):
            assert pinecone_db.clear_session_vectors("session") is False


@pytest.mark.unit
class TestMemoryExtractionBoundaries:
    def test_relational_and_vector_wrappers_report_provider_results(self):
        with patch.object(extractor, "update_user_profile") as update:
            assert extractor._update_relational_db("user", {"home_country": "Malaysia"})
        update.assert_called_once()

        with patch.object(
            extractor, "insert_mem0_preferences", return_value=True
        ) as insert:
            assert extractor._update_vector_db("user", {"interest": ["food"]})
        insert.assert_called_once()

    def test_storage_wrappers_fail_closed_on_exceptions(self):
        with patch.object(
            extractor, "update_user_profile", side_effect=RuntimeError("db")
        ):
            assert extractor._update_relational_db("user", {}) is False
        with patch.object(
            extractor, "insert_mem0_preferences", side_effect=RuntimeError("db")
        ):
            assert extractor._update_vector_db("user", {}) is False

    @pytest.mark.parametrize("message", ["", "   ", "thanks"])
    def test_trait_extraction_skips_empty_and_trivial_messages(self, message):
        with patch.object(extractor, "_get_extraction_chain") as chain:
            assert extractor.analyse_and_extract_traits(message) is None
        chain.assert_not_called()

    def test_trait_extraction_truncates_long_messages_and_returns_structured_result(
        self,
    ):
        expected = extractor.LongTermMemory(interests=["Museums"])
        chain = Mock()
        chain.invoke.return_value = expected
        with patch.object(extractor, "_get_extraction_chain", return_value=chain):
            assert extractor.analyse_and_extract_traits("x" * 10_000) is expected
        assert (
            len(chain.invoke.call_args.args[0]["message"])
            == extractor._MAX_MESSAGE_LENGTH
        )

    def test_trait_extraction_retries_then_returns_none(self):
        chain = Mock()
        chain.invoke.side_effect = RuntimeError("model unavailable")
        with (
            patch.object(extractor, "_get_extraction_chain", return_value=chain),
            patch.object(extractor, "_calculate_backoff", return_value=0),
            patch.object(extractor.time, "sleep") as sleep,
        ):
            assert extractor.analyse_and_extract_traits("I prefer museums") is None
        assert chain.invoke.call_count == extractor.MAX_LLM_RETRIES
        assert sleep.call_count == extractor.MAX_LLM_RETRIES - 1

    @pytest.mark.parametrize("user_id", ["", None])
    def test_memory_task_requires_user_identity(self, user_id):
        with patch.object(extractor, "analyse_and_extract_traits") as analyse:
            assert extractor.run_memory_extraction_task(
                "session", user_id, "message"
            ) == {"memory_updated": False}
        analyse.assert_not_called()

    def test_memory_task_skips_empty_extraction(self):
        with patch.object(extractor, "analyse_and_extract_traits", return_value=None):
            assert extractor.run_memory_extraction_task(
                "session", "user", "message"
            ) == {"memory_updated": False}

    def test_memory_task_routes_relational_and_vector_traits(self):
        memory = extractor.LongTermMemory(
            home_country="Malaysia",
            home_state="Selangor",
            travel_pacing="relaxed",
            dietary_restrictions=["vegan"],
            interests=["history"],
            accommodation_preferences=["boutique"],
        )
        with (
            patch.object(extractor, "analyse_and_extract_traits", return_value=memory),
            patch.object(
                extractor, "_update_relational_db", return_value=True
            ) as relational,
            patch.object(extractor, "_update_vector_db", return_value=True) as vector,
        ):
            assert extractor.run_memory_extraction_task(
                "session", "user", "message"
            ) == {"memory_updated": True}
        relational.assert_called_once_with(
            "user",
            {
                "home_country": "Malaysia",
                "home_state": "Selangor",
                "travel_pacing": "relaxed",
            },
        )
        vector.assert_called_once_with(
            "user",
            {
                "dietary": ["vegan"],
                "interest": ["history"],
                "accommodation": ["boutique"],
            },
        )

    def test_memory_task_with_no_populated_traits_does_not_write(self):
        with (
            patch.object(
                extractor,
                "analyse_and_extract_traits",
                return_value=extractor.LongTermMemory(),
            ),
            patch.object(extractor, "_update_relational_db") as relational,
            patch.object(extractor, "_update_vector_db") as vector,
        ):
            assert extractor.run_memory_extraction_task(
                "session", "user", "message"
            ) == {"memory_updated": False}
        relational.assert_not_called()
        vector.assert_not_called()
