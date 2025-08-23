import unittest
import uuid
from unittest.mock import MagicMock

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointIdsList, PointStruct, VectorParams

from mem0.vector_stores.qdrant import Qdrant


class TestQdrant(unittest.TestCase):
    def setUp(self):
        self.client_mock = MagicMock(spec=QdrantClient)
        self.qdrant = Qdrant(
            collection_name="test_collection",
            embedding_model_dims=128,
            client=self.client_mock,
            path="test_path",
            on_disk=True,
        )

    def test_create_col(self):
        self.client_mock.get_collections.return_value = MagicMock(collections=[])

        self.qdrant.create_col(vector_size=128, on_disk=True)

        expected_config = VectorParams(size=128, distance=Distance.COSINE, on_disk=True)

        self.client_mock.create_collection.assert_called_with(
            collection_name="test_collection", vectors_config=expected_config
        )

    def test_insert(self):
        vectors = [[0.1, 0.2], [0.3, 0.4]]
        payloads = [{"key": "value1"}, {"key": "value2"}]
        ids = [str(uuid.uuid4()), str(uuid.uuid4())]

        self.qdrant.insert(vectors=vectors, payloads=payloads, ids=ids)

        self.client_mock.upsert.assert_called_once()
        points = self.client_mock.upsert.call_args[1]["points"]

        self.assertEqual(len(points), 2)
        for point in points:
            self.assertIsInstance(point, PointStruct)

        self.assertEqual(points[0].payload, payloads[0])

    def test_search(self):
        vectors = [[0.1, 0.2]]
        mock_point = MagicMock(id=str(uuid.uuid4()), score=0.95, payload={"key": "value"})
        self.client_mock.query_points.return_value = MagicMock(points=[mock_point])

        results = self.qdrant.search(query="", vectors=vectors, limit=1)

        self.client_mock.query_points.assert_called_once_with(
            collection_name="test_collection",
            query=vectors,
            query_filter=None,
            limit=1,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].payload, {"key": "value"})
        self.assertEqual(results[0].score, 0.95)

    def test_delete(self):
        vector_id = str(uuid.uuid4())
        self.qdrant.delete(vector_id=vector_id)

        self.client_mock.delete.assert_called_once_with(
            collection_name="test_collection",
            points_selector=PointIdsList(points=[vector_id]),
        )

    def test_update(self):
        vector_id = str(uuid.uuid4())
        updated_vector = [0.2, 0.3]
        updated_payload = {"key": "updated_value"}

        self.qdrant.update(vector_id=vector_id, vector=updated_vector, payload=updated_payload)

        self.client_mock.upsert.assert_called_once()
        point = self.client_mock.upsert.call_args[1]["points"][0]
        self.assertEqual(point.id, vector_id)
        self.assertEqual(point.vector, updated_vector)
        self.assertEqual(point.payload, updated_payload)

    def test_get(self):
        vector_id = str(uuid.uuid4())
        self.client_mock.retrieve.return_value = [{"id": vector_id, "payload": {"key": "value"}}]

        result = self.qdrant.get(vector_id=vector_id)

        self.client_mock.retrieve.assert_called_once_with(
            collection_name="test_collection", ids=[vector_id], with_payload=True
        )
        self.assertEqual(result["id"], vector_id)
        self.assertEqual(result["payload"], {"key": "value"})

    def test_list_cols(self):
        self.client_mock.get_collections.return_value = MagicMock(collections=[{"name": "test_collection"}])
        result = self.qdrant.list_cols()
        self.assertEqual(result.collections[0]["name"], "test_collection")

    def test_delete_col(self):
        self.qdrant.delete_col()
        self.client_mock.delete_collection.assert_called_once_with(collection_name="test_collection")

    def test_col_info(self):
        self.qdrant.col_info()
        self.client_mock.get_collection.assert_called_once_with(collection_name="test_collection")

    def test_count_memories_no_filter(self):
        """Test count_memories without filters returns total count"""
        from unittest.mock import MagicMock
        
        # Mock the count method to return a count result
        count_result = MagicMock()
        count_result.count = 42
        self.client_mock.count.return_value = count_result
        
        result = self.qdrant.count_memories()
        
        # Should call count with no filter
        self.client_mock.count.assert_called_once_with(
            collection_name="test_collection",
            count_filter=None,
            exact=True
        )
        self.assertEqual(result, 42)

    def test_count_memories_with_user_filter(self):
        """Test count_memories with user_id filter"""
        from unittest.mock import MagicMock
        
        count_result = MagicMock()
        count_result.count = 15
        self.client_mock.count.return_value = count_result
        
        # Test with user_id filter
        filters = {"user_id": "test_user_123"}
        result = self.qdrant.count_memories(filters)
        
        # Should call count with proper filter
        self.client_mock.count.assert_called_once()
        call_args = self.client_mock.count.call_args
        
        self.assertEqual(call_args[1]["collection_name"], "test_collection")
        self.assertEqual(call_args[1]["exact"], True)
        self.assertIsNotNone(call_args[1]["count_filter"])
        self.assertEqual(result, 15)

    def test_count_memories_with_multiple_filters(self):
        """Test count_memories with multiple filters"""
        from unittest.mock import MagicMock
        
        count_result = MagicMock()
        count_result.count = 5
        self.client_mock.count.return_value = count_result
        
        # Test with multiple filters
        filters = {
            "user_id": "test_user_123",
            "source": "chat",
            "created_at": {"gte": "2024-01-01", "lte": "2024-12-31"}
        }
        result = self.qdrant.count_memories(filters)
        
        # Should call count with combined filter
        self.client_mock.count.assert_called_once()
        self.assertEqual(result, 5)

    def test_count_memories_fallback_on_attribute_error(self):
        """Test count_memories falls back to scroll when count method is not available"""
        from unittest.mock import MagicMock
        
        # Mock count method to raise AttributeError (older client version)
        self.client_mock.count.side_effect = AttributeError("count method not available")
        
        # Mock scroll to return points for counting
        mock_points = [MagicMock() for _ in range(8)]  # 8 mock points
        self.client_mock.scroll.return_value = (mock_points, None)
        
        filters = {"user_id": "test_user_123"}
        result = self.qdrant.count_memories(filters)
        
        # Should fall back to scroll and count the points
        self.client_mock.scroll.assert_called_once_with(
            collection_name="test_collection",
            scroll_filter=unittest.mock.ANY,  # Filter object
            limit=10000,
            with_payload=False,
            with_vectors=False,
        )
        self.assertEqual(result, 8)

    def test_count_memories_handles_general_exception(self):
        """Test count_memories handles general exceptions gracefully"""
        # Mock count to raise a general exception
        self.client_mock.count.side_effect = Exception("Connection error")
        
        result = self.qdrant.count_memories()
        
        # Should return 0 on error
        self.assertEqual(result, 0)

    def test_count_memories_empty_result(self):
        """Test count_memories handles empty results correctly"""
        from unittest.mock import MagicMock
        
        count_result = MagicMock()
        count_result.count = 0
        self.client_mock.count.return_value = count_result
        
        result = self.qdrant.count_memories({"user_id": "nonexistent_user"})
        
        self.assertEqual(result, 0)

    def test_count_memories_scroll_fallback_empty_points(self):
        """Test scroll fallback handles empty points correctly"""
        # Mock count method to raise AttributeError
        self.client_mock.count.side_effect = AttributeError("count method not available")
        
        # Mock scroll to return empty points list
        self.client_mock.scroll.return_value = ([], None)
        
        result = self.qdrant.count_memories({"user_id": "test_user"})
        
        self.assertEqual(result, 0)

    def test_count_memories_scroll_fallback_exception(self):
        """Test scroll fallback handles exceptions gracefully"""
        # Mock count method to raise AttributeError
        self.client_mock.count.side_effect = AttributeError("count method not available")
        
        # Mock scroll to also raise an exception
        self.client_mock.scroll.side_effect = Exception("Scroll error")
        
        result = self.qdrant.count_memories({"user_id": "test_user"})
        
        self.assertEqual(result, 0)

    def tearDown(self):
        del self.qdrant
