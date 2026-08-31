import pytest
from hitl.batch_queue import BatchReviewQueue


@pytest.fixture
def queue():
    return BatchReviewQueue()


@pytest.mark.asyncio
async def test_add_and_retrieve_pending(queue):
    await queue.add("doc1", {"text": "hello"}, 0.9)
    await queue.add("doc2", {"text": "world"}, 0.5)
    pending = await queue.get_pending()
    assert len(pending) == 2
    ids = {item["document_id"] for item in pending}
    assert ids == {"doc1", "doc2"}


@pytest.mark.asyncio
async def test_approve_removes_from_pending(queue):
    await queue.add("doc1", {"text": "hello"}, 0.9)
    await queue.approve("doc1", {"ok": True})
    pending = await queue.get_pending()
    assert len(pending) == 0


@pytest.mark.asyncio
async def test_reject_item(queue):
    await queue.add("doc1", {"text": "hello"}, 0.9)
    await queue.reject("doc1", {"reason": "bad extraction"})
    pending = await queue.get_pending()
    assert len(pending) == 0
    item = queue._queue["doc1"]
    assert item.status == "rejected"
    assert item.human_feedback["reason"] == "bad extraction"


@pytest.mark.asyncio
async def test_empty_queue_returns_empty_list(queue):
    pending = await queue.get_pending()
    assert pending == []


@pytest.mark.asyncio
async def test_add_multiple_items(queue):
    for i in range(5):
        await queue.add(f"doc{i}", {"page": i}, 0.8)
    pending = await queue.get_pending()
    assert len(pending) == 5
