import numpy as np
import pytest
from PIL import Image

from src.train import StreamingFaceBinaryDataset


class _RetryableFakeStream:
    def __init__(self, items, fail_before_index=None):
        self.items = items
        self.fail_before_index = fail_before_index

    def __iter__(self):
        for index, item in enumerate(self.items):
            if self.fail_before_index == index:
                raise ConnectionError("temporary connection loss")
            yield item

    def skip(self, count):
        return _RetryableFakeStream(self.items[count:])


def test_streaming_dataset_retries_and_resumes_after_network_error(monkeypatch, capsys):
    items = [
        {"jpg": Image.new("RGB", (1, 1), color=(10, 20, 30)), "json": {"label": 0, "source": "a"}},
        {"jpg": Image.new("RGB", (1, 1), color=(40, 50, 60)), "json": {"label": 1, "source": "b"}},
    ]
    dataset = object.__new__(StreamingFaceBinaryDataset)
    dataset._stream = _RetryableFakeStream(items, fail_before_index=1)
    dataset.STREAM_RETRY_ATTEMPTS = 1
    dataset.STREAM_RETRY_INITIAL_BACKOFF_SECONDS = 0
    dataset._build_stream = lambda: _RetryableFakeStream(items)
    monkeypatch.setattr("src.train.time.sleep", lambda _: None)

    decoded = list(dataset.iter_decoded_samples())

    assert [(label, source) for _, label, source in decoded] == [(0, "a"), (1, "b")]
    assert np.array_equal(decoded[0][0], np.array([[[10, 20, 30]]], dtype=np.uint8))
    assert np.array_equal(decoded[1][0], np.array([[[40, 50, 60]]], dtype=np.uint8))
    assert "Retrying in 0s [1/1]" in capsys.readouterr().err


def test_streaming_dataset_reraises_after_retry_limit(monkeypatch):
    dataset = object.__new__(StreamingFaceBinaryDataset)
    failed_stream = _RetryableFakeStream([{}], fail_before_index=0)
    dataset._stream = failed_stream
    dataset.STREAM_RETRY_ATTEMPTS = 1
    dataset.STREAM_RETRY_INITIAL_BACKOFF_SECONDS = 0
    dataset._build_stream = lambda: failed_stream
    monkeypatch.setattr("src.train.time.sleep", lambda _: None)

    with pytest.raises(ConnectionError, match="temporary connection loss"):
        list(dataset.iter_decoded_samples())
