import torch

from qwen3_runtime.attention.paged import flashinfer_page_tensors


def test_flashinfer_page_tensors_full_and_partial_last_page():
    tables = [
        torch.tensor([3, 7], dtype=torch.long),
        torch.tensor([1], dtype=torch.long),
    ]
    kv_lens = [32, 5]
    indptr, indices, last = flashinfer_page_tensors(tables, kv_lens, page_size=16, device=torch.device("cpu"))
    assert indptr.tolist() == [0, 2, 3]
    assert indices.tolist() == [3, 7, 1]
    # 32 is two full pages; 5 occupies 5 slots of the last page.
    assert last.tolist() == [16, 5]


def test_flashinfer_page_tensors_accepts_python_lists():
    indptr, indices, last = flashinfer_page_tensors([[4, 5, 6]], [48], page_size=16, device=torch.device("cpu"))
    assert indptr.tolist() == [0, 3]
    assert indices.tolist() == [4, 5, 6]
    assert last.tolist() == [16]
    assert indptr.device.type == "cpu"
    assert indices.device.type == "cpu"
    assert last.device.type == "cpu"


def test_flashinfer_page_tensors_stay_on_cpu_when_device_is_cuda_string():
    # plan() D2H-syncs if metadata is already on GPU. Helper must not place it there.
    indptr, indices, last = flashinfer_page_tensors(
        [[1, 2]], [32], page_size=16, device=torch.device("cuda")
    )
    assert indptr.device.type == "cpu"
    assert indices.tolist() == [1, 2]
    assert last.tolist() == [16]
