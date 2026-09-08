"""SkyRL ABC surface, vendored so tests run when skyrl_train is not installed.

Signatures match adityasoni9998/SkyRL @ 81e5a97c skyrl-train inference_engines/base.py.
Prefer the installed class when present (see inference_engine.py).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Hashable, List, NotRequired, Optional, TypedDict

MessageType = Dict[str, str]
ConversationType = List[MessageType]


class InferenceEngineInput(TypedDict):
    prompts: Optional[List[ConversationType]]
    prompt_token_ids: Optional[List[List[int]]]
    sampling_params: Optional[Dict[str, Any]]
    session_ids: Optional[List[Hashable]]


class InferenceEngineOutput(TypedDict):
    responses: List[str]
    response_ids: List[List[int]]
    stop_reasons: List[str]
    response_logprobs: Optional[List[List[float]]]


class NamedWeightsUpdateRequest(TypedDict):
    names: List[str]
    dtypes: List[str]
    shapes: List[List[int]]
    sizes: NotRequired[List[int]]
    extras: Optional[List[Dict[str, Any]]]
    packed: NotRequired[bool]


class InferenceEngineInterface(ABC):
    @abstractmethod
    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        raise NotImplementedError()

    @abstractmethod
    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError()

    @abstractmethod
    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError()

    @abstractmethod
    async def wake_up(self, *args: Any, **kwargs: Any):
        raise NotImplementedError()

    @abstractmethod
    async def sleep(self, *args: Any, **kwargs: Any):
        raise NotImplementedError()

    @abstractmethod
    async def init_weight_update_communicator(
        self, master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing: bool = False
    ):
        raise NotImplementedError()

    @abstractmethod
    async def update_named_weights(self, request: NamedWeightsUpdateRequest):
        raise NotImplementedError()

    @abstractmethod
    async def teardown(self):
        raise NotImplementedError()

    @abstractmethod
    async def reset_prefix_cache(self):
        raise NotImplementedError()

    @abstractmethod
    def tp_size(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def pp_size(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def dp_size(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    async def abort_generation(self) -> None:
        raise NotImplementedError()
