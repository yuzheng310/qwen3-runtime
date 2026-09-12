"""Preserve SkyRL's wrapper and add our trajectory-finish control method."""

from skyrl_train.inference_engines.ray_wrapped_inference_engine import RayWrappedInferenceEngine


class Qwen3RayWrappedInferenceEngine(RayWrappedInferenceEngine):
    async def finish_session(self, token_ids: list[int]) -> int:
        return await self.inference_engine_actor.finish_session.remote(token_ids)
