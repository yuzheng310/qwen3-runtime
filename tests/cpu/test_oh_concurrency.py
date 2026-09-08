"""In-flight cap for CodeScout OpenHands clones (AutoDL cgroup ~92 GiB)."""

import asyncio

from qwen3_runtime.integrations.skyrl.inject import wrap_async_with_semaphore


def test_semaphore_caps_inflight():
    inflight = 0
    peak = 0

    async def job():
        nonlocal inflight, peak
        inflight += 1
        peak = max(peak, inflight)
        await asyncio.sleep(0.05)
        inflight -= 1
        return 1

    wrapped = wrap_async_with_semaphore(job, 2)

    async def run():
        return await asyncio.gather(*[wrapped() for _ in range(6)])

    assert asyncio.run(run()) == [1] * 6
    assert peak <= 2
    assert peak >= 1
