import asyncio
import sounddevice as sd
import numpy as np

from typing import AsyncIterator
from abc import ABC, abstractmethod


class AudioSource(ABC):
    @abstractmethod
    async def read(self) -> bytes:
        pass


class MicrophoneSource(AudioSource):
    def __init__(self, sample_rate: int = 24000, frame_ms: int = 20):
        try:
            import sounddevice as sd  # type: ignore
        except Exception as e:
            raise RuntimeError("sounddevice is required for MicSource. pip install sounddevice") from e
        self.sd = sd
        self.sample_rate = sample_rate
        self.frame_samples = int(sample_rate * frame_ms / 1000)

    async def read(self) -> AsyncIterator[bytes]:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=32)

        def _callback(indata, frames, time, status):
            loop.call_soon_threadsafe(q.put_nowait, bytes(indata))
            
        with self.sd.InputStream(
            samplerate=self.sample_rate, channels=1, dtype="int16",
            blocksize=self.frame_samples, callback=_callback
        ):
            while True:
                chunk = await q.get()
                yield chunk

    async def close(self):
        self.sd.stop()