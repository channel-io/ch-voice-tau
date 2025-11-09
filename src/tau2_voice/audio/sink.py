import asyncio
import sounddevice as sd
import numpy as np

from abc import ABC, abstractmethod


class AudioSink(ABC):
    @abstractmethod
    async def write(self, pcm_bytes: bytes) -> None:
        pass

    @abstractmethod
    async def close(self) -> None:
        pass


class SpeakerSink(AudioSink):
    def __init__(self, sample_rate: int = 24000, frame_ms: int = 20):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self._queue = bytearray()
        self._lock = asyncio.Lock()
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='int16',
            blocksize=self.frame_ms * self.sample_rate // 1000,
            callback=self._callback,
        )
        self._stream.start()

    async def write(self, pcm_bytes: bytes) -> None:
        # Avoid holding the lock longer than needed
        await self._lock.acquire()
        try:
            self._queue.extend(pcm_bytes)
        finally:
            self._lock.release()

    async def close(self) -> None:
        await self._lock.acquire()
        try:
            self._queue.clear()
        finally:
            self._lock.release()
        self._stream.stop()
        self._stream.close()

    def _callback(
        self, 
        outdata: np.ndarray, 
        frames: int, 
        time,
        status,
    ) -> None:
        need = frames * 2
        # Non-blocking check of the queue to avoid deadlocks in callback thread
        available = len(self._queue)
        if available < need:
            outdata.fill(0)
            return
        # Slice without lock; reading and slicing bytearray is atomic enough for audio
        chunk = self._queue[:need]
        del self._queue[:need]
        outdata[:] = np.frombuffer(chunk, dtype=np.int16).reshape(-1, 1)