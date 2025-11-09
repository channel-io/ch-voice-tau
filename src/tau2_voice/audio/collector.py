import base64
import json
import time
import wave
from pathlib import Path
from typing import Literal, Optional


Role = Literal["assistant", "user"]


class AudioCollector:
    """
    Collects audio chunks from assistant and user, writes a single WAV file of the
    conversation, and tracks per-turn timing and sample offsets.

    - Audio chunks are expected as base64-encoded PCM bytes
    - WAV is written as mono, 16-bit PCM at the configured sample rate
    - Turn boundaries are inferred from audio.done events (end_turn)
    - Timestamps recorded as absolute (ISO-like) and relative seconds since start
    """

    def __init__(
        self,
        output_wav_path: Path | str,
        sample_rate: int = 24000,
        channels: int = 1,
        sample_width_bytes: int = 2,
    ):
        self.output_wav_path = Path(output_wav_path)
        self.output_meta_path = self.output_wav_path.with_suffix(".json")
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width_bytes = sample_width_bytes

        self._wav: Optional[wave.Wave_write] = None
        self._samples_written: int = 0

        self._conversation_started_at_wall: Optional[float] = None
        self._current_turn: Optional[dict] = None
        self._turns: list[dict] = []
        self._transcripts: dict[str, str] = {}  # message_id -> transcript

        # Ensure directory exists
        self.output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    def _ensure_open(self):
        if self._wav is not None:
            return
        self._wav = wave.open(str(self.output_wav_path), "wb")
        self._wav.setnchannels(self.channels)
        self._wav.setsampwidth(self.sample_width_bytes)
        self._wav.setframerate(self.sample_rate)

    def _now_wall(self) -> float:
        return time.time()

    def _rel_seconds(self, wall: float) -> float:
        assert self._conversation_started_at_wall is not None
        return max(0.0, wall - self._conversation_started_at_wall)

    def handle_audio_chunk(self, role: Role, message_id: str, audio_chunk_b64: str):
        """
        Append an audio chunk to the WAV and start a turn if needed.
        """
        wall_now = self._now_wall()
        if self._conversation_started_at_wall is None:
            self._conversation_started_at_wall = wall_now

        self._ensure_open()

        # Start a new turn lazily on first chunk after previous turn ended
        if self._current_turn is None:
            self._current_turn = {
                "role": role,
                "message_id": message_id,
                "start_wall_time": wall_now,
                "start_rel_sec": self._rel_seconds(wall_now),
                "start_sample": self._samples_written,
            }

        # Decode and write PCM
        pcm = base64.b64decode(audio_chunk_b64)
        if self._wav is not None:
            self._wav.writeframes(pcm)
        # Update sample cursor (bytes / bytes_per_sample)
        bytes_per_sample = self.channels * self.sample_width_bytes
        self._samples_written += len(pcm) // bytes_per_sample

    def handle_audio_done(self, role: Role, message_id: str):
        """
        Finalize the current turn on audio.done.
        """
        if self._current_turn is None:
            return
        # Only close the turn if it matches the current speaker/message
        if (
            self._current_turn.get("role") != role
            or self._current_turn.get("message_id") != message_id
        ):
            return

        wall_now = self._now_wall()
        end_rel_sec = self._rel_seconds(wall_now)
        turn = {
            **self._current_turn,
            "end_wall_time": wall_now,
            "end_rel_sec": end_rel_sec,
            "end_sample": self._samples_written,
            "duration_sec": max(0.0, end_rel_sec - self._current_turn["start_rel_sec"]),
            "transcript": self._transcripts.get(message_id),  # Add transcript if available
        }
        self._turns.append(turn)
        self._current_turn = None

    def handle_transcript_update(self, role: Role, message_id: str, transcript: str):
        """
        Store transcript for the given message_id to be included in the turn.
        """
        self._transcripts[message_id] = transcript
        # If there's an already-closed turn with this message_id, update it
        for turn in self._turns:
            if turn.get("message_id") == message_id and turn.get("role") == role:
                turn["transcript"] = transcript
                break

    def finalize(self):
        """
        Close the WAV and write the metadata JSON describing the conversation turns.
        """
        # If there is an open turn, close it at the current moment
        if self._current_turn is not None:
            self.handle_audio_done(
                role=self._current_turn["role"],
                message_id=self._current_turn["message_id"],
            )

        if self._wav is not None:
            try:
                self._wav.close()
            finally:
                self._wav = None

        meta = {
            "wav_path": str(self.output_wav_path),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "sample_width_bytes": self.sample_width_bytes,
            "conversation_start_wall": self._conversation_started_at_wall,
            "total_samples": self._samples_written,
            "turns": self._turns,
        }
        with open(self.output_meta_path, "w") as fp:
            json.dump(meta, fp, indent=2)


