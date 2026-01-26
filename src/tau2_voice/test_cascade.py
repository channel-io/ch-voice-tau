"""
Test script for the cascade pipeline (ASR -> LLM -> TTS).

This script tests each component individually and then the full pipeline.

Usage:
    # Test with OpenAI TTS (requires OPENAI_API_KEY)
    python -m tau2_voice.test_cascade
    
    # Test with Chatterbox TTS (fully local)
    python -m tau2_voice.test_cascade --local-tts
    
    # Test with local LLM (fully local)
    python -m tau2_voice.test_cascade --local-llm --local-tts
"""

import asyncio
import argparse
import base64
import struct
import wave
from pathlib import Path

from loguru import logger


async def test_asr():
    """Test ASR provider with sample audio."""
    from tau2_voice.providers.asr import WhisperLocalProvider
    
    logger.info("=" * 60)
    logger.info("Testing ASR (Whisper)")
    logger.info("=" * 60)
    
    # Use a smaller model for testing
    asr = WhisperLocalProvider(
        model_id="openai/whisper-base",  # Use base for faster testing
        device="auto",
        language="en",
    )
    
    await asr.initialize()
    
    # Generate synthetic test audio (silence with some noise)
    # In real use, this would be actual speech audio
    import numpy as np
    duration = 2.0  # seconds
    sample_rate = 24000
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Generate a simple tone that sounds like speech
    audio = np.sin(2 * np.pi * 440 * t) * 0.1  # Quiet sine wave
    audio = (audio * 32767).astype(np.int16)
    audio_bytes = audio.tobytes()
    
    logger.info(f"Test audio: {len(audio_bytes)} bytes, {duration}s at {sample_rate}Hz")
    
    # Note: This will likely return empty or noise transcription
    # since we're using synthetic audio
    transcript = await asr.transcribe(audio_bytes, sample_rate=sample_rate)
    logger.info(f"Transcription: '{transcript}' (expected: empty or noise)")
    
    await asr.shutdown()
    logger.info("ASR test completed ✓")
    return True


async def test_llm(use_local: bool = False):
    """Test LLM provider."""
    logger.info("=" * 60)
    logger.info(f"Testing LLM ({'Local' if use_local else 'OpenAI'})")
    logger.info("=" * 60)
    
    from tau2_voice.models.message import SystemMessage, UserMessage
    
    if use_local:
        from tau2_voice.providers.llm import LocalLLMProvider
        llm = LocalLLMProvider(
            model_id="nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1",
            device="auto",
            max_new_tokens=100,
        )
    else:
        from tau2_voice.providers.llm import OpenAILLMProvider
        llm = OpenAILLMProvider(
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=100,
        )
    
    await llm.initialize()
    
    messages = [
        SystemMessage(role="system", content="You are a helpful assistant. Keep responses brief."),
        UserMessage(role="user", content="Hello! What's 2 + 2?"),
    ]
    
    logger.info("Streaming completion...")
    full_response = ""
    async for chunk in llm.stream_completion(messages):
        if chunk.content:
            full_response += chunk.content
            print(chunk.content, end="", flush=True)
    print()  # newline
    
    logger.info(f"Full response: {full_response}")
    
    await llm.shutdown()
    logger.info("LLM test completed ✓")
    return True


async def test_tts(use_local: bool = False):
    """Test TTS provider."""
    logger.info("=" * 60)
    logger.info(f"Testing TTS ({'Chatterbox' if use_local else 'OpenAI'})")
    logger.info("=" * 60)
    
    if use_local:
        from tau2_voice.providers.tts import ChatterboxTTSProvider
        tts = ChatterboxTTSProvider(
            device="auto",
            exaggeration=0.5,
            cfg_weight=0.5,
        )
    else:
        from tau2_voice.providers.tts import OpenAITTSProvider
        tts = OpenAITTSProvider(
            model="gpt-4o-mini-tts",
            voice="alloy",
        )
    
    await tts.initialize()
    
    text = "Hello! This is a test of the text to speech system."
    logger.info(f"Synthesizing: '{text}'")
    
    audio_chunks = []
    async for chunk in tts.synthesize_stream(text):
        audio_chunks.append(chunk)
    
    audio_data = b"".join(audio_chunks)
    logger.info(f"Generated {len(audio_data)} bytes of audio ({len(audio_data) / (tts.sample_rate * 2):.2f}s)")
    
    # Save to file for verification
    output_path = Path("test_tts_output.wav")
    save_wav(audio_data, output_path, tts.sample_rate)
    logger.info(f"Saved to {output_path}")
    
    await tts.shutdown()
    logger.info("TTS test completed ✓")
    return True


async def test_full_pipeline(use_local_llm: bool = False, use_local_tts: bool = False):
    """Test the full cascade pipeline."""
    logger.info("=" * 60)
    logger.info("Testing Full Cascade Pipeline")
    logger.info(f"  LLM: {'Local' if use_local_llm else 'OpenAI'}")
    logger.info(f"  TTS: {'Chatterbox' if use_local_tts else 'OpenAI'}")
    logger.info("=" * 60)
    
    from tau2_voice.agent.cascade import CascadeAgent
    from tau2_voice.providers.asr import WhisperLocalProvider
    from tau2_voice.providers.llm import OpenAILLMProvider, LocalLLMProvider
    from tau2_voice.providers.tts import OpenAITTSProvider, ChatterboxTTSProvider
    from tau2_voice.models.events import TranscriptUpdateEvent
    
    # Create providers
    asr = WhisperLocalProvider(
        model_id="openai/whisper-base",
        device="auto",
        language="en",
    )
    
    if use_local_llm:
        llm = LocalLLMProvider(
            model_id="nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1",
            device="auto",
            max_new_tokens=200,
        )
    else:
        llm = OpenAILLMProvider(model="gpt-4o-mini")
    
    if use_local_tts:
        tts = ChatterboxTTSProvider(device="auto")
    else:
        tts = OpenAITTSProvider(model="gpt-4o-mini-tts", voice="alloy")
    
    # Create cascade agent
    agent = CascadeAgent(
        tools=None,
        domain_policy="You are a helpful assistant. Keep responses brief and friendly.",
        asr_provider=asr,
        llm_provider=llm,
        tts_provider=tts,
        role="assistant",
    )
    
    await agent.connect()
    
    # Instead of audio, send a transcript directly (simulates ASR output)
    logger.info("Sending test transcript...")
    test_transcript = TranscriptUpdateEvent(
        role="user",
        message_id="test_1",
        transcript="Hello! Can you tell me a short joke?",
    )
    await agent.publish(test_transcript)
    
    # Collect responses
    logger.info("Collecting responses...")
    audio_chunks = []
    transcript = ""
    
    async def collect_events():
        nonlocal transcript, audio_chunks
        timeout = 30  # seconds
        start = asyncio.get_event_loop().time()
        
        async for event in agent.subscribe():
            if asyncio.get_event_loop().time() - start > timeout:
                break
            
            if event.type == "audio.chunk":
                audio_chunks.append(base64.b64decode(event.audio_chunk))
            elif event.type == "transcript.update":
                transcript = event.transcript
                logger.info(f"Response transcript: {transcript}")
            elif event.type == "audio.done":
                logger.info("Audio done received")
                break
    
    await asyncio.wait_for(collect_events(), timeout=60)
    
    # Save audio output
    if audio_chunks:
        audio_data = b"".join(audio_chunks)
        output_path = Path("test_pipeline_output.wav")
        save_wav(audio_data, output_path, tts.sample_rate)
        logger.info(f"Saved audio to {output_path} ({len(audio_data)} bytes)")
    
    await agent.disconnect()
    
    logger.info("Full pipeline test completed ✓")
    logger.info(f"Response: {transcript}")
    return True


def save_wav(pcm_data: bytes, path: Path, sample_rate: int):
    """Save PCM data to WAV file."""
    with wave.open(str(path), 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)


async def main():
    parser = argparse.ArgumentParser(description="Test cascade pipeline")
    parser.add_argument("--local-llm", action="store_true", help="Use local LLM (Nemotron)")
    parser.add_argument("--local-tts", action="store_true", help="Use local TTS (Chatterbox)")
    parser.add_argument("--skip-asr", action="store_true", help="Skip ASR test")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM test")
    parser.add_argument("--skip-tts", action="store_true", help="Skip TTS test")
    parser.add_argument("--only-pipeline", action="store_true", help="Only test full pipeline")
    args = parser.parse_args()
    
    try:
        if args.only_pipeline:
            await test_full_pipeline(args.local_llm, args.local_tts)
        else:
            if not args.skip_asr:
                await test_asr()
                print()
            
            if not args.skip_llm:
                await test_llm(args.local_llm)
                print()
            
            if not args.skip_tts:
                await test_tts(args.local_tts)
                print()
            
            await test_full_pipeline(args.local_llm, args.local_tts)
        
        logger.info("=" * 60)
        logger.info("All tests completed successfully! ✓")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(main())

