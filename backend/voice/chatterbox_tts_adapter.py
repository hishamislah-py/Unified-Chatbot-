"""
Chatterbox TTS Adapter for LiveKit Agents
Emotion-based TTS using Chatterbox API at ART Technology server
"""
from __future__ import annotations

import os
import io
import wave
import struct
import asyncio
import aiohttp
from dataclasses import dataclass

from livekit.agents import (
    APIConnectOptions,
    APIConnectionError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

# Chatterbox TTS audio settings
SAMPLE_RATE = 24000
NUM_CHANNELS = 1


def convert_float32_wav_to_pcm16(audio_data: bytes) -> bytes:
    """
    Convert IEEE float32 WAV (format 3) to PCM16 WAV (format 1).
    LiveKit only supports PCM format.

    Note: Python's wave module doesn't support float32, so we parse manually.
    """
    try:
        # Check if it's a WAV file
        if len(audio_data) < 44 or audio_data[:4] != b'RIFF' or audio_data[8:12] != b'WAVE':
            print("[ChatterboxTTS] Not a valid WAV file, returning as-is")
            return audio_data

        # Parse WAV header manually
        # RIFF header: 12 bytes
        # fmt chunk: variable

        # Find fmt chunk
        pos = 12
        fmt_found = False
        audio_format = 1
        n_channels = 1
        sample_rate = 24000
        bits_per_sample = 16
        data_start = 0
        data_size = 0

        while pos < len(audio_data) - 8:
            chunk_id = audio_data[pos:pos+4]
            chunk_size = struct.unpack('<I', audio_data[pos+4:pos+8])[0]

            if chunk_id == b'fmt ':
                fmt_found = True
                audio_format = struct.unpack('<H', audio_data[pos+8:pos+10])[0]
                n_channels = struct.unpack('<H', audio_data[pos+10:pos+12])[0]
                sample_rate = struct.unpack('<I', audio_data[pos+12:pos+16])[0]
                # byte_rate = struct.unpack('<I', audio_data[pos+16:pos+20])[0]
                # block_align = struct.unpack('<H', audio_data[pos+20:pos+22])[0]
                bits_per_sample = struct.unpack('<H', audio_data[pos+22:pos+24])[0]

            elif chunk_id == b'data':
                data_start = pos + 8
                data_size = chunk_size
                break

            pos += 8 + chunk_size
            # Align to even byte
            if chunk_size % 2 == 1:
                pos += 1

        if not fmt_found or data_start == 0:
            print("[ChatterboxTTS] Invalid WAV structure, returning as-is")
            return audio_data

        print(f"[ChatterboxTTS] WAV format: {audio_format}, channels: {n_channels}, rate: {sample_rate}, bits: {bits_per_sample}")

        # Format 3 = IEEE float, Format 1 = PCM
        if audio_format == 3 and bits_per_sample == 32:
            # Extract float32 audio data
            raw_data = audio_data[data_start:data_start + data_size]
            num_samples = len(raw_data) // 4

            # Convert float32 to int16
            float_samples = struct.unpack(f'<{num_samples}f', raw_data)
            int16_samples = []
            for sample in float_samples:
                # Clamp to [-1, 1] range and convert to int16
                sample = max(-1.0, min(1.0, sample))
                int16_samples.append(int(sample * 32767))

            pcm_data = struct.pack(f'<{num_samples}h', *int16_samples)

            # Build new PCM WAV file
            output_buffer = io.BytesIO()
            with wave.open(output_buffer, 'wb') as wav_out:
                wav_out.setnchannels(n_channels)
                wav_out.setsampwidth(2)  # 16-bit PCM
                wav_out.setframerate(sample_rate)
                wav_out.writeframes(pcm_data)

            print(f"[ChatterboxTTS] Converted float32 to PCM16: {len(raw_data)} -> {len(pcm_data)} bytes")
            return output_buffer.getvalue()

        elif audio_format == 1:
            # Already PCM, return as-is
            print("[ChatterboxTTS] Already PCM format, returning as-is")
            return audio_data
        else:
            print(f"[ChatterboxTTS] Unknown format {audio_format}, returning as-is")
            return audio_data

    except Exception as e:
        print(f"[ChatterboxTTS] WAV conversion failed: {e}, returning original")
        return audio_data


@dataclass
class _TTSOptions:
    voice: str
    emotion: str
    api_base: str


# Agent configurations - voice + emotion per agent
# Voices: Update once you know available voices from your Chatterbox server
# Emotions: neutral, happy, sad, angry, excited, calm, professional, friendly
AGENT_CONFIG = {
    "personal": {
        "voice": os.getenv("TTS_VOICE_PERSONAL", "default"),
        "emotion": os.getenv("TTS_EMOTION_PERSONAL", "friendly"),
    },
    "hr": {
        "voice": os.getenv("TTS_VOICE_HR", "default"),
        "emotion": os.getenv("TTS_EMOTION_HR", "professional"),
    },
    "it": {
        "voice": os.getenv("TTS_VOICE_IT", "default"),
        "emotion": os.getenv("TTS_EMOTION_IT", "calm"),
    },
}


class ChatterboxTTS(tts.TTS):
    """
    Chatterbox TTS - Emotion-based text-to-speech

    Supports emotions: neutral, happy, sad, angry, excited, calm, professional, friendly

    API Endpoints:
    - POST /synthesize-emotion: Emotion-based synthesis
    - POST /synthesize-with-voice: Voice cloning (requires voice_file)
    - POST /v1/audio/speech: OpenAI-compatible endpoint
    """

    def __init__(
        self,
        *,
        voice: str = "default",
        emotion: str = "friendly",
        api_base: str | None = None,
    ) -> None:
        """
        Create a new instance of Chatterbox TTS.

        Args:
            voice: Voice to use. Default is "default".
            emotion: Emotion for synthesis. Options: neutral, happy, sad, angry, excited, calm, professional, friendly
            api_base: Base URL for Chatterbox API. Defaults to env var CHATTERBOX_TTS_URL.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),  # Chatterbox doesn't support streaming
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )

        if api_base is None:
            api_base = os.getenv("CHATTERBOX_TTS_URL", "https://ai.arttechgroup.com:7777/tts")

        self._opts = _TTSOptions(voice=voice, emotion=emotion, api_base=api_base)
        self._http_session: aiohttp.ClientSession | None = None

    @property
    def model(self) -> str:
        return "chatterbox-tts"

    @property
    def provider(self) -> str:
        return "Chatterbox"

    def update_options(
        self,
        *,
        voice: str | None = None,
        emotion: str | None = None,
    ) -> None:
        """
        Update the TTS options (for agent voice/emotion switching).

        Args:
            voice: New voice to use
            emotion: New emotion for synthesis
        """
        if voice is not None:
            self._opts.voice = voice
        if emotion is not None:
            self._opts.emotion = emotion

    def update_for_agent(self, agent: str) -> None:
        """
        Update voice and emotion settings for a specific agent.

        Args:
            agent: Agent name (personal, hr, it)
        """
        config = AGENT_CONFIG.get(agent, AGENT_CONFIG["personal"])
        self._opts.voice = config["voice"]
        self._opts.emotion = config["emotion"]

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._http_session is None or self._http_session.closed:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self._http_session = aiohttp.ClientSession(timeout=timeout)
        return self._http_session

    async def aclose(self) -> None:
        if self._http_session is not None and not self._http_session.closed:
            await self._http_session.close()

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "ChatterboxChunkedStream":
        return ChatterboxChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
        )


class ChatterboxChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: ChatterboxTTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: ChatterboxTTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            session = await self._tts._ensure_session()

            # Use synthesize-emotion endpoint
            # Send as form data (matches the curl example)
            form_data = aiohttp.FormData()
            form_data.add_field("text", self._input_text)
            form_data.add_field("emotion", self._tts._opts.emotion)

            async with session.post(
                f"{self._tts._opts.api_base}/synthesize-emotion",
                data=form_data,
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise APIConnectionError(
                        f"Chatterbox API error: {resp.status} - {error_text}"
                    )

                # Get audio data
                audio_data = await resp.read()

                if not audio_data:
                    raise APIConnectionError("No audio data received from Chatterbox TTS")

                # Determine content type from response
                content_type = resp.headers.get("content-type", "audio/wav")

                # Convert float32 WAV to PCM16 if needed (LiveKit only supports PCM)
                if "wav" in content_type:
                    audio_data = convert_float32_wav_to_pcm16(audio_data)
                    mime_type = "audio/wav"
                elif "mp3" in content_type or "mpeg" in content_type:
                    mime_type = "audio/mpeg"
                elif "ogg" in content_type:
                    mime_type = "audio/ogg"
                else:
                    # Try to convert as WAV anyway
                    audio_data = convert_float32_wav_to_pcm16(audio_data)
                    mime_type = "audio/wav"

                # Initialize emitter
                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=SAMPLE_RATE,
                    num_channels=NUM_CHANNELS,
                    mime_type=mime_type,
                )

                # Push audio data
                output_emitter.push(audio_data)
                output_emitter.flush()

        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientError as e:
            raise APIConnectionError(str(e)) from e
