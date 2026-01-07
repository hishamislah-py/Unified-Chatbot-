"""
Edge TTS Adapter for LiveKit Agents
Free TTS using Microsoft Edge's speech synthesis - no API key required
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import edge_tts

from livekit.agents import (
    APIConnectOptions,
    APIConnectionError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

# Edge TTS outputs MP3 at 24kHz
SAMPLE_RATE = 24000
NUM_CHANNELS = 1


@dataclass
class _TTSOptions:
    voice: str


class EdgeTTS(tts.TTS):
    """
    Edge TTS - Free text-to-speech using Microsoft Edge

    No API key required! Uses Microsoft's edge-tts library.

    Popular voices:
    - en-US-AriaNeural (female, natural)
    - en-US-GuyNeural (male, natural)
    - en-US-JennyNeural (female, conversational)
    - en-GB-SoniaNeural (British female)
    - en-AU-NatashaNeural (Australian female)
    """

    def __init__(
        self,
        *,
        voice: str = "en-US-AriaNeural",
    ) -> None:
        """
        Create a new instance of Edge TTS.

        Args:
            voice (str): Voice to use. Default is "en-US-AriaNeural".
                        Run `edge-tts --list-voices` to see all available voices.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),  # Chunked mode (Edge TTS doesn't support true streaming)
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )

        self._opts = _TTSOptions(voice=voice)

    @property
    def model(self) -> str:
        return "edge-tts"

    @property
    def provider(self) -> str:
        return "Microsoft Edge"

    def update_options(self, *, voice: str | None = None) -> None:
        """Update the TTS options."""
        if voice is not None:
            self._opts.voice = voice

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "EdgeChunkedStream":
        return EdgeChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class EdgeChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: EdgeTTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: EdgeTTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            # Create Edge TTS communicator
            communicate = edge_tts.Communicate(
                self._input_text,
                self._tts._opts.voice
            )

            # Initialize emitter before first chunk (for streaming)
            initialized = False
            has_audio = False

            # Stream audio chunks as they arrive (low latency)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    # Initialize on first audio chunk
                    if not initialized:
                        output_emitter.initialize(
                            request_id=utils.shortuuid(),
                            sample_rate=SAMPLE_RATE,
                            num_channels=NUM_CHANNELS,
                            mime_type="audio/mpeg",  # Edge TTS outputs MP3
                        )
                        initialized = True

                    # Push chunk immediately (streaming)
                    output_emitter.push(chunk["data"])
                    has_audio = True

            if not has_audio:
                raise APIConnectionError("No audio data received from Edge TTS")

            # Flush remaining data
            output_emitter.flush()

        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except Exception as e:
            raise APIConnectionError(str(e)) from e
