"""
Whisper Edge STT Adapter for LiveKit Agents
Speech-to-Text using Whisper Edge API at ART Technology server
"""
from __future__ import annotations

import os
import asyncio
import io
import wave
import aiohttp
from dataclasses import dataclass

from livekit.agents import (
    APIConnectOptions,
    APIConnectionError,
    APITimeoutError,
    stt,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS


def create_wav_from_pcm(pcm_data: bytes, sample_rate: int = 16000, channels: int = 1, sample_width: int = 2) -> bytes:
    """
    Create a WAV file from raw PCM data.
    LiveKit provides raw PCM16 audio, but Whisper API expects WAV files.
    """
    output_buffer = io.BytesIO()
    with wave.open(output_buffer, 'wb') as wav_out:
        wav_out.setnchannels(channels)
        wav_out.setsampwidth(sample_width)
        wav_out.setframerate(sample_rate)
        wav_out.writeframes(pcm_data)
    return output_buffer.getvalue()


@dataclass
class _STTOptions:
    api_base: str
    language: str


class WhisperEdgeSTT(stt.STT):
    """
    Whisper Edge STT - Speech-to-text using self-hosted Whisper

    API Endpoint:
    - POST /transcribe: Transcribe audio file, returns JSON with transcription
    """

    def __init__(
        self,
        *,
        api_base: str | None = None,
        language: str = "en",
    ) -> None:
        """
        Create a new instance of Whisper Edge STT.

        Args:
            api_base: Base URL for Whisper Edge API. Defaults to env var WHISPER_EDGE_URL.
            language: Language code for transcription. Default is "en".
        """
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )

        if api_base is None:
            api_base = os.getenv("WHISPER_EDGE_URL", "https://ai.arttechgroup.com:7777/stt")

        self._opts = _STTOptions(api_base=api_base, language=language)
        self._http_session: aiohttp.ClientSession | None = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._http_session is None or self._http_session.closed:
            timeout = aiohttp.ClientTimeout(total=60, connect=10)
            self._http_session = aiohttp.ClientSession(timeout=timeout)
        return self._http_session

    async def aclose(self) -> None:
        if self._http_session is not None and not self._http_session.closed:
            await self._http_session.close()

    def _sanitize_options(self, *, language: str | None = None) -> _STTOptions:
        """Create options with overrides"""
        return _STTOptions(
            api_base=self._opts.api_base,
            language=language or self._opts.language,
        )

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        """
        Recognize speech from audio buffer.

        Args:
            buffer: Audio buffer containing speech data
            language: Language code override
            conn_options: Connection options

        Returns:
            SpeechEvent with transcription results
        """
        opts = self._sanitize_options(language=language)

        try:
            session = await self._ensure_session()

            # Get raw PCM data from buffer
            # LiveKit AudioBuffer contains raw PCM16 audio
            if hasattr(buffer, 'data'):
                pcm_data = buffer.data
            elif hasattr(buffer, 'frames'):
                # Try to get from frames
                pcm_data = b''.join([f.data for f in buffer.frames])
            else:
                # Last resort - try iterating
                try:
                    pcm_data = b''.join([frame.data for frame in buffer])
                except:
                    pcm_data = bytes(buffer) if buffer else b''

            if not pcm_data:
                print("[WhisperEdgeSTT] No audio data to transcribe")
                return stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[stt.SpeechData(language=opts.language, text="", confidence=0.0)],
                )

            # Get sample rate from buffer if available (LiveKit uses 48kHz internally, but resamples)
            sample_rate = getattr(buffer, 'sample_rate', 16000)
            channels = getattr(buffer, 'num_channels', 1)

            print(f"[WhisperEdgeSTT] Processing audio: {len(pcm_data)} bytes, {sample_rate}Hz, {channels}ch")

            # Convert raw PCM to WAV format (Whisper API expects WAV files)
            wav_data = create_wav_from_pcm(pcm_data, sample_rate=sample_rate, channels=channels)

            print(f"[WhisperEdgeSTT] WAV created: {len(wav_data)} bytes")

            # Create form data with file upload
            form_data = aiohttp.FormData()
            form_data.add_field(
                "file",
                io.BytesIO(wav_data),
                filename="audio.wav",
                content_type="audio/wav",
            )

            print(f"[WhisperEdgeSTT] Sending to {opts.api_base}/transcribe")

            async with session.post(
                f"{opts.api_base}/transcribe",
                data=form_data,
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    print(f"[WhisperEdgeSTT] API error: {resp.status} - {error_text[:200]}")
                    raise APIConnectionError(
                        f"Whisper Edge API error: {resp.status} - {error_text}"
                    )

                result = await resp.json()
                print(f"[WhisperEdgeSTT] Response: {result}")

                # Extract transcription from response
                # Expected format: {"text": "transcription..."}
                text = result.get("text", "").strip()

                if not text:
                    # Return empty result for no speech detected
                    return stt.SpeechEvent(
                        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                        alternatives=[
                            stt.SpeechData(
                                language=opts.language,
                                text="",
                                confidence=0.0,
                            )
                        ],
                    )

                print(f"[WhisperEdgeSTT] Transcription: {text}")
                return stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[
                        stt.SpeechData(
                            language=opts.language,
                            text=text,
                            confidence=1.0,  # Whisper doesn't provide confidence
                        )
                    ],
                )

        except asyncio.TimeoutError:
            print("[WhisperEdgeSTT] Timeout error")
            raise APITimeoutError() from None
        except aiohttp.ClientError as e:
            print(f"[WhisperEdgeSTT] Client error: {e}")
            raise APIConnectionError(str(e)) from e
        except Exception as e:
            print(f"[WhisperEdgeSTT] Unexpected error: {e}")
            raise APIConnectionError(str(e)) from e

    def stream(
        self,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "WhisperEdgeRecognizeStream":
        """
        Create a streaming recognition session.
        Note: Whisper Edge doesn't support true streaming, so this accumulates audio
        and sends it when finalized.
        """
        opts = self._sanitize_options(language=language)
        return WhisperEdgeRecognizeStream(
            stt=self,
            opts=opts,
            conn_options=conn_options,
        )


class WhisperEdgeRecognizeStream(stt.RecognizeStream):
    """
    Streaming wrapper for Whisper Edge STT.
    Accumulates audio and transcribes when finalized.
    """

    def __init__(
        self,
        *,
        stt: WhisperEdgeSTT,
        opts: _STTOptions,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options)
        self._stt: WhisperEdgeSTT = stt
        self._opts = opts
        self._audio_chunks: list[bytes] = []

    def push_frame(self, frame: stt.AudioFrame) -> None:
        """Add audio frame to buffer"""
        self._audio_chunks.append(frame.data)

    async def _run(self) -> None:
        """Process accumulated audio when stream ends"""
        if not self._audio_chunks:
            return

        try:
            session = await self._stt._ensure_session()

            # Combine all audio chunks (raw PCM data)
            combined_pcm = b"".join(self._audio_chunks)

            # Convert raw PCM to WAV format (Whisper API expects WAV files)
            # Default to 16kHz mono for voice
            wav_data = create_wav_from_pcm(combined_pcm, sample_rate=16000, channels=1)

            # Create form data with file upload
            form_data = aiohttp.FormData()
            form_data.add_field(
                "file",
                io.BytesIO(wav_data),
                filename="audio.wav",
                content_type="audio/wav",
            )

            async with session.post(
                f"{self._opts.api_base}/transcribe",
                data=form_data,
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise APIConnectionError(
                        f"Whisper Edge API error: {resp.status} - {error_text}"
                    )

                result = await resp.json()
                text = result.get("text", "").strip()

                if text:
                    event = stt.SpeechEvent(
                        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                        alternatives=[
                            stt.SpeechData(
                                language=self._opts.language,
                                text=text,
                                confidence=1.0,
                            )
                        ],
                    )
                    self._event_ch.send_nowait(event)

        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientError as e:
            raise APIConnectionError(str(e)) from e

    async def aclose(self) -> None:
        """Clean up resources"""
        self._audio_chunks.clear()
