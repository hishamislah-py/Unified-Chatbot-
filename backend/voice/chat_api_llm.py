"""
Chat API LLM Adapter for LiveKit Agents
Routes voice transcriptions through the existing Chat API (RAG + Multi-Agent)
"""
from __future__ import annotations

import asyncio
import json
import aiohttp
from dataclasses import dataclass
from typing import Optional, Any

from livekit.agents import llm, APIConnectOptions
from livekit.agents.llm import ChatChunk, ChoiceDelta
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS


# Voice configurations for each agent
AGENT_VOICES = {
    "personal": "en-US-AriaNeural",      # Female, friendly
    "hr": "en-US-JennyNeural",           # Female, professional
    "it": "en-US-GuyNeural",             # Male, technical
}


@dataclass
class ChatAPIOptions:
    api_base: str = "http://localhost:8000"
    session_id: Optional[str] = None
    current_agent: str = "personal"


class ChatAPILLM(llm.LLM):
    """
    Custom LLM that routes requests through the Chat API.
    """

    def __init__(
        self,
        *,
        api_base: str = "http://localhost:8000",
        session_id: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._opts = ChatAPIOptions(api_base=api_base, session_id=session_id)
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._on_agent_change_callback = None
        self._closing = False

    async def aclose(self) -> None:
        """Clean up HTTP session"""
        self._closing = True
        if self._http_session is not None and not self._http_session.closed:
            await self._http_session.close()
            self._http_session = None
            print("[ChatAPILLM] HTTP session closed")

    def set_session_id(self, session_id: str) -> None:
        """Set the session ID to use for API calls"""
        self._opts.session_id = session_id
        print(f"[ChatAPILLM] Session ID set to: {session_id}")

    def set_agent_change_callback(self, callback):
        """Set callback to be called when agent changes (for voice switching)"""
        self._on_agent_change_callback = callback

    def get_current_voice(self) -> str:
        """Get the TTS voice for the current agent"""
        return AGENT_VOICES.get(self._opts.current_agent, "en-US-AriaNeural")

    async def _ensure_session(self) -> None:
        """Ensure we have an HTTP session and chat session ID"""
        if self._closing:
            return

        if self._http_session is None or self._http_session.closed:
            # Create session with reasonable timeouts
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self._http_session = aiohttp.ClientSession(timeout=timeout)

        if self._opts.session_id is None:
            try:
                async with self._http_session.post(
                    f"{self._opts.api_base}/api/sessions"
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self._opts.session_id = data.get("session_id")
                        print(f"[ChatAPILLM] Created session: {self._opts.session_id}")
                    else:
                        raise Exception(f"Failed to create session: {resp.status}")
            except aiohttp.ClientError as e:
                print(f"[ChatAPILLM] Connection error creating session: {e}")
                raise

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: Optional[list] = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        fnc_ctx: Optional[llm.FunctionContext] = None,
        temperature: Optional[float] = None,
        n: Optional[int] = None,
        parallel_tool_calls: Optional[bool] = None,
        tool_choice: Any = None,
        **kwargs,
    ) -> "ChatAPILLMStream":
        return ChatAPILLMStream(
            llm_instance=self,
            chat_ctx=chat_ctx,
            conn_options=conn_options,
            tools=tools or [],
        )


class ChatAPILLMStream(llm.LLMStream):
    """Stream that calls the Chat API and yields response chunks"""

    def __init__(
        self,
        *,
        llm_instance: ChatAPILLM,
        chat_ctx: llm.ChatContext,
        conn_options: APIConnectOptions,
        tools: list,
    ) -> None:
        super().__init__(
            llm=llm_instance,
            chat_ctx=chat_ctx,
            conn_options=conn_options,
            tools=tools,
        )
        self._llm_instance: ChatAPILLM = llm_instance
        self._chunk_id = 0
        self._cancelled = False

    def _create_chunk(self, content: str) -> ChatChunk:
        """Create a ChatChunk with the given content using v1.x format"""
        self._chunk_id += 1
        return ChatChunk(
            id=f"chatapi-{self._chunk_id}",
            delta=ChoiceDelta(role="assistant", content=content)
        )

    async def _run(self) -> None:
        """Execute the API call and emit response chunks"""
        # Check if already cancelled or closing
        if self._cancelled or self._llm_instance._closing:
            return

        try:
            await self._llm_instance._ensure_session()
        except Exception as e:
            print(f"[ChatAPILLM] Failed to ensure session: {e}")
            if not self._cancelled:
                self._event_ch.send_nowait(
                    self._create_chunk("I'm having connection issues. Please try again.")
                )
            return

        # Get the last user message from chat context
        user_message = ""
        for msg in reversed(self._chat_ctx.items):
            role = None
            if hasattr(msg, 'role'):
                role = str(msg.role).lower()

            if role == "user":
                if hasattr(msg, 'text_content') and msg.text_content:
                    user_message = msg.text_content
                    break
                elif hasattr(msg, 'content'):
                    content = msg.content
                    if isinstance(content, str) and content:
                        user_message = content
                        break
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, str) and item:
                                user_message = item
                                break
                            elif hasattr(item, 'text') and item.text:
                                user_message = item.text
                                break
                        if user_message:
                            break

        if not user_message:
            print("[ChatAPILLM] No user message found in context")
            if not self._cancelled:
                self._event_ch.send_nowait(self._create_chunk("I didn't catch that. Could you repeat?"))
            return

        print(f"[ChatAPILLM] Sending to API ({self._llm_instance._opts.current_agent}): {user_message[:50]}...")

        request_data = {
            "session_id": self._llm_instance._opts.session_id,
            "message": user_message,
            "agent": self._llm_instance._opts.current_agent,
            "source": "voice",  # Identify this as a voice request for event broadcasting
        }

        accumulated_text = ""
        current_event_type = None
        buffer = ""

        try:
            # Check if session is still valid
            if self._llm_instance._http_session is None or self._llm_instance._http_session.closed:
                print("[ChatAPILLM] HTTP session closed, skipping request")
                return

            async with self._llm_instance._http_session.post(
                f"{self._llm_instance._opts.api_base}/api/chat/stream",
                json=request_data,
                headers={"Accept": "text/event-stream"},
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    print(f"[ChatAPILLM] API error: {resp.status} - {error_text}")
                    if not self._cancelled:
                        self._event_ch.send_nowait(
                            self._create_chunk("I'm sorry, I'm having trouble connecting.")
                        )
                    return

                # Parse SSE stream - use proper line-based parsing
                async for chunk in resp.content.iter_any():
                    if self._cancelled or self._llm_instance._closing:
                        break

                    buffer += chunk.decode('utf-8')

                    # Process complete lines from buffer
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line_text = line.strip()

                        if not line_text:
                            continue

                        if line_text.startswith('event:'):
                            current_event_type = line_text[6:].strip()

                        elif line_text.startswith('data:'):
                            data_str = line_text[5:].strip()
                            if data_str:
                                try:
                                    data = json.loads(data_str)

                                    if current_event_type == 'token' or data.get('type') == 'token':
                                        content = data.get('content', '')
                                        accumulated_text += content
                                        if not self._cancelled:
                                            self._event_ch.send_nowait(self._create_chunk(content))

                                    elif current_event_type == 'complete':
                                        # Agent transfer happened - update for next request
                                        new_agent = data.get('agent', self._llm_instance._opts.current_agent)
                                        print(f"[ChatAPILLM] Complete event received - agent: {new_agent}")
                                        if new_agent != self._llm_instance._opts.current_agent:
                                            old_agent = self._llm_instance._opts.current_agent
                                            self._llm_instance._opts.current_agent = new_agent
                                            print(f"[ChatAPILLM] Agent changed: {old_agent} -> {new_agent}")
                                            print(f"[ChatAPILLM] Voice changed to: {self._llm_instance.get_current_voice()}")

                                            # Call the callback if set
                                            if self._llm_instance._on_agent_change_callback:
                                                self._llm_instance._on_agent_change_callback(new_agent)

                                    # Reset event type after processing data
                                    current_event_type = None

                                except json.JSONDecodeError:
                                    pass

                print(f"[ChatAPILLM] Response ({self._llm_instance._opts.current_agent}): {accumulated_text[:100]}...")

        except asyncio.CancelledError:
            print("[ChatAPILLM] Request cancelled")
            self._cancelled = True
        except aiohttp.ClientError as e:
            print(f"[ChatAPILLM] Connection error: {e}")
            if not self._cancelled:
                self._event_ch.send_nowait(
                    self._create_chunk("I'm having connection issues. Please try again.")
                )
        except Exception as e:
            print(f"[ChatAPILLM] Error: {e}")
            if not self._cancelled:
                self._event_ch.send_nowait(
                    self._create_chunk("I encountered an error.")
                )

    async def aclose(self) -> None:
        """Clean up resources"""
        self._cancelled = True
