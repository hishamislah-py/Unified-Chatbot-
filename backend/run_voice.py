"""
Run LiveKit Voice Agent
Usage: python run_voice.py
"""
import os
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Load environment
from dotenv import load_dotenv
env_path = backend_dir / "env" / ".env"
load_dotenv(env_path)

print("=" * 60)
print("LiveKit Voice Agent")
print("=" * 60)
print(f"LIVEKIT_URL: {os.getenv('LIVEKIT_URL', 'not set')}")
print(f"GROQ_API_KEY: {'set' if os.getenv('GROQ_API_KEY') else 'NOT SET!'}")
print("=" * 60)

# Now import and run the voice agent
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins.silero import VAD
from livekit.plugins.groq import STT
from voice.edge_tts_adapter import EdgeTTS
from voice.chat_api_llm import ChatAPILLM, AGENT_VOICES


async def entrypoint(ctx: JobContext):
    """Main entry point for LiveKit voice agent"""
    print(f"[VOICE] Agent starting for room: {ctx.room.name}")

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    print("[VOICE] Connected to LiveKit room")

    participant = await ctx.wait_for_participant()
    print(f"[VOICE] Participant joined: {participant.identity}")

    # Create the agent with instructions
    agent = Agent(
        instructions="""You are a voice interface for the enterprise assistant.
The backend handles all responses through the multi-agent system (Personal Assistant, HR, IT).
Keep your acknowledgments brief as responses come from the chat API.""",
    )

    # Create TTS with initial voice (Personal Assistant)
    tts = EdgeTTS(voice=AGENT_VOICES["personal"])

    # Create LLM with Chat API
    chat_llm = ChatAPILLM(api_base="http://localhost:8000")

    # Set up voice switching callback
    def on_agent_change(new_agent: str):
        new_voice = AGENT_VOICES.get(new_agent, AGENT_VOICES["personal"])
        tts.update_options(voice=new_voice)
        print(f"[VOICE] Switched to {new_agent} voice: {new_voice}")

    chat_llm.set_agent_change_callback(on_agent_change)

    # Create session with STT (Groq), LLM (Chat API), TTS (Edge TTS), VAD (Silero)
    # Optimized VAD settings for faster response
    session = AgentSession(
        stt=STT(model="whisper-large-v3"),
        llm=chat_llm,
        tts=tts,
        vad=VAD.load(
            min_silence_duration=0.25,  # 250ms (was 550ms) - faster end detection
            min_speech_duration=0.05,   # 50ms - minimum speech to register
            prefix_padding_duration=0.3, # 300ms (was 500ms) - less audio prefix
        ),
    )
    print("[VOICE] Using Chat API LLM (RAG + Multi-Agent)")
    print(f"[VOICE] Agent voices: Personal={AGENT_VOICES['personal']}, HR={AGENT_VOICES['hr']}, IT={AGENT_VOICES['it']}")

    # Start the session (must await)
    await session.start(agent=agent, room=ctx.room)

    # Generate initial greeting
    await session.generate_reply(
        instructions="Greet the user and offer your assistance as an enterprise assistant."
    )


if __name__ == "__main__":
    print("\nStarting Voice Agent Worker...")
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )
