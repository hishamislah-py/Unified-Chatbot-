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
from livekit.plugins.groq import STT, LLM
from voice.edge_tts_adapter import EdgeTTS


async def entrypoint(ctx: JobContext):
    """Main entry point for LiveKit voice agent"""
    print(f"[VOICE] Agent starting for room: {ctx.room.name}")

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    print("[VOICE] Connected to LiveKit room")

    participant = await ctx.wait_for_participant()
    print(f"[VOICE] Participant joined: {participant.identity}")

    # Create the agent with instructions
    agent = Agent(
        instructions="""You are a helpful enterprise assistant for voice conversations.
You can answer general questions about company policies, HR matters, and IT support.
Be concise and professional. Keep responses brief and natural for voice output.
Limit responses to 2-3 sentences for natural conversation flow.""",
    )

    # Create session with STT (Groq), LLM (Groq), TTS (Edge TTS - FREE), VAD (Silero)
    session = AgentSession(
        stt=STT(model="whisper-large-v3"),
        llm=LLM(model="llama-3.3-70b-versatile"),
        tts=EdgeTTS(voice="en-US-AriaNeural"),  # Free Microsoft TTS
        vad=VAD.load(),
    )

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
