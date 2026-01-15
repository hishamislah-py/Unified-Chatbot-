import { useState, useCallback, useEffect, useRef } from "react";
import AIChatCard from "@/components/ui/ai-chat";
import VoiceAssistant from "@/components/ui/voice-assistant";

const API_BASE_URL = "http://localhost:8000/api";

type AgentType = "personal" | "hr" | "it";

interface Message {
  sender: "ai" | "user";
  text: string;
  sources?: { source: string; page: number; rank: number; preview: string }[];
}

export default function App() {
  const [sessionId, setSessionId] = useState<string>("");
  const [voiceMessage, setVoiceMessage] = useState<string | null>(null);
  const [voiceActive, setVoiceActive] = useState(false);
  const [partialTranscription, setPartialTranscription] = useState("");

  // Refs to hold the chat component's exposed functions
  const addMessageRef = useRef<((msg: Message, agent?: AgentType) => void) | null>(null);
  const setActiveAgentRef = useRef<((agent: AgentType) => void) | null>(null);

  // Called when chat session is ready
  const handleSessionReady = useCallback((id: string) => {
    setSessionId(id);
  }, []);

  // Called when voice transcription is complete (fallback for non-SSE mode)
  const handleVoiceTranscription = useCallback((text: string) => {
    if (text.trim()) {
      setVoiceMessage(text);
    }
  }, []);

  // Called when external message has been processed by chat
  const handleExternalMessageProcessed = useCallback(() => {
    setVoiceMessage(null);
  }, []);

  // Handle voice active state change
  const handleVoiceActiveChange = useCallback((active: boolean) => {
    setVoiceActive(active);
    if (!active) {
      // Clear partial transcription when voice disconnects
      setPartialTranscription("");
    }
  }, []);

  // Handle partial transcription updates (real-time as user speaks)
  const handlePartialTranscription = useCallback((text: string) => {
    setPartialTranscription(text);
  }, []);

  // Subscribe to voice events SSE when voice is active
  useEffect(() => {
    if (!sessionId || !voiceActive) return;

    console.log("[App] Subscribing to voice events for session:", sessionId);
    const eventSource = new EventSource(`${API_BASE_URL}/voice/events/${sessionId}`);

    eventSource.addEventListener("user_message", (e) => {
      try {
        const data = JSON.parse(e.data);
        console.log("[App] Voice user_message:", data);
        // Clear partial transcription and add final message
        setPartialTranscription("");
        if (addMessageRef.current) {
          addMessageRef.current({ sender: "user", text: data.text }, data.agent);
        }
      } catch (err) {
        console.error("[App] Error parsing user_message:", err);
      }
    });

    eventSource.addEventListener("ai_complete", (e) => {
      try {
        const data = JSON.parse(e.data);
        console.log("[App] Voice ai_complete:", data);
        if (addMessageRef.current) {
          addMessageRef.current(
            { sender: "ai", text: data.text, sources: data.sources },
            data.agent
          );
        }
        // Switch agent if changed
        if (setActiveAgentRef.current && data.agent) {
          setActiveAgentRef.current(data.agent);
        }
      } catch (err) {
        console.error("[App] Error parsing ai_complete:", err);
      }
    });

    eventSource.addEventListener("keepalive", () => {
      // Ignore keepalive events
    });

    eventSource.onerror = (err) => {
      console.error("[App] Voice events SSE error:", err);
    };

    return () => {
      console.log("[App] Closing voice events SSE connection");
      eventSource.close();
    };
  }, [sessionId, voiceActive]);

  // Callbacks to receive the exposed functions from chat component
  const handleAddMessage = useCallback((fn: (msg: Message, agent?: AgentType) => void) => {
    addMessageRef.current = fn;
  }, []);

  const handleSetActiveAgent = useCallback((fn: (agent: AgentType) => void) => {
    setActiveAgentRef.current = fn;
  }, []);

  return (
    <div className="flex min-h-screen bg-gradient-to-br from-slate-900 via-black to-slate-900">
      {/* Left Side - Voice Assistant */}
      <div className="flex-1 flex items-center justify-center p-8">
        <VoiceAssistant
          sessionId={sessionId}
          onTranscriptionComplete={handleVoiceTranscription}
          onVoiceActiveChange={handleVoiceActiveChange}
          onPartialTranscription={handlePartialTranscription}
        />
      </div>

      {/* Right Side - Chatbot */}
      <div className="flex-1 flex items-center justify-center p-8">
        <AIChatCard
          onSessionReady={handleSessionReady}
          externalMessage={voiceMessage}
          onExternalMessageProcessed={handleExternalMessageProcessed}
          voiceActive={voiceActive}
          partialTranscription={partialTranscription}
          onAddMessage={handleAddMessage}
          onSetActiveAgent={handleSetActiveAgent}
        />
      </div>
    </div>
  );
}
