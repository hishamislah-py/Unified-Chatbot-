import { useState, useCallback } from "react";
import AIChatCard from "@/components/ui/ai-chat";
import VoiceAssistant from "@/components/ui/voice-assistant";

export default function App() {
  const [sessionId, setSessionId] = useState<string>("");
  const [voiceMessage, setVoiceMessage] = useState<string | null>(null);

  // Called when chat session is ready
  const handleSessionReady = useCallback((id: string) => {
    setSessionId(id);
  }, []);

  // Called when voice transcription is complete
  const handleVoiceTranscription = useCallback((text: string) => {
    if (text.trim()) {
      setVoiceMessage(text);
    }
  }, []);

  // Called when external message has been processed by chat
  const handleExternalMessageProcessed = useCallback(() => {
    setVoiceMessage(null);
  }, []);

  return (
    <div className="flex min-h-screen bg-gradient-to-br from-slate-900 via-black to-slate-900">
      {/* Left Side - Voice Assistant */}
      <div className="flex-1 flex items-center justify-center p-8">
        <VoiceAssistant
          sessionId={sessionId}
          onTranscriptionComplete={handleVoiceTranscription}
        />
      </div>

      {/* Right Side - Chatbot */}
      <div className="flex-1 flex items-center justify-center p-8">
        <AIChatCard
          onSessionReady={handleSessionReady}
          externalMessage={voiceMessage}
          onExternalMessageProcessed={handleExternalMessageProcessed}
        />
      </div>
    </div>
  );
}
