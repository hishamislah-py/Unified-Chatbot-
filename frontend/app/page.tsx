import AIChatCard from "@/components/ui/ai-chat";
import VoiceAssistant from "@/components/ui/voice-assistant";

export default function Home() {
  return (
    <div className="flex min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Left Side - Voice Assistant */}
      <div className="flex-1 flex items-center justify-center p-8">
        <VoiceAssistant />
      </div>

      {/* Right Side - Chatbot */}
      <div className="flex-1 flex items-center justify-center p-8">
        <AIChatCard />
      </div>
    </div>
  );
}
