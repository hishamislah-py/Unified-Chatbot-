import { useState, useCallback, useEffect } from "react";
import { motion } from "framer-motion";
import { Mic, X, MicOff } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  LiveKitRoom,
  useVoiceAssistant,
  RoomAudioRenderer,
  useConnectionState,
} from "@livekit/components-react";
import { ConnectionState } from "livekit-client";

const API_BASE_URL = "http://localhost:8000/api";

type VoiceState = "idle" | "connecting" | "connected" | "listening" | "speaking" | "error";

interface VoiceAssistantProps {
  className?: string;
  sessionId?: string;
  onTranscriptionComplete?: (text: string) => void;
}

// Inner component for LiveKit hooks (must be inside LiveKitRoom)
function LiveKitVoiceHandler({
  onStateChange,
  onTranscription,
  onUserTranscriptionComplete,
}: {
  onStateChange: (state: VoiceState) => void;
  onTranscription: (userText: string, agentText: string) => void;
  onUserTranscriptionComplete?: (text: string) => void;
}) {
  const voiceAssistant = useVoiceAssistant();
  const connectionState = useConnectionState();
  const [lastTranscriptionCount, setLastTranscriptionCount] = useState(0);

  useEffect(() => {
    if (connectionState === ConnectionState.Connecting) {
      onStateChange("connecting");
    } else if (connectionState === ConnectionState.Connected) {
      if (voiceAssistant.state.agentState === "speaking") {
        onStateChange("speaking");
      } else if (voiceAssistant.state.agentState === "listening") {
        onStateChange("listening");
      } else {
        onStateChange("connected");
      }
    } else if (connectionState === ConnectionState.Disconnected) {
      onStateChange("idle");
    }
  }, [connectionState, voiceAssistant.state.agentState, onStateChange]);

  // Track transcriptions for display
  useEffect(() => {
    const userText = voiceAssistant.userTranscriptions?.[voiceAssistant.userTranscriptions.length - 1]?.text || "";
    const agentText = voiceAssistant.agentTranscriptions?.[voiceAssistant.agentTranscriptions.length - 1]?.text || "";
    onTranscription(userText, agentText);
  }, [voiceAssistant.userTranscriptions, voiceAssistant.agentTranscriptions, onTranscription]);

  // Detect when a new complete user transcription is available and send to chat
  useEffect(() => {
    const transcriptions = voiceAssistant.userTranscriptions || [];
    if (transcriptions.length > lastTranscriptionCount) {
      const latestTranscription = transcriptions[transcriptions.length - 1];
      if (latestTranscription?.text && onUserTranscriptionComplete) {
        onUserTranscriptionComplete(latestTranscription.text);
      }
      setLastTranscriptionCount(transcriptions.length);
    }
  }, [voiceAssistant.userTranscriptions, lastTranscriptionCount, onUserTranscriptionComplete]);

  return <RoomAudioRenderer />;
}

export default function VoiceAssistant({ className, sessionId, onTranscriptionComplete }: VoiceAssistantProps) {
  const [voiceState, setVoiceState] = useState<VoiceState>("idle");
  const [connectionInfo, setConnectionInfo] = useState<{
    token: string;
    livekitUrl: string;
  } | null>(null);
  const [userName] = useState("User");
  const [userText, setUserText] = useState("");
  const [agentText, setAgentText] = useState("");
  const [error, setError] = useState("");

  const isConnected = connectionInfo !== null;
  const isListening = voiceState === "listening" || voiceState === "speaking" || voiceState === "connected";

  // Handle transcription updates
  const handleTranscription = useCallback((user: string, agent: string) => {
    if (user) setUserText(user);
    if (agent) setAgentText(agent);
  }, []);

  // Connect to LiveKit
  const connectVoice = useCallback(async () => {
    setVoiceState("connecting");
    setError("");
    setUserText("");
    setAgentText("");

    try {
      const response = await fetch(`${API_BASE_URL}/livekit/token`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      });

      if (!response.ok) throw new Error("Failed to connect");

      const data = await response.json();
      setConnectionInfo({
        token: data.token,
        livekitUrl: data.livekit_url,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Connection failed");
      setVoiceState("error");
    }
  }, [sessionId]);

  // Disconnect
  const disconnectVoice = useCallback(() => {
    setConnectionInfo(null);
    setVoiceState("idle");
    setUserText("");
    setAgentText("");
  }, []);

  // Toggle connection
  const handleMicClick = () => {
    if (isConnected) {
      disconnectVoice();
    } else {
      connectVoice();
    }
  };

  // Status text
  const getStatusText = () => {
    switch (voiceState) {
      case "connecting": return "Connecting...";
      case "connected": return "Connected - speak now";
      case "listening": return `I'm listening, ${userName}...`;
      case "speaking": return "Speaking...";
      case "error": return error || "Connection error";
      default: return "Click mic to start";
    }
  };

  const content = (
    <div
      className={cn(
        "relative w-full h-full flex flex-col items-center justify-center px-8 py-12",
        className
      )}
    >
      {/* Glowing Orb */}
      <div className="relative mb-12">
        <motion.div
          className="relative w-48 h-48"
          animate={{
            scale: isListening ? [1, 1.05, 1] : 1,
          }}
          transition={{
            duration: 3,
            repeat: isListening ? Infinity : 0,
            ease: "easeInOut",
          }}
        >
          {/* Outer glow - largest */}
          <motion.div
            className="absolute inset-0 rounded-full bg-gradient-to-br from-blue-400 via-cyan-400 to-blue-500 blur-3xl opacity-40"
            animate={{
              scale: isListening ? [1, 1.3, 1] : 1,
              opacity: isListening ? [0.4, 0.6, 0.4] : 0.2,
            }}
            transition={{
              duration: 2.5,
              repeat: isListening ? Infinity : 0,
              ease: "easeInOut",
            }}
          />

          {/* Middle glow */}
          <motion.div
            className="absolute inset-6 rounded-full bg-gradient-to-br from-blue-400 via-cyan-400 to-blue-500 blur-2xl opacity-60"
            animate={{
              scale: isListening ? [1.1, 1, 1.1] : 1,
              opacity: isListening ? [0.6, 0.8, 0.6] : 0.3,
            }}
            transition={{
              duration: 2,
              repeat: isListening ? Infinity : 0,
              ease: "easeInOut",
            }}
          />

          {/* Inner glow */}
          <motion.div
            className="absolute inset-12 rounded-full bg-gradient-to-br from-blue-300 via-cyan-300 to-blue-400 blur-xl opacity-80"
            animate={{
              scale: isListening ? [1, 1.15, 1] : 1,
            }}
            transition={{
              duration: 1.8,
              repeat: isListening ? Infinity : 0,
              ease: "easeInOut",
            }}
          />

          {/* Core orb */}
          <div className="absolute inset-16 rounded-full bg-gradient-to-br from-blue-400 via-cyan-300 to-blue-500 shadow-2xl">
            <motion.div
              className="absolute inset-0 rounded-full bg-gradient-to-tr from-white/40 via-white/10 to-transparent"
              animate={{
                rotate: isListening ? [0, 360] : 0,
              }}
              transition={{
                duration: 15,
                repeat: isListening ? Infinity : 0,
                ease: "linear",
              }}
            />
            <div className="absolute top-6 left-6 w-8 h-8 rounded-full bg-white/50 blur-md" />
          </div>

          {/* Pulsing rings */}
          {isListening && (
            <>
              <motion.div
                className="absolute inset-0 rounded-full border-2 border-cyan-300/30"
                animate={{ scale: [1, 1.5], opacity: [0.5, 0] }}
                transition={{ duration: 2, repeat: Infinity, ease: "easeOut" }}
              />
              <motion.div
                className="absolute inset-0 rounded-full border-2 border-blue-300/30"
                animate={{ scale: [1, 1.5], opacity: [0.5, 0] }}
                transition={{ duration: 2, repeat: Infinity, ease: "easeOut", delay: 1 }}
              />
            </>
          )}
        </motion.div>
      </div>

      {/* Status Text */}
      <div className="text-center space-y-2 mb-8">
        <motion.h2
          className="text-2xl font-semibold text-white"
          animate={{ opacity: isListening ? [1, 0.8, 1] : 1 }}
          transition={{ duration: 2, repeat: isListening ? Infinity : 0, ease: "easeInOut" }}
        >
          {getStatusText()}
        </motion.h2>
        <p className="text-lg text-white/70">
          {voiceState === "idle" ? "What's on your mind?" : ""}
        </p>
      </div>

      {/* Transcriptions */}
      {(userText || agentText) && (
        <div className="text-center space-y-2 mb-8 max-w-md">
          {userText && (
            <motion.p
              className="text-sm text-white/60"
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
            >
              You: "{userText}"
            </motion.p>
          )}
          {agentText && (
            <motion.p
              className="text-sm text-cyan-300"
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
            >
              Assistant: "{agentText}"
            </motion.p>
          )}
        </div>
      )}

      {/* Control Buttons */}
      <div className="flex items-center gap-6">
        {/* Close/Disconnect Button */}
        <motion.button
          onClick={disconnectVoice}
          className="w-14 h-14 rounded-full bg-white/10 backdrop-blur-md border border-white/20 flex items-center justify-center hover:bg-white/20 transition-colors"
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.95 }}
        >
          <X className="w-6 h-6 text-white" />
        </motion.button>

        {/* Microphone Button */}
        <motion.button
          onClick={handleMicClick}
          disabled={voiceState === "connecting"}
          className={cn(
            "w-14 h-14 rounded-full backdrop-blur-md border flex items-center justify-center transition-all",
            isConnected
              ? "bg-white/10 border-white/20 hover:bg-white/20"
              : "bg-red-500/20 border-red-500/30 hover:bg-red-500/30"
          )}
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.95 }}
          animate={{
            boxShadow: isConnected
              ? ["0 0 0 0 rgba(96, 165, 250, 0.4)", "0 0 0 10px rgba(96, 165, 250, 0)"]
              : "0 0 0 0 rgba(239, 68, 68, 0)",
          }}
          transition={{ duration: 1.5, repeat: isConnected ? Infinity : 0, ease: "easeOut" }}
        >
          {voiceState === "connecting" ? (
            <motion.div
              className="w-6 h-6 border-2 border-white border-t-transparent rounded-full"
              animate={{ rotate: 360 }}
              transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
            />
          ) : isConnected ? (
            <Mic className="w-6 h-6 text-white" />
          ) : (
            <MicOff className="w-6 h-6 text-red-300" />
          )}
        </motion.button>
      </div>

      {/* Listening indicator dots */}
      {isListening && (
        <div className="flex gap-2 mt-8">
          {[0, 1, 2].map((i) => (
            <motion.div
              key={i}
              className="w-2 h-2 rounded-full bg-cyan-400"
              animate={{ scale: [1, 1.5, 1], opacity: [0.5, 1, 0.5] }}
              transition={{ duration: 1.5, repeat: Infinity, delay: i * 0.2, ease: "easeInOut" }}
            />
          ))}
        </div>
      )}

      {/* Error message */}
      {error && (
        <p className="mt-4 text-red-400 text-sm">{error}</p>
      )}
    </div>
  );

  // Wrap with LiveKitRoom when connected
  if (isConnected) {
    return (
      <LiveKitRoom
        token={connectionInfo.token}
        serverUrl={connectionInfo.livekitUrl}
        connect={true}
        audio={true}
        video={false}
        onDisconnected={disconnectVoice}
      >
        <LiveKitVoiceHandler
          onStateChange={setVoiceState}
          onTranscription={handleTranscription}
          onUserTranscriptionComplete={onTranscriptionComplete}
        />
        {content}
      </LiveKitRoom>
    );
  }

  return content;
}
