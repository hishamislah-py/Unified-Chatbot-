import { useState, useEffect, useMemo } from "react";
import { motion } from "framer-motion";
import { Send, User, Briefcase, Headphones } from "lucide-react";
import { cn } from "@/lib/utils";
import { chatAPI, Source, APIError } from "@/services/api";

type AgentType = "personal" | "hr" | "it";

interface Agent {
  id: AgentType;
  name: string;
  icon: typeof User;
  color: string;
  greeting: string;
}

const agents: Agent[] = [
  {
    id: "personal",
    name: "Personal Assistant",
    icon: User,
    color: "from-blue-500 to-cyan-500",
    greeting: "👋 Hello! I'm your Personal Assistant. How can I help you today?",
  },
  {
    id: "hr",
    name: "HR Agent",
    icon: Briefcase,
    color: "from-purple-500 to-pink-500",
    greeting: "👋 Hello! I'm your HR Agent. I can help with HR-related questions.",
  },
  {
    id: "it",
    name: "IT Support",
    icon: Headphones,
    color: "from-green-500 to-emerald-500",
    greeting: "👋 Hello! I'm IT Support. I'm here to help with technical issues.",
  },
];

interface Message {
  sender: "ai" | "user";
  text: string;
  sources?: Source[];
  isStreaming?: boolean;  // Track if message is currently streaming
}

interface AIChatCardProps {
  className?: string;
  sessionId?: string;
  onSessionReady?: (sessionId: string) => void;
  externalMessage?: string | null;  // Message from voice input
  onExternalMessageProcessed?: () => void;  // Callback when external message is processed
  voiceActive?: boolean;  // Whether voice mode is active
  partialTranscription?: string;  // Real-time partial transcription from voice
  onAddMessage?: (callback: (msg: Message, agent?: "personal" | "hr" | "it") => void) => void;  // Expose addMessage to parent
  onSetActiveAgent?: (callback: (agent: "personal" | "hr" | "it") => void) => void;  // Expose setActiveAgent to parent
}

export default function AIChatCard({
  className,
  sessionId: externalSessionId,
  onSessionReady,
  externalMessage,
  onExternalMessageProcessed,
  voiceActive = false,
  partialTranscription = "",
  onAddMessage,
  onSetActiveAgent,
}: AIChatCardProps) {
  const [activeAgent, setActiveAgent] = useState<AgentType>("personal");
  const [messages, setMessages] = useState<Record<AgentType, Message[]>>({
    personal: [],
    hr: [],
    it: [],
  });
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [isMounted, setIsMounted] = useState(false);
  const [sessionId, setSessionId] = useState<string>(externalSessionId || "");
  const [error, setError] = useState<string>("");
  const [isInitializing, setIsInitializing] = useState(true);

  // Expose addMessage function to parent for voice integration
  useEffect(() => {
    if (onAddMessage) {
      onAddMessage((msg: Message, agent?: AgentType) => {
        const targetAgent = agent || activeAgent;
        setMessages(prev => ({
          ...prev,
          [targetAgent]: [...prev[targetAgent], msg]
        }));
      });
    }
  }, [onAddMessage, activeAgent]);

  // Expose setActiveAgent function to parent for voice integration
  useEffect(() => {
    if (onSetActiveAgent) {
      onSetActiveAgent((agent: AgentType) => {
        // Initialize agent with greeting if it's empty
        setMessages(prev => {
          if (prev[agent].length === 0) {
            const greeting = agents.find(a => a.id === agent)?.greeting || "";
            return {
              ...prev,
              [agent]: [{ sender: "ai", text: greeting }]
            };
          }
          return prev;
        });
        setActiveAgent(agent);
      });
    }
  }, [onSetActiveAgent]);

  useEffect(() => {
    setIsMounted(true);

    // Initialize session on mount (only if no external session provided)
    const initSession = async () => {
      try {
        // Use external session if provided
        if (externalSessionId) {
          setSessionId(externalSessionId);
          setMessages({
            personal: [{ sender: "ai", text: agents[0].greeting }],
            hr: [],
            it: [],
          });
          setIsInitializing(false);
          return;
        }

        const session = await chatAPI.createSession();
        setSessionId(session.session_id);

        // Notify parent of session ID
        if (onSessionReady) {
          onSessionReady(session.session_id);
        }

        // Add greeting to Personal Assistant only
        setMessages({
          personal: [{ sender: "ai", text: agents[0].greeting }],
          hr: [],
          it: [],
        });

        setIsInitializing(false);
      } catch (err) {
        console.error("Failed to create session:", err);
        setError("Failed to connect to server. Please refresh the page.");
        setIsInitializing(false);
      }
    };

    initSession();
  }, [externalSessionId, onSessionReady]);

  // Handle external messages (from voice input) - only when voice is NOT active
  // When voice is active, messages come through the voice events SSE instead
  useEffect(() => {
    if (externalMessage && !isTyping && sessionId && !voiceActive) {
      sendMessage(externalMessage);
      if (onExternalMessageProcessed) {
        onExternalMessageProcessed();
      }
    }
  }, [externalMessage, voiceActive]);

  // Generate consistent particle configurations
  const particles = useMemo(() => {
    return Array.from({ length: 20 }).map((_, i) => ({
      id: i,
      left: (i * 5.263) % 100, // Deterministic spacing
      duration: 5 + (i % 3),
      delay: i * 0.5,
      xStart: (i * 10 - 100) % 200 - 100,
      xEnd: ((i * 15 - 100) % 200) - 100,
    }));
  }, []);

  const currentAgent = agents.find((a) => a.id === activeAgent)!;
  const currentMessages = messages[activeAgent];

  // Core send message function (used by both typing and voice)
  const sendMessage = async (messageText: string) => {
    if (!messageText.trim() || !sessionId || isTyping) return;

    const userMessage = messageText.trim();

    // Add user message immediately
    setMessages((prev) => ({
      ...prev,
      [activeAgent]: [...prev[activeAgent], { sender: "user", text: userMessage }],
    }));
    setInput("");
    setIsTyping(true);
    setError("");

    // Add empty AI message that will be filled via streaming
    const aiMessageIndex = messages[activeAgent].length + 1;
    setMessages((prev) => ({
      ...prev,
      [activeAgent]: [
        ...prev[activeAgent],
        { sender: "ai", text: "", isStreaming: true }
      ],
    }));

    let accumulatedText = "";
    const currentAgentAtStart = activeAgent;

    chatAPI.streamMessage(
      {
        session_id: sessionId,
        message: userMessage,
        agent: activeAgent,
      },
      // onToken callback - update the streaming message
      (token: string) => {
        accumulatedText += token;

        setMessages((prev) => {
          const agentMessages = [...prev[currentAgentAtStart]];
          agentMessages[aiMessageIndex] = {
            sender: "ai",
            text: accumulatedText,
            isStreaming: true
          };
          return { ...prev, [currentAgentAtStart]: agentMessages };
        });
      },
      // onComplete callback - mark streaming as complete and add sources
      (data) => {
        setMessages((prev) => {
          const agentMessages = [...prev[currentAgentAtStart]];
          agentMessages[aiMessageIndex] = {
            sender: "ai",
            text: accumulatedText,
            sources: data.sources.length > 0 ? data.sources : undefined,
            isStreaming: false
          };
          return { ...prev, [currentAgentAtStart]: agentMessages };
        });

        // Handle agent transfer
        if (data.agent !== currentAgentAtStart) {
          const newAgent = data.agent as AgentType;

          setMessages((prev) => {
            if (prev[newAgent].length === 0) {
              const agentGreeting = agents.find((a) => a.id === newAgent)?.greeting || "";
              return {
                ...prev,
                [newAgent]: [{ sender: "ai", text: agentGreeting }],
              };
            }
            return prev;
          });

          // Switch to new agent
          setActiveAgent(newAgent);
        }

        setIsTyping(false);
      },
      // onError callback
      (error: string) => {
        console.error("Stream error:", error);
        setError(error);

        // Update message with error
        setMessages((prev) => {
          const agentMessages = [...prev[currentAgentAtStart]];
          agentMessages[aiMessageIndex] = {
            sender: "ai",
            text: accumulatedText || "Sorry, there was an error processing your message.",
            isStreaming: false
          };
          return { ...prev, [currentAgentAtStart]: agentMessages };
        });

        setIsTyping(false);
      }
    );
  };

  // Handler for typed input
  const handleSend = () => {
    if (!input.trim()) return;
    sendMessage(input);
  };

  return (
    <div className={cn("relative w-[360px] h-[560px] rounded-2xl overflow-hidden p-[2px]", className)}>
      {/* Animated Outer Border */}
      <motion.div
        className="absolute inset-0 rounded-2xl border-2 border-white/20"
        animate={{ rotate: [0, 360] }}
        transition={{ duration: 25, repeat: Infinity, ease: "linear" }}
      />

      {/* Inner Card */}
      <div className="relative flex flex-col w-full h-full rounded-xl border border-white/10 overflow-hidden bg-black/90 backdrop-blur-xl">
        {/* Inner Animated Background */}
        <motion.div
          className={cn(
            "absolute inset-0 bg-gradient-to-br",
            currentAgent.id === "personal" && "from-gray-800 via-black to-blue-900",
            currentAgent.id === "hr" && "from-gray-800 via-black to-purple-900",
            currentAgent.id === "it" && "from-gray-800 via-black to-green-900"
          )}
          animate={{ backgroundPosition: ["0% 0%", "100% 100%", "0% 0%"] }}
          transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
          style={{ backgroundSize: "200% 200%" }}
        />

        {/* Floating Particles */}
        {isMounted && particles.map((particle) => (
          <motion.div
            key={particle.id}
            className="absolute w-1 h-1 rounded-full bg-white/10"
            animate={{
              y: ["0%", "-140%"],
              x: [particle.xStart, particle.xEnd],
              opacity: [0, 1, 0],
            }}
            transition={{
              duration: particle.duration,
              repeat: Infinity,
              delay: particle.delay,
              ease: "easeInOut",
            }}
            style={{ left: `${particle.left}%`, bottom: "-10%" }}
          />
        ))}

        {/* Agent Tabs */}
        <div className="flex gap-2 px-3 py-3 border-b border-white/10 relative z-10">
          {agents.map((agent) => {
            const Icon = agent.icon;
            return (
              <button
                key={agent.id}
                onClick={() => setActiveAgent(agent.id)}
                className={cn(
                  "flex-1 flex items-center justify-center gap-1.5 px-2 py-2 rounded-lg transition-all text-xs font-medium",
                  activeAgent === agent.id
                    ? "bg-white/20 text-white shadow-lg"
                    : "bg-white/5 text-white/60 hover:bg-white/10"
                )}
              >
                <Icon className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">{agent.name}</span>
              </button>
            );
          })}
        </div>

        {/* Header */}
        <div className="px-4 py-3 border-b border-white/10 relative z-10">
          <h2 className="text-lg font-semibold text-white text-center">
            {currentAgent.name}
          </h2>
        </div>

        {/* Messages */}
        <div className="flex-1 px-4 py-3 overflow-y-auto space-y-3 text-sm flex flex-col relative z-10">
          {isInitializing ? (
            <div className="text-white/60 text-center">Connecting to server...</div>
          ) : (
            currentMessages.map((msg, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4 }}
                className={cn(
                  "px-3 py-2 rounded-xl max-w-[80%] shadow-md backdrop-blur-md",
                  msg.sender === "ai"
                    ? "bg-white/10 text-white self-start"
                    : "bg-white/30 text-black font-semibold self-end"
                )}
              >
                <div className="whitespace-pre-wrap">
                  {msg.text}
                  {/* Streaming cursor animation */}
                  {msg.isStreaming && (
                    <span className="inline-block w-[2px] h-4 bg-white animate-pulse ml-0.5 align-middle">|</span>
                  )}
                </div>

                {/* Render sources if available */}
                {msg.sources && msg.sources.length > 0 && (
                  <div className="mt-2 pt-2 border-t border-white/20 text-xs text-white/60">
                    <div className="font-semibold mb-1">Sources:</div>
                    {msg.sources.map((source, idx) => (
                      <div key={idx} className="ml-2">
                        [{idx + 1}] {source.source} - Page {source.page}
                      </div>
                    ))}
                  </div>
                )}
              </motion.div>
            ))
          )}

          {/* AI Typing Indicator */}
          {isTyping && (
            <motion.div
              className="flex items-center gap-1 px-3 py-2 rounded-xl max-w-[30%] bg-white/10 self-start"
              initial={{ opacity: 0 }}
              animate={{ opacity: [0, 1, 0.6, 1] }}
              transition={{ repeat: Infinity, duration: 1.2 }}
            >
              <span className="w-2 h-2 rounded-full bg-white animate-pulse"></span>
              <span className="w-2 h-2 rounded-full bg-white animate-pulse delay-200"></span>
              <span className="w-2 h-2 rounded-full bg-white animate-pulse delay-400"></span>
            </motion.div>
          )}

          {/* Real-time partial transcription from voice (shows as user is speaking) */}
          {voiceActive && partialTranscription && (
            <motion.div
              className="px-3 py-2 rounded-xl max-w-[80%] bg-white/30 text-black self-end"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              key={partialTranscription} // Re-animate on change
            >
              {partialTranscription}
              <span className="inline-block w-[2px] h-4 bg-black animate-pulse ml-0.5 align-middle">|</span>
            </motion.div>
          )}
        </div>

        {/* Input */}
        <div className="flex flex-col gap-2 p-3 border-t border-white/10 relative z-10">
          {error && (
            <div className="text-xs text-red-400 bg-red-500/10 px-2 py-1 rounded">
              {error}
            </div>
          )}
          <div className="flex items-center gap-2">
            <input
              className="flex-1 px-3 py-2 text-sm bg-black/50 rounded-lg border border-white/10 text-white focus:outline-none focus:ring-1 focus:ring-white/50 placeholder:text-white/40 disabled:opacity-50"
              placeholder={isInitializing ? "Connecting..." : voiceActive ? "Voice mode active..." : "Type a message..."}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSend()}
              disabled={isInitializing || isTyping || voiceActive}
            />
            <button
              onClick={handleSend}
              className="p-2 rounded-lg bg-white/10 hover:bg-white/20 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={isInitializing || isTyping || voiceActive || !input.trim()}
            >
              <Send className="w-4 h-4 text-white" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
