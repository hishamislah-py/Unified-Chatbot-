"use client";

import { useState, useEffect, useMemo } from "react";
import { motion } from "framer-motion";
import { Send, User, Briefcase, Headphones } from "lucide-react";
import { cn } from "@/lib/utils";

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

export default function AIChatCard({ className }: { className?: string }) {
  const [activeAgent, setActiveAgent] = useState<AgentType>("personal");
  const [messages, setMessages] = useState<Record<AgentType, { sender: "ai" | "user"; text: string }[]>>({
    personal: [{ sender: "ai", text: agents[0].greeting }],
    hr: [{ sender: "ai", text: agents[1].greeting }],
    it: [{ sender: "ai", text: agents[2].greeting }],
  });
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
  }, []);

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

  const handleSend = () => {
    if (!input.trim()) return;

    setMessages({
      ...messages,
      [activeAgent]: [...currentMessages, { sender: "user", text: input }],
    });
    setInput("");
    setIsTyping(true);

    // Simulate AI response
    setTimeout(() => {
      setMessages((prev) => ({
        ...prev,
        [activeAgent]: [
          ...prev[activeAgent],
          { sender: "ai", text: `🤖 This is a response from ${currentAgent.name}.` },
        ],
      }));
      setIsTyping(false);
    }, 1200);
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

        {/* Glowing Orb */}
        <div className="absolute top-16 left-1/2 -translate-x-1/2 z-20">
          <motion.div
            className="relative w-24 h-24"
            animate={{
              scale: [1, 1.1, 1],
            }}
            transition={{
              duration: 3,
              repeat: Infinity,
              ease: "easeInOut",
            }}
          >
            {/* Outer glow */}
            <motion.div
              className={cn(
                "absolute inset-0 rounded-full blur-xl opacity-60 bg-gradient-to-br",
                currentAgent.color
              )}
              animate={{
                scale: [1, 1.2, 1],
                opacity: [0.6, 0.8, 0.6],
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                ease: "easeInOut",
              }}
            />

            {/* Middle glow */}
            <motion.div
              className={cn(
                "absolute inset-2 rounded-full blur-lg opacity-80 bg-gradient-to-br",
                currentAgent.color
              )}
              animate={{
                scale: [1.1, 1, 1.1],
              }}
              transition={{
                duration: 2.5,
                repeat: Infinity,
                ease: "easeInOut",
              }}
            />

            {/* Core orb */}
            <div
              className={cn(
                "absolute inset-4 rounded-full bg-gradient-to-br shadow-2xl",
                currentAgent.color
              )}
            />
          </motion.div>
        </div>

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
          {currentMessages.map((msg, i) => (
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
              {msg.text}
            </motion.div>
          ))}

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
        </div>

        {/* Input */}
        <div className="flex items-center gap-2 p-3 border-t border-white/10 relative z-10">
          <input
            className="flex-1 px-3 py-2 text-sm bg-black/50 rounded-lg border border-white/10 text-white focus:outline-none focus:ring-1 focus:ring-white/50 placeholder:text-white/40"
            placeholder="Type a message..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
          />
          <button
            onClick={handleSend}
            className="p-2 rounded-lg bg-white/10 hover:bg-white/20 transition-colors"
          >
            <Send className="w-4 h-4 text-white" />
          </button>
        </div>
      </div>
    </div>
  );
}
