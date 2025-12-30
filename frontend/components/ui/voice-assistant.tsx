"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Mic, X, MicOff } from "lucide-react";
import { cn } from "@/lib/utils";

export default function VoiceAssistant({ className }: { className?: string }) {
  const [isListening, setIsListening] = useState(true);
  const [userName] = useState("Armando");

  return (
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
            {/* Shine effect */}
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

            {/* Inner highlight */}
            <div className="absolute top-6 left-6 w-8 h-8 rounded-full bg-white/50 blur-md" />
          </div>

          {/* Pulsing rings */}
          {isListening && (
            <>
              <motion.div
                className="absolute inset-0 rounded-full border-2 border-cyan-300/30"
                animate={{
                  scale: [1, 1.5],
                  opacity: [0.5, 0],
                }}
                transition={{
                  duration: 2,
                  repeat: Infinity,
                  ease: "easeOut",
                }}
              />
              <motion.div
                className="absolute inset-0 rounded-full border-2 border-blue-300/30"
                animate={{
                  scale: [1, 1.5],
                  opacity: [0.5, 0],
                }}
                transition={{
                  duration: 2,
                  repeat: Infinity,
                  ease: "easeOut",
                  delay: 1,
                }}
              />
            </>
          )}
        </motion.div>
      </div>

      {/* Text */}
      <div className="text-center space-y-2 mb-16">
        <motion.h2
          className="text-2xl font-semibold text-white"
          animate={{
            opacity: isListening ? [1, 0.8, 1] : 1,
          }}
          transition={{
            duration: 2,
            repeat: isListening ? Infinity : 0,
            ease: "easeInOut",
          }}
        >
          {isListening ? `I'm listening, ${userName}...` : "Click to start listening"}
        </motion.h2>
        <p className="text-lg text-white/70">
          {isListening ? "What's on your mind?" : ""}
        </p>
      </div>

      {/* Control Buttons */}
      <div className="flex items-center gap-6">
        {/* Close Button */}
        <motion.button
          onClick={() => setIsListening(false)}
          className="w-14 h-14 rounded-full bg-white/10 backdrop-blur-md border border-white/20 flex items-center justify-center hover:bg-white/20 transition-colors"
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.95 }}
        >
          <X className="w-6 h-6 text-white" />
        </motion.button>

        {/* Microphone Button */}
        <motion.button
          onClick={() => setIsListening(!isListening)}
          className={cn(
            "w-14 h-14 rounded-full backdrop-blur-md border flex items-center justify-center transition-all",
            isListening
              ? "bg-white/10 border-white/20 hover:bg-white/20"
              : "bg-red-500/20 border-red-500/30 hover:bg-red-500/30"
          )}
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.95 }}
          animate={{
            boxShadow: isListening
              ? [
                  "0 0 0 0 rgba(96, 165, 250, 0.4)",
                  "0 0 0 10px rgba(96, 165, 250, 0)",
                ]
              : "0 0 0 0 rgba(239, 68, 68, 0)",
          }}
          transition={{
            duration: 1.5,
            repeat: isListening ? Infinity : 0,
            ease: "easeOut",
          }}
        >
          {isListening ? (
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
              animate={{
                scale: [1, 1.5, 1],
                opacity: [0.5, 1, 0.5],
              }}
              transition={{
                duration: 1.5,
                repeat: Infinity,
                delay: i * 0.2,
                ease: "easeInOut",
              }}
            />
          ))}
        </div>
      )}
    </div>
  );
}
