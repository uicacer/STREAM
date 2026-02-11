# STREAM React Migration Plan

## Overview

Migrate from Streamlit to React to dramatically improve perceived latency and user experience. The goal is to create an engaging, responsive chat interface that feels as fast as ChatGPT or Claude.

---

## Why Migrate?

### Current Streamlit Limitations

| Issue | Impact |
|-------|--------|
| Full script re-execution on every interaction | ~500-800ms overhead |
| Server-side session state | Round-trip latency on every state change |
| Polling-based streaming | Delayed token display |
| No instant UI feedback | Users stare at blank screen |

### React Advantages

| Benefit | Impact |
|---------|--------|
| Instant UI response | <50ms from click to visual feedback |
| Client-side state (Zustand) | Zero round-trip for UI updates |
| Native SSE streaming | Tokens appear in real-time |
| Virtual DOM diffing | Only changed elements update |

### Expected Improvement

```
BEFORE (Streamlit):
[Click] -------- 1.5-4s waiting -------- [First token visible]
        ↑ blank/spinner, user feels stuck

AFTER (React):
[Click] [Instant typing dots] ---- 0.6-2.5s ---- [First token]
        ↑ immediate feedback, engaging animations
```

**Perceived latency reduction: 40-60%**

---

## Core Principles

### 1. Never Leave Users Waiting

- Show typing indicator IMMEDIATELY on submit
- Animate dots/pulse to show activity
- Display routing info as it becomes available
- Progressive disclosure of response

### 2. Thinking ≠ Generating

**CRITICAL**: Only show "Thinking" for actual reasoning models:
- Claude Sonnet 4 with extended thinking
- OpenAI o1/o3 models
- Models that return `<thinking>` tags

For regular models, show "Generating..." not "Thinking..."

### 3. Show Reasoning Process (When Available)

For reasoning models, display the thinking process like Claude/ChatGPT:
- Collapsible "Thinking" section
- Show reasoning steps in real-time
- Distinguish thinking from final response

---

## Architecture

### Tech Stack

| Layer | Technology | Rationale |
|-------|------------|-----------|
| Build | Vite | Fast dev server, optimized builds |
| Framework | React 18 | Concurrent features, Suspense |
| Language | TypeScript | Type safety, better DX |
| State | Zustand | Lightweight, no boilerplate |
| Styling | Tailwind CSS | Rapid development, small bundle |
| HTTP | Native fetch + SSE | No axios overhead |

### What is Zustand?

**Zustand** (German for "state") is a lightweight state management library for React (~1KB). It's much simpler than Redux:

```typescript
// Create a store in 3 lines
import { create } from 'zustand'

const useStore = create((set) => ({
  messages: [],
  addMessage: (msg) => set((state) => ({ messages: [...state.messages, msg] })),
}))

// Use anywhere in React
const messages = useStore((state) => state.messages)
```

**Why Zustand over Redux?**
- No boilerplate (no actions, reducers, providers)
- ~1KB bundle size (Redux is ~7KB)
- Works outside React components too
- Perfect for chat apps

### Project Structure

```
frontends/react/
├── package.json
├── vite.config.ts
├── tsconfig.json
├── index.html
├── public/
│   └── favicon.ico
├── src/
│   ├── main.tsx                    # Entry point
│   ├── App.tsx                     # Root component
│   │
│   ├── api/
│   │   ├── client.ts               # Base HTTP client
│   │   └── stream.ts               # SSE streaming client
│   │
│   ├── components/
│   │   ├── chat/
│   │   │   ├── ChatContainer.tsx   # Main chat area
│   │   │   ├── MessageList.tsx     # Scrollable message list
│   │   │   ├── Message.tsx         # Single message bubble
│   │   │   ├── ThinkingBlock.tsx   # Reasoning display (collapsible)
│   │   │   ├── TypingIndicator.tsx # Animated dots
│   │   │   └── StreamingText.tsx   # Token-by-token display
│   │   │
│   │   ├── input/
│   │   │   ├── ChatInput.tsx       # Text input + submit
│   │   │   └── SendButton.tsx      # Animated send button
│   │   │
│   │   ├── sidebar/
│   │   │   ├── Sidebar.tsx         # Main sidebar
│   │   │   ├── TierSelector.tsx    # Auto/Local/Lakeshore/Cloud
│   │   │   ├── JudgeSelector.tsx   # Ollama-1b/3b/Haiku
│   │   │   ├── SessionStats.tsx    # Query count, cost
│   │   │   └── ExampleQueries.tsx  # Quick start buttons
│   │   │
│   │   └── common/
│   │       ├── StatusBadge.tsx     # Tier/complexity badge
│   │       ├── CostDisplay.tsx     # Cost formatting
│   │       └── Spinner.tsx         # Loading states
│   │
│   ├── hooks/
│   │   ├── useChat.ts              # Chat logic + streaming
│   │   ├── useSSE.ts               # Server-Sent Events hook
│   │   └── useLocalStorage.ts      # Persist settings
│   │
│   ├── stores/
│   │   ├── chatStore.ts            # Messages, streaming state
│   │   ├── settingsStore.ts        # Tier, judge, temperature
│   │   └── statsStore.ts           # Session statistics
│   │
│   ├── types/
│   │   ├── message.ts              # Message types
│   │   ├── api.ts                  # API response types
│   │   └── settings.ts             # Settings types
│   │
│   └── styles/
│       ├── globals.css             # Tailwind imports
│       └── animations.css          # Custom animations
```

---

## Phase 1: Foundation (MVP)

### Goals
- Basic chat functionality
- SSE streaming working
- Instant UI feedback

### Files to Create

#### 1. Project Setup

**package.json**
```json
{
  "name": "stream-frontend",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "zustand": "^4.5.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "@vitejs/plugin-react": "^4.2.0",
    "autoprefixer": "^10.4.0",
    "postcss": "^8.4.0",
    "tailwindcss": "^3.4.0",
    "typescript": "^5.3.0",
    "vite": "^5.0.0"
  }
}
```

#### 2. SSE Streaming Client

**src/api/stream.ts**
```typescript
export interface StreamCallbacks {
  onToken: (token: string) => void;
  onMetadata: (meta: StreamMetadata) => void;
  onThinking: (thought: string) => void;  // For reasoning models
  onComplete: () => void;
  onError: (error: string) => void;
}

export interface StreamMetadata {
  tier: string;
  model: string;
  complexity?: string;
  cost?: number;
  isReasoning?: boolean;  // True for Claude Sonnet 4, o1, etc.
}

export async function streamChat(
  messages: Message[],
  settings: ChatSettings,
  callbacks: StreamCallbacks
): Promise<void> {
  const response = await fetch('/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: settings.tier,
      messages,
      stream: true,
      temperature: settings.temperature,
      judge_strategy: settings.judgeStrategy,
    }),
  });

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value);
    const lines = chunk.split('\n');

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const data = line.slice(6);
      if (data === '[DONE]') {
        callbacks.onComplete();
        return;
      }

      const parsed = JSON.parse(data);

      // Handle metadata
      if (parsed.stream_metadata) {
        callbacks.onMetadata(parsed.stream_metadata);
      }

      // Handle thinking (reasoning models)
      if (parsed.thinking) {
        callbacks.onThinking(parsed.thinking);
      }

      // Handle content
      const content = parsed.choices?.[0]?.delta?.content;
      if (content) {
        callbacks.onToken(content);
      }
    }
  }
}
```

#### 3. Chat Store (Zustand)

**src/stores/chatStore.ts**
```typescript
import { create } from 'zustand';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  thinking?: string;           // Reasoning process (if available)
  metadata?: {
    tier: string;
    model: string;
    complexity?: string;
    duration?: number;
    cost?: number;
  };
}

interface ChatState {
  messages: Message[];
  isStreaming: boolean;
  currentThinking: string;     // Live thinking stream
  currentResponse: string;     // Live response stream
  streamMetadata: StreamMetadata | null;

  // Actions
  addUserMessage: (content: string) => void;
  startStreaming: () => void;
  appendToken: (token: string) => void;
  appendThinking: (thought: string) => void;
  setMetadata: (meta: StreamMetadata) => void;
  finishStreaming: () => void;
  clearChat: () => void;
}

export const useChatStore = create<ChatState>((set, get) => ({
  messages: [],
  isStreaming: false,
  currentThinking: '',
  currentResponse: '',
  streamMetadata: null,

  addUserMessage: (content) => set((state) => ({
    messages: [...state.messages, {
      id: crypto.randomUUID(),
      role: 'user',
      content,
    }],
  })),

  startStreaming: () => set({
    isStreaming: true,
    currentResponse: '',
    currentThinking: '',
    streamMetadata: null,
  }),

  appendToken: (token) => set((state) => ({
    currentResponse: state.currentResponse + token,
  })),

  appendThinking: (thought) => set((state) => ({
    currentThinking: state.currentThinking + thought,
  })),

  setMetadata: (meta) => set({ streamMetadata: meta }),

  finishStreaming: () => set((state) => ({
    isStreaming: false,
    messages: [...state.messages, {
      id: crypto.randomUUID(),
      role: 'assistant',
      content: state.currentResponse,
      thinking: state.currentThinking || undefined,
      metadata: state.streamMetadata,
    }],
    currentResponse: '',
    currentThinking: '',
  })),

  clearChat: () => set({
    messages: [],
    currentResponse: '',
    currentThinking: '',
  }),
}));
```

#### 4. Typing Indicator (Engaging Animation)

**src/components/chat/TypingIndicator.tsx**
```typescript
export function TypingIndicator({ tier }: { tier?: string }) {
  const tierIcons: Record<string, string> = {
    local: '🏠',
    lakeshore: '🏫',
    cloud: '☁️',
    auto: '🤖',
  };

  return (
    <div className="flex items-center gap-2 text-gray-500">
      <span>{tierIcons[tier || 'auto']}</span>
      <div className="flex gap-1">
        <span className="animate-bounce [animation-delay:0ms]">●</span>
        <span className="animate-bounce [animation-delay:150ms]">●</span>
        <span className="animate-bounce [animation-delay:300ms]">●</span>
      </div>
      <span className="text-sm">Generating...</span>
    </div>
  );
}
```

#### 5. Thinking Block (For Reasoning Models)

**src/components/chat/ThinkingBlock.tsx**
```typescript
import { useState } from 'react';

interface ThinkingBlockProps {
  thinking: string;
  isStreaming?: boolean;
}

export function ThinkingBlock({ thinking, isStreaming }: ThinkingBlockProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!thinking) return null;

  return (
    <div className="mb-2 border-l-2 border-purple-400 pl-3">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-2 text-sm text-purple-600 hover:text-purple-800"
      >
        <span className={`transform transition-transform ${isExpanded ? 'rotate-90' : ''}`}>
          ▶
        </span>
        <span>
          {isStreaming ? '🧠 Thinking...' : '🧠 Thought process'}
        </span>
        {isStreaming && (
          <span className="animate-pulse">●</span>
        )}
      </button>

      {isExpanded && (
        <div className="mt-2 text-sm text-gray-600 bg-purple-50 p-3 rounded">
          <pre className="whitespace-pre-wrap font-sans">
            {thinking}
          </pre>
        </div>
      )}
    </div>
  );
}
```

---

## Phase 2: Feature Parity

### Goals
- Match all Streamlit features
- Tier/judge selection
- Session statistics
- Export functionality

### Components to Build

1. **Sidebar.tsx** - Settings panel
2. **TierSelector.tsx** - Auto/Local/Lakeshore/Cloud radio buttons
3. **JudgeSelector.tsx** - Ollama-1b/3b/Haiku (only in auto mode)
4. **SessionStats.tsx** - Query count, cost breakdown
5. **ExampleQueries.tsx** - Quick start buttons
6. **StatusBadge.tsx** - Show tier/complexity after response

---

## Phase 3: Polish & Engagement

### Goals
- Delightful micro-interactions
- Smooth animations
- Responsive design

### Engagement Features

1. **Skeleton loading** - Pulse animation while waiting
2. **Smooth scroll** - Auto-scroll to new messages
3. **Sound effects** (optional) - Subtle notification on complete
4. **Keyboard shortcuts** - Enter to send, Escape to cancel
5. **Message reactions** - Copy, regenerate buttons

### Animations (CSS)

```css
/* Typing dots bounce */
@keyframes bounce {
  0%, 80%, 100% { transform: translateY(0); }
  40% { transform: translateY(-6px); }
}

/* Token fade-in */
@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

.token-new {
  animation: fadeIn 0.1s ease-out;
}

/* Thinking pulse */
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}
```

---

## Phase 4: Integration & Deploy

### Docker Setup

**Dockerfile** (for React frontend)
```dockerfile
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
```

**nginx.conf**
```nginx
server {
    listen 80;
    root /usr/share/nginx/html;
    index index.html;

    # SPA routing
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Proxy API requests to middleware
    location /v1/ {
        proxy_pass http://middleware:5000;
        proxy_http_version 1.1;
        proxy_set_header Connection '';
        proxy_buffering off;  # Important for SSE
    }
}
```

### docker-compose.yml Addition

```yaml
services:
  # ... existing services ...

  react-frontend:
    build: ./frontends/react
    ports:
      - "3000:80"
    depends_on:
      - middleware
```

---

## Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1 | ~6-8 hours | Working chat with streaming |
| Phase 2 | ~4-6 hours | Full feature parity |
| Phase 3 | ~3-4 hours | Polish and animations |
| Phase 4 | ~2-3 hours | Docker integration |
| **Total** | **~15-21 hours** | Complete migration |

---

## Success Criteria

1. **Instant feedback** - UI responds in <50ms on user action
2. **Smooth streaming** - Tokens appear without jank
3. **Thinking display** - Reasoning shown for compatible models
4. **Feature parity** - All Streamlit features working
5. **User delight** - App feels responsive and engaging

---

## Reasoning Model Detection

To determine if a model supports "thinking":

```typescript
const REASONING_MODELS = [
  'claude-sonnet-4',      // Claude extended thinking
  'claude-opus-4',        // Claude extended thinking
  'o1',                   // OpenAI o1
  'o1-mini',              // OpenAI o1-mini
  'o3',                   // OpenAI o3
  'deepseek-r1',          // DeepSeek reasoning
];

function isReasoningModel(model: string): boolean {
  return REASONING_MODELS.some(rm =>
    model.toLowerCase().includes(rm.toLowerCase())
  );
}
```

When `isReasoningModel` returns true:
- Show "Thinking..." with animated pulse
- Display collapsible thinking block
- Parse `<thinking>` tags from response

When false:
- Show "Generating..." with typing dots
- No thinking block
- Direct response display
