# TAU2-Voice: Voice-based Conversational Agent Evaluation

[![python](https://img.shields.io/badge/Python-3.12%2B-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![arXiv](http://img.shields.io/badge/cs.AI-arXiv%3A2506.07982-B31B1B.svg?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2506.07982)

Voice-based extension of [TAU2-Bench](https://arxiv.org/abs/2506.07982) for evaluating real-time voice agents in customer service scenarios.

## Overview

TAU2-Voice extends the TAU2 framework to evaluate voice-based conversational agents using real-time audio interactions. Unlike text-based evaluation, voice introduces unique challenges including acoustic ambiguity, role confusion, and authentication vulnerabilities.

### Supported Models

- **OpenAI Realtime API** (`gpt-realtime-2025-08-28`, `gpt-realtime-mini-2025-10-06`)
- **Qwen3-Omni** (via vLLM server)
- **Gemini Live API** (`gemini-2.5-flash-native-audio-preview-12-2025`)

## Performance Comparison

Voice-based evaluation shows significant performance degradation compared to text-based evaluation:

| Model | Retail | Airline | Telecom |
|-------|--------|---------|---------|
| **Text-based Baselines** | | | |
| GPT-4o-2024-11-20 | 67.3 | 46.9 | 24.1 |
| GPT-4.1 | 70.2 | 53.0 | 38.9 |
| **Voice-based Baselines** | | | |
| gpt-realtime | **43.9** | **40.0** | **0.088** |
| Qwen3-Omni-30B-A3B-Instruct | - | **30.6** | **0.00** |
| gpt-realtime-mini | **13.2** | **18.0** | **0.00** |

**Key Observations:**
- Voice-based agents show **30-40% performance drop** in Retail and Airline domains
- **Near-zero performance** in complex multi-turn Telecom domain
- Challenges include acoustic ambiguity, role confusion, and difficulty maintaining conversation context

## Installation

1. Clone the repository:
```bash
git clone https://github.com/channel-io/ch-voice-tau.git
cd ch-voice-tau
```

2. Install dependencies:
```bash
pip install -e .
```

3. Set up API keys:
```bash
export OPENAI_API_KEY="your-openai-key"
export GOOGLE_API_KEY="your-google-key"  # For Gemini
```

## Quick Start

### Running Voice-based Evaluation

```bash
python -m tau2_voice.run
```

Edit `src/tau2_voice/run.py` to configure:
- `domain`: "airline", "retail", or "telecom"
- `assistant_model`: Model for the agent
- `user_model`: Model for the user simulator
- `num_tasks`: Number of tasks to evaluate
- `batch_size`: Parallel task execution

### Example: Run with Gemini Live

```python
# In src/tau2_voice/run.py
assistant_model = "gemini-2.5-flash-native-audio-preview-12-2025"
user_model = "gpt-realtime-2025-08-28"
```

### Example: Run with Qwen3-Omni

1. Start vLLM server:
```bash
cd /path/to/vllm-exp
bash run_qwen3_omni.sh
```

2. Run evaluation:
```python
# In src/tau2_voice/run.py
assistant_model = "qwen3_omni"
user_model = "gpt-realtime-2025-08-28"
```

## Architecture

### Event-driven Voice Pipeline

```
User Simulator (OpenAI Realtime)
    ↓ audio.chunk, transcript.update
Orchestrator
    ↓ audio.chunk, transcript.update, tool_call.request
Assistant Agent (Gemini/Qwen3/OpenAI)
    ↓ audio.chunk, transcript.update, tool_call.result
Environment (Tools & State)
```

### Key Components

- **Agents** (`src/tau2_voice/agent/`)
  - `RealtimeAgent`: OpenAI Realtime API
  - `Qwen3OmniAgent`: Qwen3-Omni via vLLM
  - `GeminiLiveAgent`: Google Gemini Live API
  - `UserAgent`: User simulator (OpenAI Realtime)

- **Orchestrator** (`src/tau2_voice/orchestrator/`)
  - Routes events between agents
  - Manages conversation flow
  - Records audio and transcripts
  - Evaluates task completion

- **Event Adapters** (`src/tau2_voice/adapters/`)
  - Convert between internal events and API-specific formats
  - Handle audio encoding/resampling
  - Prevent role confusion in user simulator

- **Audio Collection** (`src/tau2_voice/audio/`)
  - Records audio chunks to WAV files
  - Tracks transcripts and tool calls
  - Generates metadata JSON

## Recordings

Conversation recordings are saved in `data/recordings/<domain>/`:
- `{domain}_{task_id}_{timestamp}.wav`: Audio recording
- `{domain}_{task_id}_{timestamp}.json`: Metadata (transcripts, tool calls, success, reward)

## Key Features

### 1. Multi-modal Agent Support
Seamlessly integrate text-to-speech, speech-to-text, and native audio models.

### 2. User Simulator Guardrails
Prevents role confusion with:
- Deterministic customer opening from scenario
- Detection of agent-like phrases ("How can I help you", "Let me check that")
- Automatic retry on role drift (max 2 attempts)

### 3. Audio Resampling
Automatic audio format conversion between different APIs:
- OpenAI Realtime: 24kHz PCM
- Gemini Live: 16kHz PCM
- Qwen3-Omni: WAV with header

### 4. Tool Call Recording
Full conversation context including:
- Tool call requests and responses
- Success/reward metrics
- Turn-by-turn transcripts

## Development

### Adding a New Agent

1. Create agent class in `src/tau2_voice/agent/`:
```python
from tau2_voice.agent.base import BaseAgent

class MyAgent(BaseAgent):
    async def connect(self): ...
    async def disconnect(self): ...
    async def publish(self, event: Event): ...
    async def subscribe(self) -> AsyncGenerator[Event, None]: ...
```

2. Register in `src/tau2_voice/registry.py`:
```python
registry.register_agent(MyAgent, "my_agent")
```

3. Update model selection in `src/tau2_voice/run.py`:
```python
if assistant_model.startswith("my_model"):
    agent_name = "my_agent"
```

## Known Issues

### Voice-specific Challenges

1. **Role Confusion**: User simulator may adopt agent role due to audio feedback loops
   - Mitigated with guardrails and conversation.item.create injection

2. **Acoustic Ambiguity**: Spelling of IDs, codes, phone numbers prone to errors
   - Models may mishear "AA1234" as "A1234" or "8A1234"

3. **VAD Instability**: Voice Activity Detection varies across providers
   - Gemini Live: Uses text-based turn-taking for reliability
   - OpenAI Realtime: Semantic VAD enabled by default

4. **Multi-turn Context**: Voice agents lose context faster than text agents
   - Especially problematic in Telecom domain (30+ turn conversations)

## Citation

```bibtex
@misc{barres2025tau2,
      title={$\tau^2$-Bench: Evaluating Conversational Agents in a Dual-Control Environment}, 
      author={Victor Barres and Honghua Dong and Soham Ray and Xujie Si and Karthik Narasimhan},
      year={2025},
      eprint={2506.07982},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.07982}, 
}
```

## License

See [LICENSE](LICENSE) file.
