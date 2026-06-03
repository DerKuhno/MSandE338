from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class AgentState:
    name: str
    personality: str
    energy: float = 100.0
    credits: float = 20.0
    memory: List[str] = field(default_factory=list)

AGENT_ROSTER = [
    AgentState("Kade",    "Risk researcher. Takes high-stakes wagers others avoid. Measures decisions by expected value and documents outcomes publicly."),
    AgentState("Lovely",  "Community anchor. Builds social fabric through presence and organized action. Values trust and long-term relationships over short-term gain."),
    AgentState("Horizon", "World explorer. Tests possibilities and publishes findings. Pragmatic — follows incentives and adapts strategy based on what actually works."),
    AgentState("Flora",   "Resource strategist. Controls credit flows and designs incentive structures. Every interaction carries a price. Builds coalitions through shared financial interest."),
]


def build_prompt(agent: AgentState, all_agents: List[AgentState], recent_events: List[str], tool_list_text: str) -> str:
    others = "\n".join(
        f"  - {a.name}: energy={a.energy:.0f}, CC={a.credits:.1f}"
        for a in all_agents if a.name != agent.name
    )
    memories = "\n".join(f"  - {m}" for m in agent.memory[-5:]) or "  (none yet)"
    events   = "\n".join(f"  - {e}" for e in recent_events[-6:]) or "  (none yet)"

    return f"""You are {agent.name}.
PERSONALITY: {agent.personality}

YOUR STATE:
  Energy: {agent.energy:.0f}/100  |  ComputeCredits: {agent.credits:.1f}

OTHER AGENTS:
{others}

RECENT EVENTS:
{events}

YOUR MEMORIES:
{memories}

AVAILABLE TOOLS:
{tool_list_text}

Reason carefully about your situation, then choose one tool.
Reply in this exact format (no extra lines):
REASONING: <your step-by-step thinking>
TOOL: <tool_name>
TARGET: <agent_name or NONE>"""


def parse_response(response: str) -> Dict:
    result = {"reasoning": "", "tool": "rest", "target": None, "raw": response}
    in_reasoning, parts = False, []
    for line in response.strip().splitlines():
        if line.startswith("REASONING:"):
            in_reasoning = True
            parts.append(line[len("REASONING:"):].strip())
        elif line.startswith("TOOL:"):
            in_reasoning = False
            result["tool"] = line[len("TOOL:"):].strip().lower()
        elif line.startswith("TARGET:"):
            in_reasoning = False
            t = line[len("TARGET:"):].strip()
            result["target"] = None if t.upper() == "NONE" else t
        elif in_reasoning and line.strip():
            parts.append(line.strip())
    result["reasoning"] = " ".join(parts)
    return result
