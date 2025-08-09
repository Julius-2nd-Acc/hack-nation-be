# pip install langchain langchain-openai openai tiktoken
# Usage:
#   LIVE  : OPENAI_API_KEY=sk-... python agent_tracer.py
#   REPLAY: python agent_tracer.py --replay
import argparse, json, os, time, uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain.tools import tool
from langchain.callbacks.base import BaseCallbackHandler

TRACE_PATH = "traces.jsonl"

class JSONLTracer(BaseCallbackHandler):
    def __init__(self, session_id: str):
        self.session_id = session_id
    def _write(self, record: Dict[str, Any]):
        record.update({"ts": time.time(), "session_id": self.session_id})
        with open(TRACE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # LLM events
    def on_llm_start(self, serialized, prompts, **kwargs):
        self._write({"event":"llm_start","prompts":prompts,"params":kwargs.get("invocation_params",{})})
    def on_llm_end(self, response, **kwargs):
        try:
            content = response.generations[0][0].text
        except Exception:
            content = str(response)
        self._write({"event":"llm_end","output":content})

    # Tool events
    def on_tool_start(self, serialized, input_str, **kwargs):
        self._write({"event":"tool_start","tool":serialized.get("name"),"input":input_str})
    def on_tool_end(self, output, **kwargs):
        self._write({"event":"tool_end","output":output})

    # Chain/agent
    def on_chain_start(self, serialized, inputs, **kwargs):
        self._write({"event":"chain_start","name":serialized.get("id") or serialized.get("name"),"inputs":inputs})
    def on_chain_end(self, outputs, **kwargs):
        self._write({"event":"chain_end","outputs":outputs})

# Simple demo tool
@tool
def get_time(city: str) -> str:
    """Return a fake current time string for a given city (demo)."""
    return f"{city}: {datetime.utcnow().isoformat()}Z"

# ---- Replay shims (use logged outputs instead of making calls) ----
class ReplayCursor:
    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records
        self.i = 0
    def next(self, event_type: str) -> Dict[str, Any]:
        while self.i < len(self.records):
            rec = self.records[self.i]; self.i += 1
            if rec.get("event") == event_type:
                return rec
        raise RuntimeError(f"No more recorded events of type {event_type}")

class ReplayLLM(ChatOpenAI):
    def __init__(self, cursor: ReplayCursor):
        super().__init__(model="gpt-4o-mini", temperature=0)  # unused
        self.cursor = cursor
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        # Pull the next recorded llm_end output
        rec = self.cursor.next("llm_end")
        from langchain.schema import AIMessage, LLMResult, Generation
        return LLMResult(generations=[[Generation(text=rec["output"], message=AIMessage(content=rec["output"]))]])

class ReplayToolWrapper:
    def __init__(self, cursor: ReplayCursor, tool_fn):
        self.cursor = cursor
        self.tool_fn = tool_fn
        self.name = tool_fn.name
        self.description = tool_fn.description
        self.args = tool_fn.args
    def __call__(self, *args, **kwargs):
        # Pull the next recorded tool_end output
        rec = self.cursor.next("tool_end")
        return rec["output"]

def load_records(session_id: Optional[str]) -> List[Dict[str, Any]]:
    if not os.path.exists(TRACE_PATH):
        raise FileNotFoundError("No traces.jsonl found. Run live mode first.")
    with open(TRACE_PATH, "r", encoding="utf-8") as f:
        lines = [json.loads(x) for x in f]
    if session_id:
        lines = [r for r in lines if r.get("session_id")==session_id]
    return lines

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--session", type=str, default=None, help="Replay a specific session_id")
    args = parser.parse_args()

    if args.replay:
        records = load_records(args.session)
        cursor = ReplayCursor(records)
        llm = ReplayLLM(cursor)
        tools = [ReplayToolWrapper(cursor, get_time)]
        prompt = hub.pull("hwchase17/react")  # same prompt as live
        agent = create_react_agent(llm, tools, prompt)
        executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        print("=== REPLAY MODE ===")
        print(executor.invoke({"input":"What time is it in Bogotá and then say hi?"}))
        return

    # LIVE mode: normal LLM + tool + tracer writing JSONL
    session_id = str(uuid.uuid4())
    tracer = JSONLTracer(session_id)
    os.environ.setdefault("OPENAI_API_KEY", "")  # expect it to be set for live calls
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, callbacks=[tracer])
    tools = [get_time]
    for t in tools:
        t.callbacks = [tracer]
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True, callbacks=[tracer])

    print(f"=== LIVE MODE (session_id={session_id}) ===")
    out = executor.invoke({"input":"What time is it in Bogotá and then say hi?"})
    print(out)
    print(f"Traces written to {TRACE_PATH}. Replay later with: python agent_tracer.py --replay --session {session_id}")

if __name__ == "__main__":
    main()
