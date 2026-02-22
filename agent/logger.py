import os
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

LOG_LEVEL = os.getenv("FS_LOG_LEVEL", "INFO").upper()

LEVEL_ORDER = {
    "TRACE": 0,
    "DEBUG": 1,
    "INFO": 2,
    "ERROR": 3
}

LOG_DIR = Path("logs")
SESSION_DIR = LOG_DIR / "sessions"
LOG_DIR.mkdir(exist_ok=True)
SESSION_DIR.mkdir(exist_ok=True)

def _should_log(level):
    return LEVEL_ORDER[level] >= LEVEL_ORDER.get(LOG_LEVEL, 2)

def _write_jsonl(path, data):
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(data, default=str) + "\n")

def log_event(level, event, *, chat_id=None, node=None, meta=None):
    if not _should_log(level):
        return
    
    payload = {"timestamp": datetime.now(timezone.utc).isoformat(),
               "level": level,
               "event": event,
               "node": node,
               "meta": meta or {}}
    
    _write_jsonl(LOG_DIR / "global.jsonl", payload)

    if chat_id:
        _write_jsonl(SESSION_DIR / f"{chat_id}.jsonl", payload)

def log_exception(e, *, chat_id=None, node=None):
    log_event("ERROR", "exception", chat_id=chat_id,
              node=node, meta={"error": str(e),
                               "traceback": traceback.format_exc()})
    
def log_node_enter(name, chat_id):
    log_event("DEBUG", "node_enter", chat_id=chat_id, node=name)
    
def log_vision_event(chat_id, kind, detail):
    log_event("DEBUG", f"vision_{kind}",
              chat_id=chat_id, meta=detail)
    
def invoke_llm(llm, prompt, *, chat_id=None, node=None, image=None):
    t0 = time.time()

    try:
        if image is not None:
            results = llm.invoke(prompt, image=image)
        
        else:
            results = llm.invoke(prompt)
        
        latency = int((time.time() - t0)*1000)

        tokens = getattr(results, "usage", None)

        log_event("DEBUG", "llm_invoke", chat_id=chat_id,
                  meta={"node": node, "latency": latency,
                        "tokens": getattr(tokens, "total_tokens", None)})
        
        return results
    
    except Exception as e:
        log_exception(e, chat_id=chat_id, node=node)
        raise

def stream_llm(llm, prompt, *, chat_id=None, node=None):
    t0 = time.time()

    for token in llm.stream(prompt):
        yield token
    
    latency = int((time.time() - t0)*1000)
    log_event("DEBUG", "llm_stream_complete", chat_id=chat_id,
              meta={"node": node, "latency": latency})