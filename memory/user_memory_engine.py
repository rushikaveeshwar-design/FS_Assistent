from agent.logger import log_event

class UserMemoryEngine:
    def __init__(self, chat_store):
        self.chat_store = chat_store

    def load(self, chat_id):
        return self.chat_store.get_memory(chat_id) or ""
    
    def update(self, chat_id, llm, new_info):
        old = self.load(chat_id)

        prompt = f"""
You are maintaining a structured Formula Student project memory.

Existing memory:
{old}

New interaction:
{new_info}

Rules:
- Keep ONLY persistent engineering facts.
- Keep uncertainties.
- Detect contradictions.
- Update subsystem-specific notes.
- Do not hallucinate details.
- If information not explicitly stated, do not add it.

Return concise structured memory summary.
"""
        updated = llm.invoke(prompt)
        self.chat_store.update_memory(chat_id, updated)
        log_event("DEBUG", "memory_update",
                  chat_id=chat_id, meta={"old_len": len(old),
                                         "new_len": len(updated)})
        return updated