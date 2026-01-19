class UserMemory:
    def __init__(self):
        self.summary = ""
    
    def update(self, new_info: str, llm):
        prompt = f"""
Existing summary:
{self.summary}
New interaction:
{new_info}

Update the summary conservatively.
Do not remove uncertainty.

"""
        
        self.summary = llm.invoke(prompt)
        