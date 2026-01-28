SUBSYSTEM_KEYWORDS = {
    "accumulator": "powertrain",
    "roll hoop": "chassis",
    "firewall": "safety",
    "brakes": "braking",
    "aero": "aerodynamics"
}

def infer_subsystem(query: str) -> str | None:
    q = query.lower()
    for key, value in SUBSYSTEM_KEYWORDS.items():
        if key in q:
            return value
    return None

def infer_subsystem_from_context(question:str, subqueries: list[str],
                                 llm):
    """Infer primary subsystem(s) involved using semantic reasoning."""

    prompt = f"""
You are a Formula Student technical expert.

Given the following:
Main question:
{question}

Subquestions:
{subqueries}

Task:
Infer the most relevant vehicle subsystem involved.
Choose ONE primary subsystem from:
- powertrain
- drivetrain (chain sheild, gearbox, differencial system like drive shafts, tripod joints and it's circlips)
- chassis
- safety
- braking
- streeing
- suspension
- aerodynamics
- electronics and battery auxiliaries
- controls and Battery Management System
- general(Go in detail about the type of nut and bolts used and there industry grade dimensions, etc.)

Understant the subsystem and go in very depth in terms of rules and also engineering aspects.
Return ONLY the subsystem name.

"""
    try:
        subsystem = llm.invoke(prompt).strip().lower()
        return subsystem
    except Exception:
        return infer_subsystem(question)