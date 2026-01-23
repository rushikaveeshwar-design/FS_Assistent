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

