DENY = ["bypass kyc", "disable fraud", "steal"]

def allowed(text: str) -> bool:
    t = text.lower()
    return not any(bad in t for bad in DENY)
