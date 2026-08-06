class MockEmbedder:
    """Deterministic mock: on dim per known token, counts encodes"""

    def __init__(self) -> None:
        self.calls: list[str] = []

    @property
    def available(self) -> bool:
        return True

    def encode(self, text: str) -> list[float]:
        self.calls.append(text)

        return [float(hash(w) % 100) for w in text.lower().split()] or [0.0]
