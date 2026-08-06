class Embedder:
    def __init__(self, *, enabled: bool = True) -> None:
        self._model = None
        if not enabled:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except ImportError:
            self._model = None

    @property
    def available(self) -> bool:
        return self._model is not None

    def encode(self, text: str) -> list[float] | None:
        if not self.available:
            return None

        return self._model.encode(text).tolist()
