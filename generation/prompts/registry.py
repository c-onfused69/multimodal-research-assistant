from pathlib import Path


class PromptRegistry:
    def __init__(self):
        self.dir = Path(__file__).parent

    def get(self, name: str) -> str:
        path = self.dir / f"{name}.txt"
        return path.read_text()


registry = PromptRegistry()
