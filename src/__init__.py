"""News Interpretation Portal — pipeline stages.

Each stage module exposes run(config: dict, **inputs) -> dict, matching the contract
used by the firm's MF ranking models so this slots into the shared orchestrator later.
"""
