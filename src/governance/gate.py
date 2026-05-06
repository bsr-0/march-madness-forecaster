"""Minimal governance gate placeholder."""


class GovernanceGate:
    """Compatibility shim for the production pipeline."""

    def allow(self, *_args, **_kwargs) -> bool:
        return True
