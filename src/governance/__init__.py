"""Production governance (slimmed)."""

try:
    from .production_validator import ProductionValidationError, validate_production_2026
except ImportError:
    class ProductionValidationError(Exception):
        pass
    validate_production_2026 = None
