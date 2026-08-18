"""Domain exceptions for the recommend_hybrid Phase 2 foundation."""


class RecommendHybridError(ValueError):
    """Base error for validated recommendation foundation inputs."""


class ContractValidationError(RecommendHybridError):
    """A typed contract violates its declared schema."""


class AuthorityValidationError(RecommendHybridError):
    """Frozen prediction authority metadata is inconsistent."""


class PostCutoffDataError(RecommendHybridError):
    """An observation is not strictly earlier than its cutoff."""


class SensitiveFeatureError(RecommendHybridError):
    """A prohibited sensitive or outcome feature was supplied."""


class CatalogValidationError(RecommendHybridError):
    """The controlled action catalog is invalid."""


class ExpertLabelValidationError(RecommendHybridError):
    """A raw expert review cannot be normalized safely."""
