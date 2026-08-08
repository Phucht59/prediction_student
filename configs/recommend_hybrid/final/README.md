# Final recommendation configuration

These version-neutral filenames expose the exact configuration blobs used by
the scientifically released recommendation system:

- `recommendation.yaml` — frozen recommendation protocol/configuration;
- `actions.yaml` — canonical five-action definitions;
- `feature_contract.yaml` — pre-cutoff feature and leakage contract;
- `literature_sources.yaml` — source registry supporting the policy design;
- `llm_annotation_protocol.yaml` — external-review protocol.

The original versioned files remain in `configs/recommend_hybrid/` because the
final scientific manifests and historical audit trail were created against
those paths. The canonical copies in this directory preserve the exact blob
contents; they are naming aliases for release use, not a post-Panel-B retuning.
