# Data storage policy

Shipped browser inputs and evaluation artifacts use UTF-8 JSON with compact
serialization so they remain inspectable, cacheable, and portable on static
hosting. SHA-256 sidecars accompany generated candidate artifacts.

Large numeric matrices remain JSON when they are consumed directly by the
browser or by the existing Python/Node validation tools. They should not be
duplicated into a second format solely for convenience. Static hosting may
apply HTTP gzip or Brotli compression; no runtime database is required.

New generated data should:

- use JSON (or newline-delimited JSON for append-only logs) when practical;
- avoid embedding duplicate copies of the same candidate bank;
- record schema/version, settings, generation scale, and provenance in-band;
- include a checksum for artifacts used in scoring or evaluation;
- keep temporary simulation output outside `docs/data/` and `artifacts/`.

Season payloads keep the canonical candidate/filter index once. Alternate pool
variants carry only their recalculated strategies, fitted-evaluation metadata,
and provenance; they do not duplicate the candidate bank. This keeps a typical
season payload near 0.2–0.3 MB instead of several megabytes while preserving
the same browser controls and settings checks.
