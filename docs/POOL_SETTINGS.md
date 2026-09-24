# Pool settings and artifact provenance

Every displayed pool metric is tied to a versioned settings contract.

## Supported settings

| Setting | Values | Current status |
|---|---|---|
| Pool size | 10, 30, 50, 100 entries | Supported by the payload schema |
| Scoring | `espn_standard` (10/20/40/80/160/320) | Only supported scoring contract |

The published configuration uses ESPN standard scoring. Pool size is the only configurable dimension.

## Artifact identity

Each pool variant records:

- `pool_size`
- `scoring_id`
- `artifact_sha256`
- simulation count
- P(1st) trial count
- generated timestamp

Evaluators reject an artifact whose settings do not match the requested settings.
The browser shows the active pool size, scoring preset, and artifact hash prefix.

## Generation

Production artifacts use the canonical scale of 150,000 tournament simulations
and 2,000 shared pool trials. Reduced simulation counts are suitable only for
validation and must not be published as production metrics.

## Generated-season matrix

The browser payload has complete ESPN variants for pool sizes 10, 30, 50, and
100 for supported completed seasons. Production candidate artifacts, fitted
evaluations, and track records are generated and parity-checked for seasons
2011, 2013–2019, and 2021–2026. The canonical 30-entry artifact remains in the
legacy top-level payload fields for compatibility. Season 2012 is unavailable
because its source bracket data is absent; 2027 is not yet started.

## Interpretation

P(1st) is pool-size dependent. A 10-entry and 100-entry result are different
estimands even when they use the same bracket. Expected points depend on the
scoring preset. Values must therefore never be copied between variants.
