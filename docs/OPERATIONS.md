# Operations, analytics, and delivery

This document covers the operational contract for the browser site, pool settings, generated data, deployment, and current site-review status.

## Analytics

The site has dormant UX instrumentation for the council review. It records only:

- disclosure open/close events
- scroll depth at 25%, 50%, 75%, and 90%
- time from page load to first bracket view
- device type (`desktop`/`mobile`)
- visitor type (`new`/`returning`)
- selected season and strategy

No URL, team choice, IP address, or persistent identifier is added by the page. If the deployment sets `window.BRACKET_LAB_ANALYTICS_ENDPOINT` before loading the script, a JSON `POST` is sent; otherwise the instrumentation is a no-op.

## Pool settings and interpretation

Supported pool settings are intentionally narrow:

- pool size: 10, 30, 50, 100 entries
- scoring: ESPN standard only

Every pool metric is tied to a settings contract. P(1st) is pool-size dependent, so a 10-entry and 100-entry result are different estimands even when the same bracket is used. Do not copy results across pool sizes or scoring presets.

Generated pool artifacts should record pool size, scoring ID, artifact hash, simulation count, trial count, generation timestamp, and provenance. Production metrics must use the canonical setting set rather than validation-only reduced simulations.

The 2027 prospective source cutoffs, candidate-generation settings, score/missing-data contract, re-freeze, and release-parity criteria are controlled by the [2027 roadmap](./PROJECT_ROADMAP_2027.md).

### Prospective 2027 selector and fallback

The frozen selector and historical source gate are run with:

```bash
python -m scripts.evaluate_prospective_2027_points
```

The command writes `artifacts/prospective_2027/selector_result.json` and its `.sha256` sidecar. Use `--out PATH` to write a separate audit report. Exit code `0` means the source and promotion gates passed; `1` means a gate failed; `2` means eligibility or evidence is indeterminate. A failed or indeterminate source/promotion gate is not a scored comparison and must not be presented as a candidate performance result.

The production payload builder loads only a hash-verified, current PASS report. It also checks the 2027 candidate artifact against the frozen settings, point-in-time inputs, field topology, provenance, and artifact checksum before using a non-baseline choice. When no usable result exists or any required gate does not pass, **Recommended** uses the explicit seed-only bracket and leaves P(1st)/EV undefined. A 2027-or-later payload without a Recommended entry is shown as unavailable; the browser never silently relabels P(1st) as the prospective recommendation. Pre-2027 legacy payloads retain their existing P(1st)-default compatibility. P(1st), EV, champion, and filter strategies remain separate user choices. Do not replace the 2027 fallback with another strategy or describe it as a v4 promotion.

To regenerate the payload after a valid release artifact is available:

```bash
python -m scripts.build_ui_payload
```

Verify the reported selector status and compare all Recommended round picks with the frozen artifact before publishing. The roadmap defines the full source capture, checksum, replay, and site-parity evidence required for the 2027 release.

## Data storage

The project favors compact UTF-8 JSON and newline-delimited JSON where append-only logs are needed. Generated artifacts should avoid duplicating the same candidate bank across multiple files.

Rules:

- prefer JSON for browser and validation consumption
- keep checksums for evaluation artifacts
- record schema/version and provenance in-band
- keep large temporary simulation output outside `docs/data/` and `artifacts/`
- avoid a second format when the JSON is already the effective source of truth

## Static delivery

The browser payloads are compact JSON and should be served from a static host with Brotli or gzip enabled. Modern CDNs negotiate encoding automatically from `Accept-Encoding`.

Do not commit precompressed `.gz` files. The CDN should set `Content-Encoding` itself. The project also uses a shared `DATA_V` version query parameter so that versioned payloads can remain safely cacheable.

Current production payloads are ~0.2–0.3 MB per season uncompressed; Brotli usually reduces that by ~70–85%.

## Site review status

The site review identified several important gaps and a few delivered fixes:

- delivered: per-game win probabilities on the board
- delivered: champion callout and title-probability display
- delivered: track-record realisation display for historical seasons
- missing: direct pool settings customization beyond the supported preset set
- missing: lock-one-game then re-optimize flow
- missing: multi-entry support
- open: source-date and play-in metadata contract for future seasons

## Working policy

- keep root docs minimal and canonical
- keep historical notes in [archive/README.md](./archive/README.md)
- treat archived notes as historical context, not current project truth
- use concise artifact records for evidence and audit history
