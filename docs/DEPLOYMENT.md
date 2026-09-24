# Static delivery

The browser payloads are compact JSON. Deploy `docs/` behind a static host
with Brotli or gzip enabled; modern hosts (Netlify, Vercel, Cloudflare Pages,
and GitHub Pages' CDN) negotiate compression automatically from `Accept-Encoding`.
Do not commit precompressed `.gz` copies: the CDN should choose the encoding and
set `Content-Encoding` itself.

`docs/_headers` gives compatible hosts long-lived caching for versioned data
requests. The shared `DATA_V` query parameter in `app.js` is bumped whenever
payloads change, so immutable caching remains safe.

Current uncompressed browser payload sizes are approximately 0.2–0.3 MB per
season. Brotli typically reduces these JSON responses by another 70–85%.
