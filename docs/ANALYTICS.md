# Optional UX measurement

`app.js` includes dormant instrumentation for the council's UX questions:

- `disclosure_toggle` records which native disclosure opened or closed.
- `scroll_depth` records the first visit to 25%, 50%, 75%, and 90% depth.
- `first_bracket_view` records seconds from page load until the bracket enters
  the viewport.

Events include only the event name, `desktop`/`mobile`, `new`/`returning`, the
selected season, and the current strategy. No URL, team choice, IP address, or
persistent identifier is added by the page.

The site sends nothing until the deployment defines an endpoint before loading
`app.js`:

```html
<script>
  window.BRACKET_LAB_ANALYTICS_ENDPOINT = 'https://example.invalid/collect';
</script>
<script src="app.js?v=..."></script>
```

The endpoint must accept a JSON `POST`. If it is absent, the instrumentation is
a no-op. Measurement is intentionally endpoint-neutral so the site can use a
privacy-preserving first-party collector or a hosted analytics service later.
