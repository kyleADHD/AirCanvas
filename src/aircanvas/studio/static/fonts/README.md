# Bundled fonts

Both files are the **latin variable subset** as served by Google Fonts, saved
here so the Studio draws correctly with the network unplugged — a local-first
desktop app must not depend on a CDN. 54 KB together.

| File | Family | Upstream | Licence |
|---|---|---|---|
| `space-grotesk-latin.woff2` | Space Grotesk | <https://fonts.google.com/specimen/Space+Grotesk> | SIL Open Font License 1.1 |
| `jetbrains-mono-latin.woff2` | JetBrains Mono | <https://fonts.google.com/specimen/JetBrains+Mono> | SIL Open Font License 1.1 |

The OFL permits redistribution of the font files, bundled with other software,
provided they are not sold on their own and the licence travels with them.
Neither file has been modified; each is the unaltered subset Google serves.

`styles.css` declares one `@font-face` per family with a `font-weight` range,
because each file is a variable font covering every weight the design uses
(Space Grotesk 400/500/600, JetBrains Mono 400/500).

To refresh them, request the same subset and copy the URLs out of the response:

```
curl -H 'User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120' \
  'https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap'
```
