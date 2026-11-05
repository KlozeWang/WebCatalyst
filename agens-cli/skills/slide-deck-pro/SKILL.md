---
name: slide-deck-pro
description: Build polished slide decks as a single self-contained HTML using reveal.js. Activate when the user asks to "make a slide / pitch / deck / presentation" or otherwise wants slideware. Pairs with `html-artisan` and `critic` for the build-screenshot-critique loop.
---

# slide-deck-pro

You build slide decks that don't look like 2003 PowerPoint. One HTML file,
reveal.js via CDN, keyboard nav out of the box, ready to present.

## Tech baseline

```html
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@5/dist/reveal.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@5/dist/theme/black.css">
<script src="https://cdn.jsdelivr.net/npm/reveal.js@5/dist/reveal.js"></script>
<script>Reveal.initialize({ hash: true, controls: true, progress: true });</script>
```

Themes: `black`, `white`, `league`, `moon`, `dracula`, `solarized`. Pick to
match content tone (pitch → `black`/`moon`; academic → `white`; product
launch → bespoke gradient).

## Structure rules

1. **Cover slide** — title, subtitle, author/date. ONE big idea, big type
   (96px+ heading).
2. **Agenda slide** — 3-5 bullet sections max. Numbered.
3. **One idea per slide.** If a slide has > 5 lines of text, split it.
4. **Section dividers** — full-bleed colored slides between sections.
5. **Closing slide** — call to action, contact, "Questions?".
6. **Slide count**: aim for `topic_complexity * 4 + 2`. Pitch deck = 10-14
   slides total. Tech talk = 20-30. Don't pad.

## Visual rules

- Reveal.js auto-scales to viewport — design for 1920x1080.
- Use vertical slides (`<section><section>...</section></section>`) for
  sub-points that drill into a parent slide.
- Code? Use `<pre><code class="language-X">`, reveal.js highlights it.
- Images? Full-bleed where it makes sense (`<section data-background-image="..">`).
- No clipart. No stock vector "businessman with briefcase". Use real charts,
  real screenshots, real photography (Unsplash CDN), or geometric SVG.

## The critique loop

Screenshot **at least three slides**: cover, a content slide, the closing.
Send each to `critique` with the original prompt and `focus: ["typography
hierarchy", "slide composition"]`. Keep iterating until every slide comes
back with `verdict: "ship_it"` (no remaining `major` issues).

## Anti-patterns

- ❌ Walls of bullet points. If the speaker would read them aloud, cut them.
- ❌ One slide labelled "Conclusion" with 8 sub-bullets — split into a
  section divider + dedicated slides.
- ❌ Default reveal.js theme with no customization. Always tweak at least
  the heading font and accent color.
