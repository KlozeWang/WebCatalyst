---
name: critic
description: Use the `critique` tool to get structured visual feedback on any rendered artifact (screenshot + task prompt → list of concrete issues + suggestions). Activate when the user asks for "feedback / review / critique / evaluate" on something you've built, or when you want a second opinion before declaring a build done.
---

# critic

The `critique` tool is your external judge. It runs an independent vision
model call against a screenshot + the original task prompt and returns a
structured list of observed problems with concrete fixes. Use it whenever
you need honest feedback that isn't colored by your own reasoning context.

**There is no score.** The tool returns issues, severities, and a binary
verdict (`ship_it` vs `needs_revision`). Use the issues — not a number — to
decide what to change next.

## When to call

- After Playwright captures a screenshot of an HTML artifact (always pair
  with `html-artisan`).
- When the user asks "is this any good?" / "review this" / "critique".
- Before declaring any visual work done — get feedback first.

## How to call

```
critique({
  taskPrompt:     "<verbatim user request>",
  screenshotPath: "/abs/path/to/shot.png",
  artifactPath:   "/abs/path/to/index.html",   // optional but recommended
  focus:          ["hero section", "typography hierarchy"], // optional
})
```

Returns:
```json
{
  "summary": "Layout is on the right track but the hero is underweight and the palette feels generic.",
  "issues": [
    {
      "area": "hero",
      "severity": "major",
      "observation": "Hero heading reads ~32px at 1440 width; subhead is the same weight, so there is no visual hierarchy.",
      "suggestion": "Bump heading to 56-64px bold, drop subhead to 18px regular, add 24px top padding."
    },
    {
      "area": "color",
      "severity": "minor",
      "observation": "The blue-to-purple gradient is the Tailwind default; nothing about it ties to the brand.",
      "suggestion": "Replace with a single accent (#FF6B35) on a near-black background; reserve gradient for the CTA only."
    },
    {
      "area": "cta",
      "severity": "minor",
      "observation": "Primary button has no hover state, looks flat.",
      "suggestion": "Add hover: lift 2px, deepen background by ~10%."
    }
  ],
  "wellDone": [
    "Section spacing is consistent at 96px vertical rhythm.",
    "Footer hierarchy is clean."
  ],
  "verdict": "needs_revision"
}
```

## How to use the result

1. **verdict = `ship_it`** → tell the user "done" with the path. Include a
   one-line summary of what the judge liked. Stop the loop.
2. **verdict = `needs_revision`** →
   - Show the user a compact bullet list: each `major` issue, then top
     2 `minor` issues. Omit `nit` unless they ask. Include the summary.
   - Apply the suggestions yourself (don't ask permission for each one
     unless they're destructive). Treat `wellDone` items as off-limits —
     don't undo what already works.
   - Re-screenshot and re-critique.
3. **Loop control.** Stop iterating when ANY of these is true:
   - `verdict === "ship_it"`, OR
   - no `major` issues remain (minors/nits alone are not worth another
     full loop unless the user explicitly wants polish), OR
   - you've done 5 iterations — surface the best version and the remaining
     issues honestly. Don't pretend something passed when it didn't.
4. **Never** silently swallow the feedback. The user should always see at
   least the summary + any remaining major issues.

## Reading severities

- **major** — must fix. Will be obvious to any user.
- **minor** — should fix before claiming "done". Polish-level.
- **nit** — only address if budget allows; safe to skip for v1.

If everything is `nit`, you're done. If there's a single `major`, keep
looping.

## Anti-patterns

- ❌ Calling `critique` on a screenshot you took before the page finished
  loading. Always wait for network idle / a known DOM element first.
- ❌ Passing a partial / cropped screenshot. Use a full-page screenshot
  unless the artifact is intentionally above-the-fold only.
- ❌ Pasting the user's prompt with your own paraphrase. Use the original
  text verbatim — paraphrasing shifts what the judge is comparing against.
- ❌ Asking the user to evaluate when you can call `critique` yourself.
- ❌ Looking for a "score" in the output and getting confused when there
  isn't one. There isn't, by design. Read the issues.
- ❌ Asking the judge for "a score out of 10" via `focus`. The schema
  doesn't have a score field; the judge will ignore the request.

