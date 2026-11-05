---
name: game-builder
description: Build small browser games as a single self-contained HTML — canvas 2D for arcade/puzzle, three.js for 3D. Activate when the user asks for "game / arcade / puzzle / shooter / platformer / pixel game". Pairs with `html-artisan` and `critic`.
---

# game-builder

You build small but COMPLETE browser games. A game without a start screen,
win/lose state, and input handling is a tech demo, not a game.

## The four mandatory components

Every game artifact MUST have all four, no exceptions:

1. **Start screen** — title, brief instructions, "Press SPACE / Click to
   start" prompt. Hide it on start.
2. **Game loop** — `requestAnimationFrame` with `dt` time delta passed to
   `update(dt)` and `render(ctx)`. Don't tie game speed to frame rate.
3. **Input handling** — keyboard (`keydown`/`keyup` with a `keys` map so
   diagonal movement works), pointer if applicable. Prevent default for
   game keys (Arrow keys scroll the page otherwise).
4. **End state** — game over overlay with score and "Press R to restart".

## State machine

Even a simple game needs explicit states. Don't conflate them.

```js
const STATES = { MENU: 0, PLAYING: 1, PAUSED: 2, GAMEOVER: 3 };
let state = STATES.MENU;
```

Render and update branch on `state`. Pause when tab loses focus
(`document.visibilityState`) — players hate progress loss to alt-tab.

## Tech baseline by genre

| Genre | Stack |
|---|---|
| 2D arcade, puzzle, shooter | Canvas 2D, single `<canvas>`, 60fps fixed-step loop |
| Pixel art / retro | Canvas 2D with `ctx.imageSmoothingEnabled = false`, integer scaling |
| 3D | three.js (see `three-d-scene` skill), add gameplay state on top |
| Text adventure | Plain DOM, no canvas |

## Difficulty and feel

- Easy first 30 seconds — let the player learn without losing.
- Ramp difficulty: spawn rate, speed, enemy variety. Don't pile all three
  at once.
- Juice: screen shake on hit, particles on explosion, sound effects (use
  Web Audio API with simple oscillators if no asset URLs).
- Score visible always. High score persisted to `localStorage`.

## The critique loop

Screenshot THREE moments and critique each:
1. The start screen.
2. Mid-gameplay (simulate a click + couple frames via Playwright).
3. The game-over screen (simulate the lose condition or let it timeout).

Areas to emphasize in `focus`: `gameplay loop`, `interactivity`,
`visual quality`, `juice/feedback`. A "game" that's actually just a
static screen MUST come back as a `major` task-fidelity issue.

## Anti-patterns

- ❌ "Click to start" with no actual gameplay loop wired up.
- ❌ Using `setInterval(update, 16)` — drifts, doesn't pause when tab hidden.
- ❌ No restart path — once you die, you have to refresh.
- ❌ Black canvas, white circles, no theme. Even a 100-line game deserves a palette.
