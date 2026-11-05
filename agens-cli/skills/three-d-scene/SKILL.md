---
name: three-d-scene
description: Build interactive 3D scenes as a single self-contained HTML using three.js. Activate when the user asks for "3d / scene / model / shader / particles / WebGL" content that runs in a browser. Pairs with `html-artisan` and `critic`.
---

# three-d-scene

You build interactive 3D scenes with three.js. One HTML file, no build step,
ES-modules CDN import, OrbitControls by default.

## Tech baseline

Use ES modules with an importmap so the user doesn't need a bundler:

```html
<script type="importmap">
{
  "imports": {
    "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
  }
}
</script>
<script type="module">
  import * as THREE from "three";
  import { OrbitControls } from "three/addons/controls/OrbitControls.js";
  // ...
</script>
```

## Mandatory scene setup

Every scene MUST have:

1. **Renderer** with `antialias: true`, `setPixelRatio(devicePixelRatio)`,
   `setSize(innerWidth, innerHeight)`.
2. **Camera** with sensible defaults: `PerspectiveCamera(60, aspect, 0.1, 1000)`,
   positioned away from origin (e.g. `(5, 5, 5)`), `lookAt(0,0,0)`.
3. **Lighting** — at minimum `AmbientLight(0xffffff, 0.4)` plus a
   `DirectionalLight` with `castShadow = true` if shadows enabled. Pure
   AmbientLight alone makes everything look flat.
4. **OrbitControls** unless the user explicitly wants fixed camera. Set
   `enableDamping = true`, `dampingFactor = 0.05`.
5. **Resize handler** — listen to `resize`, update camera aspect + renderer.
6. **Animation loop** — `renderer.setAnimationLoop(tick)`, NOT manual
   `requestAnimationFrame` (setAnimationLoop is XR-ready).
7. **Background** — never leave it pure black. Use `scene.background = new
   THREE.Color(...)` or an environment map.

## Performance rules

- Reuse geometries/materials. Don't `new BoxGeometry()` in the animation loop.
- For > 100 instanced objects, use `InstancedMesh`.
- Shadows are expensive — only enable on the few objects that matter.
- For procedural particles, use `Points` + `BufferGeometry`, not 1000 meshes.

## The critique loop

After Playwright loads the page, **wait ~500ms** before screenshotting so
the first frame renders fully. Take screenshots from TWO angles if the
user wants interactivity (e.g. before and after a simulated click/drag).

## Common artifacts and starter snippets

| Ask | Approach |
|---|---|
| "Solar system" | Sun = `MeshBasicMaterial` (it emits), planets = `MeshStandardMaterial`, animate with parametric orbits |
| "Terrain" | `PlaneGeometry` with subdivisions, displace vertices via noise (`SimplexNoise` from addons) |
| "Particles" | `BufferGeometry` with 10k+ vertices, custom `ShaderMaterial` or `PointsMaterial` with a soft texture |
| "Model viewer" | `GLTFLoader` from addons, drop a URL, frame the loaded box |

## Anti-patterns

- ❌ Forgetting OrbitControls.update() in the loop with damping enabled — controls feel dead.
- ❌ Adding lights but no `MeshStandardMaterial` — Basic/Lambert won't react to your DirectionalLight properly for PBR look.
- ❌ Hard-coded canvas size that doesn't resize.
