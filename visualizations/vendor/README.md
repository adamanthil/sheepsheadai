# Vendored three.js sources

Inlined into `ppo_architecture_3d.html` by `build_3d_html.py` so the built
single-file visualization works fully offline.

- `three.module.min.js` — three@0.158.0 core (`build/three.module.min.js`
  from unpkg).
- `OrbitControls.js` — three@0.158.0
  `examples/jsm/controls/OrbitControls.js`, unmodified; its `from 'three'`
  import is rewritten to a blob URL at page load.
- `addons.bundle.js` — esbuild bundle of the addon modules the template
  uses, with bare `three` kept external (rewritten to the same blob URL at
  load). Rebuild with:

  ```sh
  npm install three@0.158.0
  cat > entry.js <<'EOF'
  export { EffectComposer } from "./node_modules/three/examples/jsm/postprocessing/EffectComposer.js";
  export { RenderPass } from "./node_modules/three/examples/jsm/postprocessing/RenderPass.js";
  export { UnrealBloomPass } from "./node_modules/three/examples/jsm/postprocessing/UnrealBloomPass.js";
  export { OutputPass } from "./node_modules/three/examples/jsm/postprocessing/OutputPass.js";
  export { Line2 } from "./node_modules/three/examples/jsm/lines/Line2.js";
  export { LineMaterial } from "./node_modules/three/examples/jsm/lines/LineMaterial.js";
  export { LineGeometry } from "./node_modules/three/examples/jsm/lines/LineGeometry.js";
  export { RoundedBoxGeometry } from "./node_modules/three/examples/jsm/geometries/RoundedBoxGeometry.js";
  export { RoomEnvironment } from "./node_modules/three/examples/jsm/environments/RoomEnvironment.js";
  EOF
  npx esbuild entry.js --bundle --format=esm --external:three --minify --outfile=addons.bundle.js
  ```

  (`--external:three` also externalizes `three/*` subpaths, hence the
  relative `node_modules` paths in the entry.)
