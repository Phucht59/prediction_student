# Historical V5.1 extraction

The original V5.1 UCI source was located in Git commit `f51e1f43d9af768194bb34ea5ff4aeb8e4cbd502`, the parent of deletion commit `46e3af01`. It was extracted with `git archive` into this directory only. No file under `src/`, `configs/final/`, or an authority artifact directory was restored or changed.

The extracted code retains historical imports rooted at `src.studies.v5*`; an isolated compatibility shim is still required before executing it. The shim must also map historical protocol-relative roots into this experiment namespace without changing scientific logic.
