# Bundled Three.js

`three.min.js` is the unmodified Three.js r134 distribution used by the existing viewer.
It is bundled to make the default HTML build independent of a CDN. The MIT license is
in `THREE-LICENSE.txt` and must accompany redistributions.

Upstream: https://github.com/mrdoob/three.js/tree/r134

SHA-256:

* `three.min.js`: `74782bdbcf6518f7745ed77035968fcae95ed4ab5c9a0f90cf646a69c20785ec`
* `THREE-LICENSE.txt`: `7dddf7c5b8fd10ee654db8857d75d104b5557889aa5a91fc4ca545ea7c07062f`

This preserves the renderer API already used by the application; it is not a renderer
version upgrade. `--three cdn` remains an explicit online option; `--three none` builds
a page without the 3D renderer.
