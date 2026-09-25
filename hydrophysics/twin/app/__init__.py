"""Helpers for the twin's decision page (``hydrophysics.twin.viewer_app``).

``prep`` holds the numpy data preparation the page is built from (every number the page
shows is computed there, from the forward and basis npz files), ``geo`` traces the public
OpenStreetMap geometry the page overlays (THSR centreline, rivers), and ``template.html``
is the page itself.
"""

from __future__ import annotations
