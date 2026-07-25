"""
Windows SSL cert-store workaround
=================================
Some Python 3.8 builds on Windows crash when creating the default SSL context,
because enumerating the Windows certificate store hits a malformed entry:

    ssl.SSLError: [ASN1: NOT_ENOUGH_DATA] not enough data (_ssl.c:4194)

This breaks every urllib-based download in the pipeline — the SegFormer
face-parsing weights, the LPIPS/AlexNet weights (measurement), and the CLIP
weights (boundary extraction). `gdown` was unaffected because it already uses
certifi, which is the clue that led to this fix.

Importing this module probes the default context; if (and only if) it is
broken, it redirects the default HTTPS context to certifi's CA bundle, which
sidesteps the Windows store while still verifying certificates. On healthy
machines nothing changes.

Import it *before* any code that downloads over HTTPS.
"""

import ssl

try:
    import certifi

    def _certifi_https_context(*args, **kwargs):
        return ssl.create_default_context(cafile=certifi.where())

    try:
        # If this succeeds, the OS trust store is fine — leave it alone.
        ssl.create_default_context()
    except ssl.SSLError:
        ssl._create_default_https_context = _certifi_https_context
        print("[ssl_patch] Windows cert store unreadable — using certifi CA bundle.")
except ImportError:
    # certifi missing: nothing we can do here; downloads may still work.
    pass
