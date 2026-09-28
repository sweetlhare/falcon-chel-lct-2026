"""Narrow internal HTTP adapter; embeddings preserve their exact float32 bytes."""
import base64
import json
import os
import urllib.error
import urllib.request
from urllib.parse import urlsplit

ORIGIN = os.environ.get('FALCON_DB_URL', 'http://database:8002').rstrip('/')
parsed = urlsplit(ORIGIN)
if parsed.scheme != 'http' or not parsed.hostname or parsed.path:
    raise ValueError('FALCON_DB_URL must be an HTTP origin')
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class GalleryUnavailable(RuntimeError):
    pass


def request(path, payload=None):
    body = None if payload is None else json.dumps(payload, allow_nan=False).encode()
    req = urllib.request.Request(ORIGIN + path, data=body, headers={'Content-Type': 'application/json'})
    try:
        with OPENER.open(req, timeout=15) as response:
            return json.load(response)
    except (OSError, ValueError, urllib.error.URLError) as error:
        raise GalleryUnavailable('Gallery service unavailable') from error


def count():
    return request('/count')['count']


def rows(model_fingerprint=None):
    records=request('/records')['records']
    if model_fingerprint is not None:
        if any(r['metadata'].get('_model_fingerprint')!=model_fingerprint for r in records):
            raise GalleryUnavailable('Gallery vectors belong to a different or unbound model; use a fresh gallery')
    return [(r['id'], base64.b64decode(r['embedding'], validate=True)) for r in records]


def upsert(records,model_fingerprint=None):
    """Accept iterable of (id, little-endian float32 bytes, metadata dict)."""
    records = list(records)
    if not records:
        return
    if model_fingerprint is not None:
        records=[(identity,raw,dict(metadata,_model_fingerprint=model_fingerprint)) for identity,raw,metadata in records]
    result = request('/upsert', {'records': [dict(id=identity, embedding=base64.b64encode(raw).decode(), metadata=metadata)
                                            for identity, raw, metadata in records]})
    if result['stored'] != len(records):
        raise GalleryUnavailable('Incomplete gallery transaction')
