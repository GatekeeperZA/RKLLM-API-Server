import sys
import urllib.request
import os

bind = os.environ.get('GUNICORN_BIND', '0.0.0.0:8000')
host, port = bind.rsplit(':', 1)
if host == '0.0.0.0':
    host = '127.0.0.1'

try:
    url = f'http://{host}:{port}/health'
    with urllib.request.urlopen(url, timeout=4) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
