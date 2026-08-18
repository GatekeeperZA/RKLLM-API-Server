import sys
import urllib.request
import os

bind = os.environ.get('GUNICORN_BIND', '0.0.0.0:8000')
# Handle IPv6 bracket notation e.g. [::]:8000
if bind.startswith('['):
    bracket_end = bind.index(']')
    host = bind[1:bracket_end]
    port = bind[bracket_end + 2:]
else:
    host, port = bind.rsplit(':', 1)
if host == '0.0.0.0':
    host = '127.0.0.1'
elif host == '::':
    host = '[::1]'

try:
    url = f'http://{host}:{port}/health'
    with urllib.request.urlopen(url, timeout=4) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
