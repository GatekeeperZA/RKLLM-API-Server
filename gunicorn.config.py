import os

workers = int(os.environ.get('GUNICORN_WORKERS', 1))
worker_class = 'gthread'
threads = int(os.environ.get('GUNICORN_THREADS', 4))
timeout = int(os.environ.get('GUNICORN_TIMEOUT', 300))
bind = os.environ.get('GUNICORN_BIND', '0.0.0.0:8000')
accesslog = '-'
errorlog = '-'
