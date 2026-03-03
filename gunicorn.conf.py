"""Gunicorn configuration file for production deployment."""
import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"
backlog = 2048

# Worker processes
# Formula: (2 x $num_cores) + 1
# But for I/O bound apps (like this RAG API), can use more workers
workers = int(os.getenv("GUNICORN_WORKERS", multiprocessing.cpu_count() * 2 + 1))
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000  # Restart workers after N requests (prevent memory leaks)
max_requests_jitter = 50  # Add randomness to avoid all workers restarting at once
timeout = 120  # Worker timeout (important for LLM calls which can be slow)
keepalive = 5

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
loglevel = os.getenv("LOG_LEVEL", "info").lower()
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "1cc-rag-api"

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# Graceful shutdown
graceful_timeout = 30  # Time to wait for requests to finish during shutdown

# Worker lifecycle hooks
def on_starting(server):
    """Called just before the master process is initialized."""
    server.log.info("Starting Gunicorn server")

def on_reload(server):
    """Called when config is reloaded."""
    server.log.info("Reloading Gunicorn configuration")

def when_ready(server):
    """Called just after the server is started."""
    server.log.info(f"Gunicorn server is ready. Listening on {bind}")
    server.log.info(f"Workers: {workers}, Worker class: {worker_class}")

def pre_fork(server, worker):
    """Called just before a worker is forked."""
    pass

def post_fork(server, worker):
    """Called just after a worker has been forked."""
    server.log.info(f"Worker spawned (pid: {worker.pid})")

def pre_exec(server):
    """Called just before a new master process is forked."""
    server.log.info("Forked child, re-executing")

def worker_int(worker):
    """Called when a worker receives the SIGINT or SIGQUIT signal."""
    worker.log.info(f"Worker received INT or QUIT signal (pid: {worker.pid})")

def worker_abort(worker):
    """Called when a worker receives the SIGABRT signal."""
    worker.log.info(f"Worker received SIGABRT signal (pid: {worker.pid})")

def pre_request(worker, req):
    """Called just before a worker processes the request."""
    pass

def post_request(worker, req, environ, resp):
    """Called after a worker processes the request."""
    pass

def child_exit(server, worker):
    """Called just after a worker has been exited."""
    server.log.info(f"Worker exited (pid: {worker.pid})")

def worker_exit(server, worker):
    """Called just after a worker has been exited."""
    server.log.info(f"Worker exiting (pid: {worker.pid})")

def nworkers_changed(server, new_value, old_value):
    """Called when worker count changes."""
    server.log.info(f"Worker count changed from {old_value} to {new_value}")

def on_exit(server):
    """Called just before the server exits."""
    server.log.info("Shutting down Gunicorn server")
