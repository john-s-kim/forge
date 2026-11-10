"""WSGI entry point:  gunicorn -k gthread --threads 32 forge.api.wsgi:app"""

import os

from forge.api.app import create_app
from forge.service import Context

app = create_app(Context.create(), os.environ.get("FORGE_WEB_DIR", "web/dist"))
