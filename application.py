"""WSGI entry point for hosting (Azure App Service, gunicorn).

Azure's Python build looks for `application.py` in the project root and serves the
`app` object from it. Locally you can still run `python app/app.py`.

    gunicorn --bind=0.0.0.0:8000 --timeout 600 --workers 1 --threads 4 application:app
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("tea_outlook_app", Path(__file__).parent / "app" / "app.py")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

app = _module.app
