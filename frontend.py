"""Static browser client and fixed-origin API proxy; Python standard library only."""
import http.client
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
BACKEND = urlsplit(os.environ.get('FALCON_BACKEND', 'http://backend:8000'))
if BACKEND.scheme != 'http' or not BACKEND.hostname or BACKEND.path not in ('', '/'):
    raise ValueError('FALCON_BACKEND must be a fixed http origin')
BODY_LIMIT = 25_000_000
API_PATHS = {'/health', '/encode', '/search', '/gallery', '/examples', '/openapi.json','/explain'}
HOP_HEADERS = {'connection', 'keep-alive', 'proxy-authenticate',
               'proxy-authorization', 'te', 'trailer', 'transfer-encoding', 'upgrade'}
STATIC = {'/': ('index.html', 'text/html; charset=utf-8'),
          '/docs': ('docs.html', 'text/html; charset=utf-8'),
          '/explanation_ui.js': ('explanation_ui.js', 'text/javascript; charset=utf-8'),
          '/static/swagger/swagger-ui-bundle.js': ('static/swagger/swagger-ui-bundle.js', 'text/javascript'),
          '/static/swagger/swagger-ui.css': ('static/swagger/swagger-ui.css', 'text/css')}


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, body, content_type='application/json'):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path in STATIC:
            name, content_type = STATIC[path]
            return self.reply(200, (ROOT / name).read_bytes(), content_type)
        if path == '/frontend-health':
            return self.reply(200, b'{"status":"ok","service":"frontend"}')
        self.proxy()

    def do_POST(self):
        self.proxy()

    def proxy(self):
        target = urlsplit(self.path)
        if target.scheme or target.netloc or not self.path.startswith('/'):
            return self.reply(400, b'{"detail":"Invalid request target"}')
        if (target.path not in API_PATHS and
                not target.path.startswith(('/preview/', '/search-example/','/explain-example/'))):
            return self.reply(404, b'{"detail":"Unknown endpoint"}')
        if self.headers.get('Transfer-Encoding'):
            return self.reply(400, b'{"detail":"Content-Length required"}')
        try:
            length = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            return self.reply(400, b'{"detail":"Invalid Content-Length"}')
        if not 0 <= length <= BODY_LIMIT:
            return self.reply(413, b'{"detail":"Request exceeds 25 MB"}')
        connection = http.client.HTTPConnection(BACKEND.hostname, BACKEND.port or 80, timeout=120)
        try:
            body = self.rfile.read(length) if length else None
            headers = {k: self.headers[k] for k in ('Content-Type', 'Accept') if k in self.headers}
            connection.request(self.command, self.path, body=body, headers=headers)
            response = connection.getresponse()
            data = response.read()
        except (OSError, http.client.HTTPException):
            return self.reply(502, b'{"detail":"Inference service unavailable; retry shortly"}')
        finally:
            connection.close()
        self.send_response(response.status)
        for key, value in response.getheaders():
            if key.lower() not in HOP_HEADERS | {'content-length', 'server', 'date'}:
                self.send_header(key, value)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == '__main__':
    ThreadingHTTPServer(('0.0.0.0', int(os.environ.get('FALCON_FRONTEND_PORT', '8080'))), Handler).serve_forever()
