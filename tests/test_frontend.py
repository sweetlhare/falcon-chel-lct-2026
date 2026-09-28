"""Contract check using a synthetic backend; does not execute or validate the model."""
import hashlib
import http.client
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
import frontend


class Backend(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        body = json.dumps({'path': self.path, 'fixture': True}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        body = self.rfile.read(int(self.headers['Content-Length']))
        self.send_response(422)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(body)


class QuietFrontend(frontend.Handler):
    def log_message(self, *args):
        pass


class ProxyContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = ThreadingHTTPServer(('127.0.0.1', 0), Backend)
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), QuietFrontend)
        cls.origin = urlsplit(f'http://127.0.0.1:{cls.backend.server_port}')
        frontend.BACKEND = cls.origin
        for server in (cls.backend, cls.server):
            threading.Thread(target=server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        for server in (cls.server, cls.backend):
            server.shutdown()
            server.server_close()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        try:
            conn.request(method, path, body, headers or {})
            result = conn.getresponse()
            return result.status, result.getheader('Content-Type'), result.read()
        finally:
            conn.close()

    def test_api_query_string(self):
        status, content_type, body = self.request('GET', '/health?probe=1')
        self.assertEqual((status, content_type), (200, 'application/json'))
        self.assertEqual(json.loads(body)['path'], '/health?probe=1')

    def test_post_status_and_body_preserved(self):
        payload = b'{"image_base64":"AA==","bbox":[0,0,10,10]}'
        status, _, body = self.request('POST', '/search', payload, {'Content-Type': 'application/json'})
        self.assertEqual((status, body), (422, payload))

    def test_unknown_and_external_target_rejected(self):
        self.assertEqual(self.request('GET', '/weights/manifest.json')[0], 404)
        self.assertEqual(self.request('GET', 'http://example.org/health')[0], 400)

    def test_request_limits(self):
        self.assertEqual(self.request('POST', '/search', headers={'Content-Length': '25000001'})[0], 413)
        self.assertEqual(self.request('POST', '/search', headers={'Content-Length': '-1'})[0], 413)
        self.assertEqual(self.request('POST', '/search', headers={'Content-Length': 'bad'})[0], 400)
        self.assertEqual(self.request('POST', '/search', headers={'Transfer-Encoding': 'chunked'})[0], 400)

    def test_unavailable_backend_is_502(self):
        class Disconnected(Backend):
            def do_GET(self):
                self.close_connection = True
        broken = ThreadingHTTPServer(('127.0.0.1', 0), Disconnected)
        threading.Thread(target=broken.serve_forever, daemon=True).start()
        frontend.BACKEND = urlsplit(f'http://127.0.0.1:{broken.server_port}')
        try:
            self.assertEqual(self.request('GET', '/health')[0], 502)
            self.assertEqual(self.request('GET', '/frontend-health')[0], 200)
        finally:
            frontend.BACKEND = self.origin
            broken.shutdown()
            broken.server_close()

    def test_offline_swagger_assets(self):
        status, _, html = self.request('GET', '/docs')
        self.assertEqual(status, 200)
        self.assertNotIn(b'https://', html)
        self.assertIn(b'validatorUrl:null', html)
        manifest = json.loads((frontend.ROOT / 'static/swagger/manifest.json').read_text())
        for item in manifest['files']:
            if item['path'].endswith(('.js', '.css')):
                status, _, body = self.request('GET', '/static/swagger/' + item['path'])
                self.assertEqual(status, 200)
                self.assertEqual(hashlib.sha256(body).hexdigest(), item['sha256'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
