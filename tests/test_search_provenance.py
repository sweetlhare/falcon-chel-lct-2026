"""Pure API-contract fixtures; no ASGI, real encoder, image data or explanation claim."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import secrets
import sys
import tempfile
import threading
import time
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch
import numpy as np
import gallery_client

ROOT=Path(__file__).resolve().parents[1]


class HTTPError(Exception):
    def __init__(self,status_code,detail):
        self.status_code=status_code;super().__init__(detail)


class Tensor:
    def __init__(self,array):self.array=np.asarray(array,np.float32)
    def detach(self):return self
    def cpu(self):return self
    def contiguous(self):return self
    def numpy(self):return self.array


class ProvenanceContract(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();root=Path(self.temp.name)
        (root/'manifest.json').write_text('{"fixture":true}')
        self.digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        model=SimpleNamespace(root=root,name='fixture',threshold=.5,global_metric=None,encode_tensors=lambda x:None)
        self.state={'model':model,'model_fingerprint':self.digest(root/'manifest.json')}
        bank=np.eye(2,1024,dtype=np.float32)
        self.rows=[('a',bank[0].tobytes()),('b',bank[1].tobytes())]
        client=SimpleNamespace(rows=lambda fingerprint=None:self.rows)
        self.env=dict(STATE=self.state,SEARCHES={},SEARCH_LOCK=threading.Lock(),SEARCH_TTL=1800,SEARCH_LIMIT=32,
                      np=np,hashlib=hashlib,json=json,secrets=secrets,time=time,sha=self.digest,
                      gallery_client=client,HTTPException=HTTPError)
        tree=ast.parse((ROOT/'api.py').read_text())
        names={'tensor_hash','gallery_hash','input_hash','current_model_fingerprint','bind_search','explain_tensor'}
        funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        self.assertEqual({f.name for f in funcs},names)
        exec(compile(ast.Module(body=funcs,type_ignores=[]),'<api-contract>','exec'),self.env)
        self.tensor=Tensor(np.ones((3,2,2)))
        self.query=bank[0]
        self.signature=self.env['input_hash'](b'fixture',[0,0,2,2])
        response=self.env['bind_search']({'ranking':[{'image_id':'a'},{'image_id':'b'}]},self.tensor,self.query,self.rows,self.signature)
        self.token=response['search_token']
        self.explanation=ModuleType('explanation');self.explanation.ExplanationError=ValueError
        def checked_encoder(encode,tensor,bank,ids,target,threshold,reference_embedding,reference_scores):
            np.testing.assert_array_equal(reference_embedding,self.query)
            np.testing.assert_array_equal(reference_scores,bank@reference_embedding)
            self.assertFalse(reference_embedding.flags.writeable)
            self.assertFalse(reference_scores.flags.writeable)
            return {'status':'fixture-only'}
        self.explanation.explain_query_pixels=checked_encoder
        self.module_patch=patch.dict(sys.modules,{'explanation':self.explanation});self.module_patch.start()

    def tearDown(self):
        self.module_patch.stop();self.temp.cleanup()

    def explain(self,tensor=None,signature=None):
        return self.env['explain_tensor'](tensor or self.tensor,self.token,signature or self.signature,'a')

    def assert_conflict(self,call):
        with self.assertRaises(HTTPError) as caught:call()
        self.assertEqual(caught.exception.status_code,409)

    def test_bound_snapshot_passed_to_core(self):
        answer=self.explain()
        self.assertTrue(answer['provenance']['verified_same_search'])
        self.assertEqual(answer['provenance']['search_token'],self.token)

    def test_input_and_model_mismatch_rejected(self):
        self.assert_conflict(lambda:self.explain(signature='wrong-input'))
        self.assert_conflict(lambda:self.explain(tensor=Tensor(np.zeros((3,2,2)))))
        (self.state['model'].root/'manifest.json').write_text('{"fixture":false}')
        self.assert_conflict(self.explain)

    def test_gallery_order_and_content_mismatch_rejected(self):
        self.rows.reverse();self.assert_conflict(self.explain);self.rows.reverse()
        identity,raw=self.rows[1];changed=np.frombuffer(raw,np.float32).copy();changed[3]=.1
        self.rows[1]=(identity,changed.tobytes());self.assert_conflict(self.explain)

    def test_expired_and_unknown_tokens_rejected(self):
        self.env['SEARCHES'][self.token]['created']-=1801
        self.assert_conflict(self.explain)
        self.token='unknown';self.assert_conflict(self.explain)

    def test_database_model_binding(self):
        raw=self.query.tobytes();record=dict(id='a',embedding=base64.b64encode(raw).decode(),metadata={})
        with patch.object(gallery_client,'request',return_value={'records':[record]}):
            with self.assertRaises(gallery_client.GalleryUnavailable):gallery_client.rows('model-A')
            record['metadata']['_model_fingerprint']='model-B'
            with self.assertRaises(gallery_client.GalleryUnavailable):gallery_client.rows('model-A')
            record['metadata']['_model_fingerprint']='model-A'
            self.assertEqual(gallery_client.rows('model-A'),[('a',raw)])
        with patch.object(gallery_client,'request',return_value={'stored':1}) as request:
            gallery_client.upsert([('a',raw,{'bbox':[0,0,2,2]})],'model-A')
            sent=request.call_args.args[1]['records'][0]
            self.assertEqual(sent['metadata']['_model_fingerprint'],'model-A')
            self.assertEqual(base64.b64decode(sent['embedding']),raw)


if __name__=='__main__':unittest.main(verbosity=2)
