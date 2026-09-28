"""Image+BBox service with persistent SQLite gallery and exact cosine retrieval."""
import base64,hashlib,io,json,os,secrets,threading,time,zipfile
from contextlib import asynccontextmanager
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
from fastapi import FastAPI,HTTPException
from fastapi.responses import FileResponse,Response,JSONResponse
from pydantic import BaseModel,Field
from inference import Model,decompose,validate_bbox,sha
import gallery_client

STATE={};LOCK=threading.Lock()
SEARCHES={};SEARCH_LOCK=threading.Lock();SEARCH_TTL=1800;SEARCH_LIMIT=32
OUTPUT=Path(os.environ.get('FALCON_OUTPUT','outputs'))
ARCHIVE=Path(os.environ.get('FALCON_ARCHIVE','data/dataset.zip'))


@asynccontextmanager
async def lifespan(app):
    STATE['model']=Model(os.environ.get('FALCON_WEIGHTS','weights'),int(os.environ.get('FALCON_THREADS','2')))
    STATE['model_fingerprint']=sha(STATE['model'].root/'manifest.json')
    gallery_client.rows(STATE['model_fingerprint'])
    if (OUTPUT/'rows.json').exists():
        binding=OUTPUT/'model_manifest.json'
        if not binding.exists() or sha(binding)!=STATE['model_fingerprint']:
            raise RuntimeError('Cached gallery lacks the exact model_manifest.json from its inference run')
        report=json.loads((OUTPUT/'report.json').read_text())
        for name in ['rows.json','embeddings.npy','pooling_weights.npy','redactions.json']:
            if report['hashes'].get(name)!=sha(OUTPUT/name):raise RuntimeError('Cached gallery output hash mismatch: '+name)
        rows=json.loads((OUTPUT/'rows.json').read_text());emb=np.load(OUTPUT/'embeddings.npy',allow_pickle=False)
        assert emb.shape==(len(rows['query'])+len(rows['gallery']),1024)
        STATE.update(rows=rows,embeddings=emb,metadata={r['image_id']:r for r in rows['query']+rows['gallery']},
                     pooling=np.load(OUTPUT/'pooling_weights.npy',allow_pickle=False),
                     positions={r['image_id']:i for i,r in enumerate(rows['query']+rows['gallery'])})
        masks=json.loads((OUTPUT/'redactions.json').read_text())['records'];STATE['masks']={r['image_id']:r['redactions'] for r in masks}
        gallery_client.upsert([(r['image_id'],emb[len(rows['query'])+i].astype('<f4').tobytes(),r) for i,r in enumerate(rows['gallery'])],STATE['model_fingerprint'])
    yield


app=FastAPI(title='Falcon: поиск автомобиля',version='1.0',lifespan=lifespan)


@app.exception_handler(gallery_client.GalleryUnavailable)
async def unavailable_gallery(request,error):
    return JSONResponse(status_code=503,content={'detail':'Gallery service unavailable; retry shortly'})


class ImageRequest(BaseModel):
    image_base64:str=Field(max_length=24_000_000)
    bbox:list[float]=Field(min_length=4,max_length=4)
    top_n:int=Field(default=10,ge=1,le=20)


def decode(req):
    try:
        raw=base64.b64decode(req.image_base64,validate=True)
        with Image.open(io.BytesIO(raw)) as im:
            if im.width*im.height>30_000_000:raise ValueError('Image exceeds30MP')
            image=im.convert('RGB')
        validate_bbox(image,req.bbox)
        return image
    except Exception as e:raise HTTPException(422,'Invalid image or BBox: '+str(e)) from e


def search_vector(embedding,top_n,rows=None):
    if rows is None:rows=gallery_client.rows(STATE.get('model_fingerprint'))
    if not rows:return dict(candidates=[],ranking=[],refused=True,threshold=STATE['model'].threshold,gallery_size=0)
    bank=np.stack([np.frombuffer(row[1],dtype='<f4') for row in rows]);scores=bank@embedding
    order=np.argsort(-scores,kind='stable')[:top_n];threshold=STATE['model'].threshold
    ranking=[dict(image_id=rows[i][0],score=float(scores[i]),accepted=bool(scores[i]>=threshold),contributions=decompose(embedding,bank[i])) for i in order]
    return dict(candidates=[r for r in ranking if r['accepted']],ranking=ranking,refused=not any(r['accepted'] for r in ranking),
                threshold=threshold,gallery_size=len(rows),score_kind='cosine similarity, not probability')


@app.get('/')
def index():return FileResponse(Path(__file__).parent/'index.html')


@app.get('/health')
def health():
    n=gallery_client.count()
    return dict(status='ok',model=STATE['model'].name,dimension=1024,gallery_size=n,device='cpu',threshold=STATE['model'].threshold)


def encode_request(req):
    image=decode(req)
    try:
        with LOCK:
            t,redaction,_=STATE['model'].prepare(image,req.bbox);e,w=STATE['model'].encode_tensors([t])
    except ValueError as error:
        raise HTTPException(422,'Invalid image or BBox: '+str(error)) from error
    rgb=np.clip((t.numpy().transpose(1,2,0)*np.array([.229,.224,.225])+np.array([.485,.456,.406]))*255,0,255).round().astype(np.uint8)
    buff=io.BytesIO();Image.fromarray(rgb).save(buff,format='PNG')
    return dict(embedding=e[0].tolist(),pooling_weights=w[0].reshape(8,8).tolist(),redaction=redaction,preview_base64=base64.b64encode(buff.getvalue()).decode(),
                explanation='Learned local pooling weights, not semantic part labels or identity confidence'),t


@app.post('/encode')
def encode(req:ImageRequest):
    return encode_request(req)[0]


def tensor_hash(tensor):
    value=tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str((value.shape,str(value.dtype))).encode()+value.tobytes()).hexdigest()


def gallery_hash(rows):
    digest=hashlib.sha256()
    for identity,vector in rows:
        encoded=identity.encode();digest.update(len(encoded).to_bytes(4,'big'));digest.update(encoded);digest.update(vector)
    return digest.hexdigest()


def input_hash(raw,bbox,image_id=None):
    return hashlib.sha256(raw+json.dumps([list(map(float,bbox)),image_id],separators=(',',':')).encode()).hexdigest()


def current_model_fingerprint():
    value=sha(STATE['model'].root/'manifest.json')
    if value!=STATE['model_fingerprint']:raise HTTPException(409,'Model manifest changed; restart service and search again')
    return value


def bind_search(answer,tensor,embedding,rows,signature):
    embedding=np.asarray(embedding,dtype='<f4').copy();embedding.setflags(write=False)
    scores=np.stack([np.frombuffer(r[1],dtype='<f4') for r in rows])@embedding if rows else np.empty(0,np.float32)
    scores.setflags(write=False)
    provenance=dict(input_tensor_sha256=tensor_hash(tensor),query_embedding_sha256=hashlib.sha256(embedding.tobytes()).hexdigest(),
                    gallery_sha256=gallery_hash(rows),weights_manifest_sha256=current_model_fingerprint(),gallery_size=len(rows))
    token=secrets.token_urlsafe(24);now=time.monotonic()
    with SEARCH_LOCK:
        for key in list(SEARCHES):
            if now-SEARCHES[key]['created']>SEARCH_TTL:del SEARCHES[key]
        while len(SEARCHES)>=SEARCH_LIMIT:del SEARCHES[next(iter(SEARCHES))]
        SEARCHES[token]=dict(created=now,provenance=provenance.copy(),embedding=embedding,scores=scores,
                             input_signature=signature,ranking_ids=[r['image_id'] for r in answer['ranking']])
    answer.update(search_token=token,provenance=provenance,search_token_expires_in_seconds=SEARCH_TTL)
    return answer


@app.post('/search')
def search(req:ImageRequest):
    result,tensor=encode_request(req);embedding=np.asarray(result['embedding'],np.float32)
    rows=gallery_client.rows(STATE['model_fingerprint']);answer=search_vector(embedding,req.top_n,rows)
    answer.update(pooling_weights=result['pooling_weights'],redaction=result['redaction'],preview_base64=result['preview_base64'])
    return bind_search(answer,tensor,embedding,rows,input_hash(base64.b64decode(req.image_base64),req.bbox))


class AddRequest(ImageRequest):
    image_id:str=Field(min_length=1,max_length=128,pattern=r'^[A-Za-z0-9_.-]+$')


class ExplainRequest(ImageRequest):
    target_id:str|None=Field(default=None,max_length=128)
    search_token:str=Field(min_length=1,max_length=128)


def explain_tensor(tensor,search_token,signature,target_id=None):
    from explanation import explain_query_pixels,ExplanationError
    with SEARCH_LOCK:
        snapshot=SEARCHES.get(search_token)
    if snapshot is None or time.monotonic()-snapshot['created']>SEARCH_TTL:
        raise HTTPException(409,'Search token expired or unknown; run search again')
    provenance=snapshot['provenance']
    if current_model_fingerprint()!=provenance['weights_manifest_sha256']:
        raise HTTPException(409,'Model differs from search; run search again')
    if signature!=snapshot['input_signature'] or tensor_hash(tensor)!=provenance['input_tensor_sha256']:
        raise HTTPException(409,'Input differs from search; run search again')
    rows=gallery_client.rows(STATE['model_fingerprint'])
    if gallery_hash(rows)!=provenance['gallery_sha256']:
        raise HTTPException(409,'Gallery differs from search; run search again')
    if len(rows)<2:raise HTTPException(422,'Explanation requires at least two gallery images')
    ids=[r[0] for r in rows];bank=np.stack([np.frombuffer(r[1],dtype='<f4') for r in rows])
    model=STATE['model'];tick=time.perf_counter()
    if target_id is None:target_id=snapshot['ranking_ids'][0]
    if target_id not in snapshot['ranking_ids']:raise HTTPException(409,'Target was not among the search results')
    try:
        answer=explain_query_pixels(model.encode_tensors,tensor,bank,ids,target_id,threshold=model.threshold,
                                    reference_embedding=snapshot['embedding'],reference_scores=snapshot['scores'])
    except ExplanationError as error:raise HTTPException(422,str(error)) from error
    if gallery_hash(gallery_client.rows(STATE['model_fingerprint']))!=provenance['gallery_sha256']:
        raise HTTPException(409,'Gallery changed during explanation; run search again')
    if current_model_fingerprint()!=provenance['weights_manifest_sha256']:
        raise HTTPException(409,'Model changed during explanation; run search again')
    answer.update(seconds=time.perf_counter()-tick,model=model.name,global_metric='KISSME' if model.global_metric is not None else 'within-ID whitening')
    answer['provenance']=dict(provenance,search_token=search_token,verified_same_search=True)
    return answer


@app.post('/explain')
def explain(req:ExplainRequest):
    image=decode(req)
    with LOCK:
        try:t,redaction,_=STATE['model'].prepare(image,req.bbox)
        except ValueError as error:raise HTTPException(422,str(error)) from error
        result=explain_tensor(t,req.search_token,input_hash(base64.b64decode(req.image_base64),req.bbox),req.target_id)
    result['redaction']=redaction
    return result


@app.get('/explain-example/{image_id}')
def explain_example(image_id:str,search_token:str,target_id:str|None=None):
    if image_id not in {r['image_id'] for r in STATE.get('rows',{}).get('query',[])}:raise HTTPException(404,'Unknown example')
    from falcon.audit import image_path
    row=STATE['metadata'][image_id]
    with zipfile.ZipFile(ARCHIVE) as z:
        raw=z.read(image_path(image_id,set(z.namelist())))
        with Image.open(io.BytesIO(raw)) as im:
            image=im.convert('RGB')
    with LOCK:
        t,_,_=STATE['model'].prepare(image,[float(row[k]) for k in ('x','y','w','h')],STATE['masks'][image_id])
        result=explain_tensor(t,search_token,input_hash(raw,[float(row[k]) for k in ('x','y','w','h')],image_id),target_id)
    result['query_id']=image_id
    return result


@app.post('/gallery')
def add(req:AddRequest):
    result=encode(req);e=np.asarray(result['embedding'],dtype='<f4')
    gallery_client.upsert([(req.image_id,e.tobytes(),dict(bbox=req.bbox))],current_model_fingerprint())
    return dict(image_id=req.image_id,stored=True)


@app.get('/examples')
def examples():return [dict(image_id=r['image_id'],bbox=[float(r[k]) for k in ['x','y','w','h']]) for r in STATE.get('rows',{}).get('query',[])[:40]]


@app.get('/search-example/{image_id}')
def example(image_id:str):
    if image_id not in {r['image_id'] for r in STATE.get('rows',{}).get('query',[])}:raise HTTPException(404,'Unknown example')
    from falcon.audit import image_path
    row=STATE['metadata'][image_id];bbox=[float(row[k]) for k in ('x','y','w','h')]
    with zipfile.ZipFile(ARCHIVE) as z:
        raw=z.read(image_path(image_id,set(z.namelist())))
        with Image.open(io.BytesIO(raw)) as im:image=im.convert('RGB')
    with LOCK:
        tensor,redaction,_=STATE['model'].prepare(image,bbox,STATE['masks'][image_id])
        embedding,pooling=STATE['model'].encode_tensors([tensor])
    rows=gallery_client.rows(STATE['model_fingerprint']);answer=search_vector(embedding[0],10,rows)
    answer.update(pooling_weights=pooling[0].reshape(8,8).tolist(),source='fresh encoding of the bound example image',query_id=image_id,redaction=redaction)
    return bind_search(answer,tensor,embedding[0],rows,input_hash(raw,bbox,image_id))


@app.get('/preview/{image_id}')
def preview(image_id:str):
    if image_id not in STATE.get('metadata',{}):raise HTTPException(404,'Preview unavailable')
    from falcon.audit import image_path
    r=STATE['metadata'][image_id]
    with zipfile.ZipFile(ARCHIVE) as z:
        with Image.open(io.BytesIO(z.read(image_path(image_id,set(z.namelist()))))) as im:
            with LOCK:t,_,_=STATE['model'].prepare(im,[float(r[k]) for k in ['x','y','w','h']],STATE['masks'][image_id])
    # Invert normalization to show the actual preprocessed256 image.
    rgb=np.clip((t.numpy().transpose(1,2,0)*np.array([.229,.224,.225])+np.array([.485,.456,.406]))*255,0,255).round().astype(np.uint8)
    buff=io.BytesIO();Image.fromarray(rgb).save(buff,format='PNG');return Response(buff.getvalue(),media_type='image/png')
