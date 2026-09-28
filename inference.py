"""Offline CHEL1024 inference. Camera/identity metadata never enter the model."""
import argparse
import csv
import hashlib
import io
import json
import os
import time
import zipfile
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import numpy as np
from PIL import Image, ImageDraw


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def validate_bbox(image,bbox):
    if len(bbox)!=4 or not np.isfinite(bbox).all():raise ValueError('BBox must contain four finite numbers')
    x,y,w,h=map(float,bbox)
    if not (w>0 and h>0 and x>=0 and y>=0 and x+w<=image.width and y+h<=image.height):raise ValueError('BBox lies outside the image')
    if round(x+w)<=round(x) or round(y+h)<=round(y):raise ValueError('Rounded BBox is empty')
    return x,y,w,h


class Model:
    def __init__(self,weights,threads=2):
        import torch,timm,onnxruntime as ort,cv2
        from timm.models import load_checkpoint
        self.torch=torch;self.cv2=cv2;cv2.setNumThreads(1);torch.set_num_threads(threads)
        self.root=Path(weights);self.manifest=json.loads((self.root/'manifest.json').read_text())
        self.name=self.manifest.get('name','CHEL1024')
        for name,h in self.manifest['sha256'].items():
            if sha(self.root/name)!=h:raise ValueError('Model hash mismatch: '+name)
        self.threshold=float(self.manifest['threshold'])
        self.model=timm.create_model(self.manifest['backbone'],pretrained=False,num_classes=0)
        load_checkpoint(self.model,str(self.root/'backbone.safetensors'),strict=True);self.model.eval()
        config=timm.data.resolve_model_data_config(self.model)
        assert tuple(config['input_size'])==(3,256,256) and config['crop_pct']==1.
        self.processor=timm.data.create_transform(**config,is_training=False)
        self.models=[dict(np.load(self.root/n,allow_pickle=False)) for n in ['chel_v1.npz','chel_v2.npz','compression.npz']]
        metric_name=self.manifest.get('global_metric')
        self.global_metric=dict(np.load(self.root/metric_name,allow_pickle=False)) if metric_name else None
        options=ort.SessionOptions();options.intra_op_num_threads=threads;options.inter_op_num_threads=1
        self.detector=ort.InferenceSession(str(self.root/'plate_redaction.onnx'),sess_options=options,providers=['CPUExecutionProvider'])

    def masks(self,image,bbox):
        # Same detector and expansion policy as the frozen research pipeline.
        x,y,w,h=[int(v) for v in validate_bbox(image,bbox)]
        if min(w,h)<1:raise ValueError('Integer BBox is empty')
        cv2=self.cv2;crop=np.asarray(image.crop((x,y,x+w,y+h)));scale=min(384/h,384/w)
        nw,nh=max(1,round(w*scale)),max(1,round(h*scale));dx,dy=(384-nw)/2,(384-nh)/2
        resized=cv2.resize(crop,(nw,nh))
        padded=cv2.copyMakeBorder(resized,round(dy-.1),round(dy+.1),round(dx-.1),round(dx+.1),cv2.BORDER_CONSTANT,value=(114,114,114))
        inputs=np.ascontiguousarray(padded.transpose(2,0,1)[None],dtype=np.float32)/255
        detections=self.detector.run(None,{'images':inputs})[0];rectangles=[]
        for det in detections:
            if det[6]<.1:continue
            a,b,c,d=(det[1:5]-np.array([dx,dy,dx,dy]))/scale;rw,rh=max(1,c-a),max(1,d-b)
            rect=[max(x,int(a+x-.3*rw)),max(y,int(b+y-.6*rh)),min(x+w,int(c+x+.3*rw)),min(y+h,int(d+y+.6*rh))]
            if (rect[2]-rect[0])*(rect[3]-rect[1])>.8*w*h:continue
            if rect[2]>rect[0] and rect[3]>rect[1]:rectangles.append(rect)
        fallback=not rectangles
        if fallback:rectangles=[[x,y+int(.35*h),x+w,y+h]]
        return rectangles,fallback

    def prepare(self,image,bbox,masks=None):
        image=image.convert('RGB');x,y,w,h=validate_bbox(image,bbox)
        if masks is None:rectangles,fallback=self.masks(image,bbox)
        else:rectangles,fallback=masks,False
        redacted=image.copy();draw=ImageDraw.Draw(redacted)
        for rect in rectangles:draw.rectangle(rect,fill=(127,127,127))
        crop=redacted.crop((round(x),round(y),round(x+w),round(y+h)))
        return self.processor(crop),dict(redactions=rectangles,fallback=fallback),crop

    def encode_tensors(self,tensors):
        from falcon.pixel_certificate import student_from_raw
        from falcon.evidence_ledger import unit,apply_metric,token_inputs,hidden_features,predict_utility,aggregate_tokens
        torch=self.torch
        with torch.inference_mode():
            state=self.model.forward_features(torch.stack(tensors));g=torch.nn.functional.normalize(self.model.forward_head(state).float(),dim=-1)
            patches=state[:,self.model.num_prefix_tokens:];assert patches.shape[1]==256
            local=torch.nn.functional.normalize(torch.nn.functional.adaptive_avg_pool2d(patches.transpose(1,2).reshape(len(g),768,16,16),(8,8)).flatten(2).transpose(1,2).float(),dim=-1)
        g=g.numpy();local=local.numpy().astype(np.float16).astype(np.float32)
        embedding=student_from_raw(g,local,*self.models)
        if self.global_metric is not None:
            metric=self.global_metric
            embedding[:,:768]=apply_metric(g,metric['center'],metric['transform'])*np.sqrt(.75)
            embedding=unit(embedding)
        # These are learned pooling weights, not semantic parts or causal proof.
        m=self.models[0];tokens=unit(local);projected=unit(tokens@m['utility_projection'])
        hidden=hidden_features(token_inputs(projected),m['hidden_weights'],m['hidden_bias'])
        utility=predict_utility(hidden,dict(mean_h=m['predictor_mean_h'],mean_y=float(m['predictor_mean_y']),beta=m['predictor_beta']),len(g),64)
        _,attention,_=aggregate_tokens(tokens,utility,float(m['temperature']))
        return embedding.astype(np.float32),attention


def decompose(query,gallery):
    return {'global':float(query[:768]@gallery[:768]),'local_64':float(query[768:832]@gallery[768:832]),'local_192':float(query[832:]@gallery[832:])}


def run(archive,weights,output,batch_size=8,threads=2,limit=None,reviewed_redactions=None,require_reviewed=False):
    from falcon.audit import image_path
    if require_reviewed and reviewed_redactions is None:
        raise ValueError('--require-reviewed requires --reviewed-redactions')
    ledger=None;ledger_records={};ledger_sha=None
    if reviewed_redactions is not None:
        ledger_bytes=Path(reviewed_redactions).read_bytes();ledger_sha=hashlib.sha256(ledger_bytes).hexdigest()
        ledger=json.loads(ledger_bytes)
        if not isinstance(ledger,dict) or ledger.get('reviewed') is not True or not isinstance(ledger.get('records'),list):
            raise ValueError('Reviewed ledger must contain reviewed:true and a records list')
        for record in ledger['records']:
            if not isinstance(record,dict) or not isinstance(record.get('image_id'),str) or not record['image_id']:
                raise ValueError('Invalid image_id in reviewed ledger')
            if record['image_id'] in ledger_records:raise ValueError('Duplicate image_id in reviewed ledger')
            ledger_records[record['image_id']]=record
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    start=time.perf_counter();matrices=[];maps=[];records=[];manifest={};forward=0.;fallbacks=0
    with zipfile.ZipFile(archive) as z:
        names=set(z.namelist());groups={}
        for split in ['test_query','test_gallery']:
            raw=z.read(split+'.csv');manifest[split+'.csv']=hashlib.sha256(raw).hexdigest()
            group=list(csv.DictReader(raw.decode('utf-8-sig').splitlines()))
            if limit is not None:group=group[:limit]
            groups[split]=group
        rows=groups['test_query']+groups['test_gallery'];assert rows
        if ledger is not None:
            expected_ids=[row['image_id'] for row in rows]
            if len(set(expected_ids))!=len(expected_ids):raise ValueError('Duplicate image_id in query/gallery metadata')
            if set(ledger_records)!=set(expected_ids):
                raise ValueError('Reviewed ledger IDs must exactly match processed query+gallery metadata')
        output.mkdir(parents=True,exist_ok=False)
        model=Model(weights,threads)
        for offset in range(0,len(rows),batch_size):
            tensors=[]
            for row in rows[offset:offset+batch_size]:
                raw=z.read(image_path(row['image_id'],names));manifest[row['image_id']]=hashlib.sha256(raw).hexdigest()
                with Image.open(io.BytesIO(raw)) as image:
                    bbox=[float(row[k]) for k in ['x','y','w','h']]
                    if ledger is None:
                        tensor,record,_=model.prepare(image,bbox)
                    else:
                        from falcon.reviewed_redactions import validate_ledger
                        automatic,automatic_fallback=model.masks(image.convert('RGB'),bbox)
                        reviewed=ledger_records[row['image_id']]
                        expected=dict(image_id=row['image_id'],source_sha256=manifest[row['image_id']],bbox=bbox)
                        validate_ledger({'records':[expected]},
                                        {'records':[dict(image_id=row['image_id'],rectangles=automatic)]},
                                        {'reviewed':ledger['reviewed'],'records':[reviewed]})
                        tensor,record,_=model.prepare(image,bbox,masks=reviewed['rectangles'])
                        record.update(reviewed=True,reviewer=reviewed['reviewer'],automatic_fallback=automatic_fallback)
                tensors.append(tensor);records.append(dict(image_id=row['image_id'],**record));fallbacks+=record['fallback']
            tick=time.perf_counter();features,attention=model.encode_tensors(tensors);forward+=time.perf_counter()-tick
            matrices.append(features);maps.append(attention)
            if offset%80==0:print(json.dumps(dict(encoded=min(offset+batch_size,len(rows)),total=len(rows))),flush=True)
    embeddings=np.concatenate(matrices);np.save(output/'embeddings.npy',embeddings,allow_pickle=False)
    np.save(output/'pooling_weights.npy',np.concatenate(maps),allow_pickle=False)
    nq=len(groups['test_query']);gallery_ids=[r['image_id'] for r in groups['test_gallery']]
    scores=embeddings[:nq]@embeddings[nq:].T;order=np.argsort(-scores,axis=1,kind='stable')[:,:min(10,len(gallery_ids))]
    with (output/'submission.csv').open('w',newline='') as f,(output/'candidates.csv').open('w',newline='') as c:
        sw,cw=csv.writer(f),csv.writer(c);sw.writerow(['query_id']+[f'gallery_id_{i}' for i in range(1,11)]);cw.writerow(['query_id','gallery_id','confidence'])
        for i,row in enumerate(groups['test_query']):
            ranked=order[i];sw.writerow([row['image_id']]+[gallery_ids[j] for j in ranked]+['']*(10-len(ranked)))
            accepted=[j for j in ranked if scores[i,j]>=model.threshold]
            if not accepted:cw.writerow([row['image_id'],'',''])
            for j in accepted:cw.writerow([row['image_id'],gallery_ids[j],format(float(scores[i,j]),'.9g')])
    (output/'rows.json').write_text(json.dumps(dict(query=groups['test_query'],gallery=groups['test_gallery']),indent=2)+'\n')
    (output/'redactions.json').write_text(json.dumps(dict(reviewed=ledger is not None,records=records),indent=2)+'\n')
    (output/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'model_manifest.json').write_bytes((Path(weights)/'manifest.json').read_bytes())
    report=dict(model=model.name,rows=len(rows),queries=nq,gallery=len(gallery_ids),dimension=1024,threshold=model.threshold,
                score='cosine similarity, not calibrated probability',weights_bytes=model.manifest['total_bytes'],fallback_redactions=fallbacks,
                seconds=time.perf_counter()-start,forward_seconds=forward,device='cpu',threads=threads,batch_size=batch_size,
                plate_coverage='automatic masks; complete coverage not established',official_score=None,
                hashes={p.name:sha(p) for p in output.iterdir() if p.is_file()})
    if ledger is not None:
        report['plate_coverage']='provided reviewed ledger passed source/BBox bindings and monotonicity; visual completeness not verified by validator'
        report['reviewed_redactions']=dict(source=str(reviewed_redactions),sha256=ledger_sha,validated_records=len(records),
                                          validation_scope='source bytes, exact BBox, reviewer/status and preservation of automatic masks; not human completeness')
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--weights',type=Path,default=Path('weights'));p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--threads',type=int,default=2);p.add_argument('--limit',type=int)
    p.add_argument('--reviewed-redactions',type=Path);p.add_argument('--require-reviewed',action='store_true')
    a=p.parse_args();run(a.archive,a.weights,a.output,a.batch_size,a.threads,a.limit,a.reviewed_redactions,a.require_reviewed)
