"""Bounded CPU batch-1/batch-8 benchmark on the same 16 supplied test images."""
import argparse,csv,io,json,os,platform,time,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from inference import Model,sha
from falcon.audit import image_path

def main(a):
    init=time.perf_counter();model=Model(a.weights,2);init=time.perf_counter()-init
    with zipfile.ZipFile(a.archive) as z:
        names=set(z.namelist());rows=list(csv.DictReader(z.read('test_query.csv').decode('utf-8-sig').splitlines()))[:16]
        images=[]
        for row in rows:
            with Image.open(io.BytesIO(z.read(image_path(row['image_id'],names)))) as im:images.append((im.convert('RGB'),[float(row[k]) for k in ['x','y','w','h']]))
    model.encode_tensors([model.prepare(*images[0])[0]])
    result={}
    outputs={}
    for batch_size in (1,8):
        rounds=[];all_latencies=[]
        for repeat in range(2):
            start=time.perf_counter();vectors=[]
            for offset in range(0,len(images),batch_size):
                tick=time.perf_counter();tensors=[model.prepare(im,bbox)[0] for im,bbox in images[offset:offset+batch_size]]
                e,_=model.encode_tensors(tensors);vectors.append(e);all_latencies.append(time.perf_counter()-tick)
            seconds=time.perf_counter()-start;rounds.append(seconds);outputs[batch_size]=np.concatenate(vectors)
        result[str(batch_size)]={'images_per_round':16,'rounds':2,'seconds':rounds,'FPS':[16/t for t in rounds],
            'batch_latency_p50_seconds':float(np.median(all_latencies)),'batch_latency_p95_seconds':float(np.quantile(all_latencies,.95))}
    diff=float(abs(outputs[1]-outputs[8]).max());assert diff<1e-4
    cpu=next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),platform.processor())
    print(json.dumps(dict(cpu=cpu,threads=2,batches=result,model_initialization_seconds=init,batch_parity_max_error=diff,
        weights_manifest_sha256=sha(Path(a.weights)/'manifest.json'),scope='Warm image preprocessing + plate detector + CHEL encoding; excludes disk JPEG read and network; 16 fixed test images, two rounds; not jury hardware'),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--weights',default='weights');main(p.parse_args())
