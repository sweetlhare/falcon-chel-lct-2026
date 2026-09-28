"""Validate official row order, cosine ranking and frozen rejection decisions."""
import argparse,csv,hashlib,json,zipfile
from pathlib import Path
import numpy as np
from inference import sha

def validate(directory,archive):
    directory=Path(directory);r=json.loads((directory/'report.json').read_text())
    for name,h in r['hashes'].items():assert sha(directory/name)==h,name
    with zipfile.ZipFile(archive) as z:
        query=list(csv.DictReader(z.read('test_query.csv').decode('utf-8-sig').splitlines()))
        gallery=list(csv.DictReader(z.read('test_gallery.csv').decode('utf-8-sig').splitlines()))
    nq,ng=len(query),len(gallery);rows=json.loads((directory/'rows.json').read_text())
    assert rows['query']==query and rows['gallery']==gallery
    e=np.load(directory/'embeddings.npy',allow_pickle=False)
    assert e.dtype==np.float32 and e.shape==(nq+ng,1024) and np.isfinite(e).all()
    assert abs(np.linalg.norm(e,axis=1)-1).max()<1e-5
    ids=[x['image_id'] for x in gallery];scores=e[:nq]@e[nq:].T;order=np.argsort(-scores,axis=1,kind='stable')[:,:10]
    with (directory/'submission.csv').open() as f:submission=list(csv.DictReader(f))
    assert len(submission)==nq
    for i,row in enumerate(submission):
        assert row['query_id']==query[i]['image_id']
        actual=[row[f'gallery_id_{j}'] for j in range(1,11)]
        assert actual==[ids[j] for j in order[i]] and len(set(actual))==10
    expected=[];refused=0
    for i,q in enumerate(query):
        accepted=[j for j in order[i] if scores[i,j]>=r['threshold']]
        if not accepted:expected.append((q['image_id'],'',''));refused+=1
        expected.extend((q['image_id'],ids[j],format(float(scores[i,j]),'.9g')) for j in accepted)
    with (directory/'candidates.csv').open() as f:actual=[tuple(x) for x in list(csv.reader(f))[1:]]
    # FP32 BLAS reduction changes by a few ulps across CPU thread counts.
    # IDs, order and threshold decisions remain exact; only scores get tolerance.
    assert len(actual)==len(expected), 'candidate row count'
    max_score_error=0.0
    for a,b in zip(actual,expected):
        assert a[:2]==b[:2], 'candidate identity/order or rejection mismatch'
        if b[2]=='':assert a[2]=='', 'rejection score must be empty'
        else:
            error=abs(float(a[2])-float(b[2]))
            assert np.isfinite(float(a[2])) and error<=1e-6, 'candidate cosine mismatch'
            max_score_error=max(max_score_error,error)
    assert r['weights_bytes']<2_000_000_000
    result=dict(valid=True,queries=nq,gallery=ng,embeddings_shape=list(e.shape),top10_rows=len(submission),candidate_rows=len(actual),refused_queries=refused,
                max_norm_error=float(abs(np.linalg.norm(e,axis=1)-1).max()),max_candidate_score_error=max_score_error,score_tolerance=1e-6,weights_bytes=r['weights_bytes'],report_sha256=sha(directory/'report.json'),
                scope='Structure and exact cosine provenance only; hidden identities and plate coverage are not validated')
    (directory/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);a=p.parse_args();validate(a.output,a.archive)
