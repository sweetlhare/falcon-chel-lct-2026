"""Audit actual archive frames, BBoxes, duplicate groups and camera coverage."""
import argparse
import collections
import csv
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image


def image_path(iid, names):
    base = 'images/' + iid
    for path in (base, base+'.jpg', base+'.png'):
        if path in names:
            return path
    raise KeyError(f'Missing image {iid}')


def audit(archive, output, limit=0):
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    objects, by_split = [], {}
    with zipfile.ZipFile(archive) as z:
        names = set(z.namelist())
        for split in ('train', 'test_query', 'test_gallery'):
            raw = z.read(split+'.csv')
            (output / (split+'.csv')).write_bytes(raw)
            rows = list(csv.DictReader(raw.decode('utf-8-sig').splitlines()))
            by_split[split] = rows
            for i, r in enumerate(rows):
                objects.append(dict(r, split=split, source_row=i))
        samples = objects[:limit] if limit else objects
        failures, exact = [], collections.defaultdict(list)
        with (output/'objects.jsonl').open('w') as dst:
            for index, row in enumerate(samples):
                path = image_path(row['image_id'], names)
                raw = z.read(path)
                raw_hash = hashlib.sha256(raw).hexdigest()
                with Image.open(io.BytesIO(raw)) as source:
                    frame = source.convert('RGB')
                x,y,w,h = (float(row[k]) for k in ('x','y','w','h'))
                bounds_ok = 0 <= x < x+w <= frame.width and 0 <= y < y+h <= frame.height
                if not bounds_ok:
                    failures.append({'image_id': row['image_id'], 'split': row['split'],
                                     'bbox': [x,y,w,h], 'frame': [frame.width,frame.height]})
                crop = frame.crop((round(x),round(y),round(x+w),round(y+h)))
                gray = np.asarray(crop.resize((65,64)).convert('L'),dtype=np.int16)
                dhash = np.packbits(gray[:,1:] > gray[:,:-1]).tobytes().hex()
                # dHash is a candidate-generator only, not a duplicate decision.
                exact[raw_hash].append(dict(row))
                pixels = np.asarray(crop.resize((64,64)).convert('L'),dtype=np.float32)
                gx = np.diff(pixels, axis=1); gy = np.diff(pixels, axis=0)
                record = dict(row, frame_width=frame.width, frame_height=frame.height,
                              frame_sha256=raw_hash, crop_dhash=dhash,
                              brightness=float(pixels.mean()),
                              gradient_energy=float((gx*gx).mean()+(gy*gy).mean()),
                              bbox_within_frame=bounds_ok)
                dst.write(json.dumps(record,sort_keys=True)+'\n')
                if (index+1)%500 == 0:
                    print(json.dumps({'audited':index+1,'total':len(samples),'seconds':round(time.monotonic()-started,1)}),flush=True)
    train = by_split['train']
    grouped = collections.defaultdict(list)
    for row in train:
        grouped[row['vehicle_id']].append(row)
    sets = {s:set(r['image_id'] for r in rows) for s,rows in by_split.items()}
    duplicate_groups = [v for v in exact.values() if len(v)>1]
    result = {'evidence':'VERIFIED-LOCAL', 'archive':str(archive), 'limit':limit,
              'objects_audited':len(samples),'objects_total':len(objects),
              'rows':{s:len(rows) for s,rows in by_split.items()},
              'train_id_count':len(grouped),'bbox_out_of_frame':failures,
              'cross_split_image_overlap':{f'{a}:{b}':len(sets[a]&sets[b])
                for a,b in [('train','test_query'),('train','test_gallery'),('test_query','test_gallery')]},
              'cameras_per_id':dict(collections.Counter(len({r['camera_id'] for r in rs}) for rs in grouped.values())),
              'exact_frame_duplicate_groups':duplicate_groups,
              'near_duplicates':'dHash candidates stored, no claim of verified near-duplicate absence',
              'plate_redaction':'not verified by this metadata/quality audit',
              'wall_seconds':time.monotonic()-started}
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('bbox_out_of_frame','exact_frame_duplicate_groups')},indent=2))
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args()
    audit(args.archive,args.output,args.limit)
