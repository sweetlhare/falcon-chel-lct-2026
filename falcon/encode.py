"""Offline DINO-family embedding baseline, with explicit audited redactions.

No network requests, metadata features, or silent model downloads.
Redactions are image-coordinate rectangles from a separately reviewed manifest.
Their completeness must be validated; presence of a JSON file is not proof.
"""
import argparse
import csv
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from .audit import image_path


def encode(archive, csv_name, weights, redactions, output, device='cpu', batch_size=8, limit=0, timm_name=None,
           selected_rows=None, dense=False, allow_unreviewed_masks=False, full_crop=False, threads=4):
    import torch
    torch.set_num_threads(threads)
    if output.exists():
        raise FileExistsError(output)
    mask_data=json.loads(redactions.read_text())
    if mask_data.get('coordinate_system') != 'full_frame_xyxy':
        raise ValueError('Requires full-frame xyxy redactions')
    if not mask_data.get('reviewed') and not allow_unreviewed_masks:
        raise ValueError('Requires a reviewed full-frame redaction manifest')
    masks=mask_data['images']
    if timm_name:
        import timm
        from timm.models import load_checkpoint
        model=timm.create_model(timm_name,pretrained=False,num_classes=0)
        load_checkpoint(model,str(weights),strict=True)
        data_config=timm.data.resolve_model_data_config(model)
        processor=timm.data.create_transform(**data_config,is_training=False)
    else:
        from transformers import AutoImageProcessor, AutoModel
        processor=AutoImageProcessor.from_pretrained(str(weights),local_files_only=True)
        model=AutoModel.from_pretrained(str(weights),local_files_only=True)
    model=model.to(device).eval()
    output.mkdir(parents=True)
    started=time.monotonic()
    chunks, row_records, batch = [], [], []
    patch_chunks, regional_chunks, local_chunks = [], [], []
    cls_chunks=[]
    inference_seconds=0.0
    def flush():
        nonlocal inference_seconds
        inputs=(torch.stack([processor(im) for im in batch]).to(device) if timm_name
                else processor(images=batch,return_tensors='pt').to(device))
        if device.startswith('cuda'): torch.cuda.synchronize()
        tick=time.monotonic()
        with torch.inference_mode():
            if timm_name:
                states=model.forward_features(inputs)
                cls_chunks.append(torch.nn.functional.normalize(states[:,0].float(),dim=-1).cpu().numpy())
                pooled=model.forward_head(states)
                if dense:
                    patches=states[:,model.num_prefix_tokens:]
                    side=int(patches.shape[1]**0.5)
                    if side*side != patches.shape[1]: raise ValueError('Non-square patch grid')
                    grid=patches.transpose(1,2).reshape(len(batch),-1,side,side)
                    mean=torch.nn.functional.normalize(patches.mean(1).float(),dim=-1)
                    regional=torch.nn.functional.adaptive_avg_pool2d(grid,(2,2)).flatten(2).transpose(1,2)
                    regional=torch.nn.functional.normalize(regional.float(),dim=-1)
                    local=torch.nn.functional.adaptive_avg_pool2d(grid,(8,8)).flatten(2).transpose(1,2)
                    local=torch.nn.functional.normalize(local.float(),dim=-1)
                    patch_chunks.append(mean.cpu().numpy())
                    regional_chunks.append(regional.cpu().numpy())
                    local_chunks.append(local.cpu().numpy().astype(np.float16))
            else:
                result=model(**inputs)
                pooled=getattr(result,'pooler_output',None)
                if pooled is None:
                    states=result.last_hidden_state
                    # ViT's first token is CLS; ConvNeXt maps are spatially pooled.
                    pooled=states[:,0] if states.ndim==3 else states.mean(dim=(-2,-1))
            pooled=torch.nn.functional.normalize(pooled.float(),dim=-1)
        if device.startswith('cuda'): torch.cuda.synchronize()
        inference_seconds+=time.monotonic()-tick
        chunks.append(pooled.cpu().numpy())
        batch.clear()
    with zipfile.ZipFile(archive) as z:
        names=set(z.namelist())
        raw=z.read(csv_name)
        rows=list(csv.DictReader(raw.decode('utf-8-sig').splitlines()))
        indexed=list(enumerate(rows))
        if selected_rows:
            selected={json.loads(line)['image_id'] for line in selected_rows.read_text().splitlines() if line}
            indexed=[(i,r) for i,r in indexed if r['image_id'] in selected]
            if len(indexed)!=len(selected): raise ValueError('Selected rows do not match source CSV')
        indexed=indexed[:limit] if limit else indexed
        for index,row in indexed:
            iid=row['image_id']
            if iid not in masks:
                raise ValueError(f'No reviewed plate redaction for {iid}')
            frame=Image.open(io.BytesIO(z.read(image_path(iid,names)))).convert('RGB')
            draw=ImageDraw.Draw(frame)
            for rectangle in masks[iid]:
                if len(rectangle)!=4: raise ValueError('Mask rectangle must be xyxy')
                draw.rectangle(rectangle,fill=(127,127,127))
            x,y,w,h=(float(row[k]) for k in ('x','y','w','h'))
            if not (0<=x<x+w<=frame.width and 0<=y<y+h<=frame.height):
                raise ValueError(f'BBox out of frame: {iid}')
            crop=frame.crop((round(x),round(y),round(x+w),round(y+h)))
            if full_crop:
                side=max(crop.size)
                dx,dy=side-crop.width,side-crop.height
                crop=ImageOps.expand(crop,(dx//2,dy//2,dx-dx//2,dy-dy//2),fill=(127,127,127))
            batch.append(crop)
            row_records.append({'image_id':iid,'source_row':index})
            if len(batch)==batch_size: flush()
            if len(row_records)%100==0:
                print(json.dumps({'encoded':len(row_records),'selected_total':len(indexed),
                                  'seconds':round(time.monotonic()-started,1)}),flush=True)
        if batch: flush()
    matrix=np.concatenate(chunks)
    np.save(output/'embeddings.npy',matrix,allow_pickle=False)
    if cls_chunks:np.save(output/'cls_token.npy',np.concatenate(cls_chunks),allow_pickle=False)
    if dense:
        np.save(output/'patch_mean.npy',np.concatenate(patch_chunks),allow_pickle=False)
        np.save(output/'regional_2x2.npy',np.concatenate(regional_chunks),allow_pickle=False)
        np.save(output/'local_8x8.npy',np.concatenate(local_chunks),allow_pickle=False)
    (output/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in row_records))
    weight_files=([weights] if weights.is_file() else sorted(p for p in weights.rglob('*')
                   if p.is_file() and p.suffix in ('.safetensors','.bin','.pth','.pt')))
    report={'evidence':'VERIFIED-LOCAL','model_path':str(weights),'rows':len(matrix),
            'dimension':matrix.shape[1],'device':device,'batch_size':batch_size,'threads':threads,
            'wall_seconds':time.monotonic()-started,'forward_seconds':inference_seconds,
            'forward_FPS':len(matrix)/inference_seconds,
            'csv_sha256':hashlib.sha256(raw).hexdigest(),
            'redactions_sha256':hashlib.sha256(redactions.read_bytes()).hexdigest(),
            'weight_bytes':sum(p.stat().st_size for p in weight_files),
            'backend':timm_name or 'transformers',
            'weights_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in weight_files},
            'runtime':{'torch':torch.__version__,'numpy':np.__version__},
            'jury_speed':'not measured','plate_manifest_review_is_not_coverage_proof':True}
    report.update({'mask_reviewed':bool(mask_data.get('reviewed')),
                   'actual_backbone_pool':model.global_pool if timm_name else 'pooler or CLS',
                   'cls_token_saved':bool(cls_chunks),
                   'preprocessing':'square pad127 before backbone transform; full BBox retained' if full_crop else 'backbone default center crop',
                   'quality_status':'provisional automatic-mask diagnostic' if not mask_data.get('reviewed') else 'mask coverage still requires evidence',
                   'dense':dense,'local_regions':'spatial grid, not semantic car parts',
                   'selected_rows_sha256':hashlib.sha256(selected_rows.read_bytes()).hexdigest() if selected_rows else None})
    (output/'extraction.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--csv',required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--redactions',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--timm-name',required=True,
                        help='Explicit offline timm architecture; no model download is attempted')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--selected-rows',type=Path)
    parser.add_argument('--dense',action='store_true')
    parser.add_argument('--full-crop',action='store_true',help='Pad redacted BBox to square before the backbone transform')
    parser.add_argument('--allow-unreviewed-masks',action='store_true',help='Provisional CPU research only; never a plate-compliance claim')
    args=parser.parse_args()
    encode(args.archive,args.csv,args.weights,args.redactions,args.output,args.device,args.batch_size,args.limit,args.timm_name,
           args.selected_rows,args.dense,args.allow_unreviewed_masks,args.full_crop,args.threads)
