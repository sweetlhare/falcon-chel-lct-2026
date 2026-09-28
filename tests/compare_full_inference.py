"""Compare fresh automatic inference with frozen outputs; export only aggregates."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, name):
    if not condition:
        raise ValueError(name)


def max_error(actual, expected, tolerance, name):
    require(actual.shape == expected.shape, name + ' shape')
    require(np.isfinite(actual).all() and np.isfinite(expected).all(), name + ' nonfinite')
    error = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    require(error <= tolerance, name + ' tolerance')
    return error


def compare(reference, actual):
    for name in ['model_manifest', 'source_manifest', 'rows', 'redactions', 'submission']:
        require(actual[name] == reference[name], name + ' equality')
    require(actual['redactions']['reviewed'] is False, 'automatic masks only')
    rows = reference['rows']
    nq, ng = len(rows['query']), len(rows['gallery'])
    for report in [reference['report'], actual['report']]:
        require((report['rows'], report['queries'], report['gallery'], report['dimension']) ==
                (nq + ng, nq, ng, 1024), 'report row counts')
    for field in ['threshold', 'weights_bytes', 'model', 'fallback_redactions', 'threads', 'batch_size']:
        require(actual['report'][field] == reference['report'][field], 'report ' + field)
    expected, fresh = reference['embeddings'], actual['embeddings']
    require(expected.shape == (nq + ng, 1024), 'reference embedding shape')
    require(expected.dtype == fresh.dtype == np.float32, 'embedding dtype')
    embedding_error = max_error(fresh, expected, 1e-4, 'embedding')
    scores_ref, scores_fresh = expected[:nq] @ expected[nq:].T, fresh[:nq] @ fresh[nq:].T
    cosine_error = max_error(scores_fresh, scores_ref, 1e-6, 'full cosine')
    require(len(actual['candidates']) == len(reference['candidates']), 'candidate count')
    score_error = 0.0
    refused = 0
    for previous, current in zip(reference['candidates'], actual['candidates']):
        require(current[:2] == previous[:2], 'candidate identity/order')
        if previous[2] == '':
            require(current[2] == '' and previous[1] == '', 'refused row')
            refused += 1
        else:
            values = np.array([float(current[2]), float(previous[2])], np.float64)
            require(np.isfinite(values).all(), 'candidate nonfinite')
            error = abs(float(values[0] - values[1]))
            require(error <= 1e-6, 'candidate score tolerance')
            score_error = max(score_error, error)
    pooling_error = max_error(actual['pooling'], reference['pooling'], float('inf'), 'pooling')
    return dict(passed=True, queries=nq, gallery=ng, rows=nq+ng,
                source_manifest_equal=True, model_manifest_equal=True, row_order_equal=True,
                redactions_exact_equal=True, automatic_masks=True,
                fallback_redactions=actual['report']['fallback_redactions'],
                top10_exact_equal=True, candidate_identity_order_equal=True,
                candidate_rows=len(actual['candidates']), refused_queries=refused,
                embedding_max_abs_error=embedding_error, embedding_tolerance=1e-4,
                full_cosine_max_abs_error=cosine_error, candidate_score_max_abs_error=score_error,
                score_tolerance=1e-6, pooling_max_abs_error_diagnostic=pooling_error)


def load(directory):
    report = json.loads((directory/'report.json').read_text())
    for name, value in report['hashes'].items():
        require(sha(directory/name) == value, 'recorded hash: ' + name)
    result = {'report': report, 'model_manifest': (directory/'model_manifest.json').read_bytes()}
    for name in ['source_manifest', 'rows', 'redactions']:
        result[name] = json.loads((directory/(name+'.json')).read_text())
    for name in ['submission', 'candidates']:
        with (directory/(name+'.csv')).open(newline='') as stream:
            data = list(csv.reader(stream))
        expected_header = (['query_id']+[f'gallery_id_{i}' for i in range(1,11)] if name == 'submission'
                           else ['query_id', 'gallery_id', 'confidence'])
        require(data[0] == expected_header, name+' header')
        result[name] = data[1:]
    result['embeddings'] = np.load(directory/'embeddings.npy', allow_pickle=False)
    result['pooling'] = np.load(directory/'pooling_weights.npy', allow_pickle=False)
    return result


def self_test():
    vectors = np.zeros((4, 1024), np.float32)
    vectors[0, 0] = vectors[2, 0] = 1
    vectors[1, 1] = vectors[3, 1] = 1
    fixture = dict(model_manifest=b'synthetic', source_manifest={'synthetic': 'fixture'},
                   rows={'query': [{'image_id':'q0'}, {'image_id':'q1'}],
                         'gallery': [{'image_id':'g0'}, {'image_id':'g1'}]},
                   redactions={'reviewed':False, 'records':[]},
                   submission=[['q0','g0','g1'], ['q1','g1','g0']],
                   candidates=[['q0','g0','1'], ['q1','g1','1']],
                   embeddings=vectors, pooling=np.ones((4,64),np.float32)/64,
                   report=dict(rows=4,queries=2,gallery=2,dimension=1024,threshold=.5,
                               weights_bytes=1,model='synthetic',fallback_redactions=0,threads=6,batch_size=8))
    require(compare(fixture, copy.deepcopy(fixture))['embedding_max_abs_error'] == 0, 'good fixture')
    bad_score = copy.deepcopy(fixture); bad_score['candidates'][0][2]='0.99'
    bad_order = copy.deepcopy(fixture); bad_order['submission'][0][1:]=['g1','g0']
    bad_vector = copy.deepcopy(fixture); bad_vector['embeddings'][0,0] += .01
    bad_candidates = copy.deepcopy(fixture); bad_candidates['candidates'].reverse()
    expected = ['candidate score tolerance','submission equality','embedding tolerance','candidate identity/order']
    for corrupted, message in zip([bad_score,bad_order,bad_vector,bad_candidates], expected):
        try:
            compare(fixture, corrupted)
        except ValueError as error:
            require(str(error) == message, 'wrong negative-control failure: '+str(error))
        else:
            raise AssertionError('Known corruption passed: '+message)
    return dict(equal_fixture_passed=True, known_corruptions_rejected=expected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--actual',type=Path)
    parser.add_argument('--report',type=Path)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    controls=self_test()
    if args.self_test:
        print(json.dumps(controls,indent=2))
    else:
        require(all([args.reference,args.actual,args.report]),'missing arguments')
        require(not args.report.exists(),'report already exists')
        result=compare(load(args.reference),load(args.actual))
        result.update(controls=controls, reference_report_sha256=sha(args.reference/'report.json'),
                      fresh_report_sha256=sha(args.actual/'report.json'),
                      comparator_sha256=sha(Path(__file__)),
                      output_hashes={p.name:sha(p) for p in args.actual.iterdir() if p.is_file()},
                      scope='Full automatic inference reproduction; not quality, official score or plate coverage')
        with args.report.open('x') as f:json.dump(result,f,indent=2)
        print(json.dumps(result,indent=2))
