"""Benchmark eligibility and survey-family reconstruction."""
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEED = 20260927

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read(path):
    path = Path(path)
    text = gzip.decompress(path.read_bytes()).decode() if path.suffix == '.gz' else path.read_text()
    return [json.loads(line) for line in text.splitlines()] if '.jsonl' in path.name else json.loads(text)


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def normalize_text(text):
    return re.sub(r'\s+', ' ', text.lower().replace('’', "'")).strip()


def grouping_keys(q):
    u = urlsplit(q['source'])
    url = urlunsplit((u.scheme.lower(), u.netloc.lower(), u.path.rstrip('/'),
                     urlencode(sorted((k, v) for k, v in parse_qsl(u.query) if not k.lower().startswith('utm_'))), ''))
    keys = [('cluster', str(q['cluster'])), ('series', q['series']), ('source', url)]
    if q['id'].startswith('the-conference-board-measure-of-ceo'):
        keys.append(('manual_cross_publication', 'conference_board_ceo'))
    if len(q['prompt']) >= 40:
        keys.append(('long_question', normalize_text(q['prompt']), tuple(normalize_text(o['label']) for o in q['options'])))
    return keys


def reconstruct(source):
    """Exactly reproduce eligibility, strict deduplication, and historical families."""
    seen = set()
    selected = []
    exclusions = []
    for q in source:
        if q['semantics'] != 'categorical_share':
            exclusions.append({'id': q['id'], 'reason': 'noncategorical', 'semantics': q['semantics']})
            continue
        signature = digest([q[k] for k in ['source', 'prompt', 'audience', 'instructions', 'date', 'cutoff']]
                           + [[(o['label'], o['p']) for o in q['options']]])
        if signature in seen:
            exclusions.append({'id': q['id'], 'reason': 'strict_duplicate'})
        else:
            selected.append(q)
            seen.add(signature)
    parent = list(range(len(selected)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen_keys = {}
    for i, q in enumerate(selected):
        for key in grouping_keys(q):
            if key in seen_keys:
                parent[find(i)] = find(seen_keys[key])
            else:
                seen_keys[key] = i
    components = defaultdict(list)
    for i, q in enumerate(selected):
        components[find(i)].append(q)
    groups = {digest(sorted(q['id'] for q in group))[:16]: group for group in components.values()}
    assignments = []
    for group in sorted(groups):
        for q in sorted(groups[group], key=lambda q: q['id']):
            assignments.append({'id': q['id'], 'group': group,
                                'series': q['series'], 'cluster': q['cluster'], 'source': q['source']})
    return selected, assignments, exclusions
