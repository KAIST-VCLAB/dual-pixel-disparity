import json
from pathlib import Path

from easydict import EasyDict as edict


def _make_edict(d):
    '''
    Converting dictionary to dot-callable easydict.
    Automatically increase the level when value is dictionary.
    Recursive call.
    '''
    args = edict()
    for k, v in d.items():
        if isinstance(v, dict):
            sub_args = _make_edict(v)
            setattr(args, k, sub_args)
        else:
            setattr(args, k, v)
    return args


def print_config(args, prefix=''):
    '''
    Print args. Recursive call.
    '''
    current_prefix = prefix
    for name, sub in args.items():
        sub = getattr(args, name)
        if isinstance(sub, dict):
            print_config(sub, f'{current_prefix}.{name}')
        else:
            print(f'{current_prefix}.{name} = {sub}')


def load_config(fpath: str):
    _path = Path(fpath)
    assert _path.exists()
    with _path.open('r') as f: _c = json.load(f)
    return _make_edict(_c)

