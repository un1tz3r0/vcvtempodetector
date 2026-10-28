import argparse

import numpy as np

from .config import Config
from .detector import detect
from .evaluate import evaluate, load_audio, load_beats, parse_fold
from .synth import write_set


def main():
    ap = argparse.ArgumentParser(prog='beatref', description='Reference beat tracker for the VCV Rack module.')
    ap.add_argument('--set', action='append', default=[], metavar='NAME=VALUE', help='override a Config setting (repeatable)')
    ap.add_argument('--engine', choices=('py', 'cpp'), default='py', help='reference (py) or the module\'s C++ engine (cpp)')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('track', help='track a file and write its beats')
    p.add_argument('audio')
    p.add_argument('--out', help='beats file (default: print)')

    p = sub.add_parser('eval', help='score every annotated file under a directory')
    p.add_argument('root')
    p.add_argument('--annotations', help='directory of <stem>.beats files, if not beside the audio')
    p.add_argument('--json', help='write per-file and mean scores here')
    p.add_argument('--jobs', type=int, help='worker processes (default: one per CPU)')
    p.add_argument('--fold', help="i/k: score only the files `train --fold i/k` held out")

    p = sub.add_parser('compare', help='check the C++ engine against the reference, beat for beat')
    p.add_argument('root')
    p.add_argument('--tolerance', type=float, default=1.0, help='max beat time difference [ms]')
    p.add_argument('--jobs', type=int)

    p = sub.add_parser('train', help='train the neural beat and downbeat network (needs torch)')
    p.add_argument('sources', nargs='+', metavar='ROOT[:ANNOTATIONS]')
    p.add_argument('--out', required=True, help='RTNeural JSON model to write')
    p.add_argument('--fold', help='i/k: hold out fold i of k (for cross-validation)')
    p.add_argument('--cache', default='data/cache', help='feature cache directory')
    p.add_argument('--hyper', action='append', default=[], metavar='NAME=VALUE', help='override a Hyper setting')
    p.add_argument('--jobs', type=int)
    p.add_argument('--threads', type=int, help='torch threads')

    p = sub.add_parser('finetune', help='learning mode: adapt a model to audio with an aligned beat grid (needs torch)')
    p.add_argument('base', help='RTNeural JSON model to start from')
    p.add_argument('sources', nargs='+', metavar='ROOT[:ANNOTATIONS]')
    p.add_argument('--out', required=True, help='adapted model to write')
    p.add_argument('--replay', action='append', default=[], metavar='ROOT[:ANNOTATIONS]',
                   help='base training data to mix in (repeatable; set its share with --tune replay=0.5)')
    p.add_argument('--tune', action='append', default=[], metavar='NAME=VALUE', help='override a Tune setting')
    p.add_argument('--cache', default='data/cache')
    p.add_argument('--jobs', type=int)
    p.add_argument('--threads', type=int)

    p = sub.add_parser('synth', help='render the synthetic test set')
    p.add_argument('root')
    p.add_argument('--count', type=int, default=24)
    p.add_argument('--seed', type=int, default=1)

    p = sub.add_parser('plot', help='render a preview of the module display')
    p.add_argument('audio')
    p.add_argument('--ref', help='annotation file to overlay')
    p.add_argument('--out', default='beatref.png')
    p.add_argument('--start', type=float, default=0.0)
    p.add_argument('--dur', type=float)

    a = ap.parse_args()
    cfg = Config().override(a.set)
    run = detect
    if a.engine == 'cpp':
        from .cpp import detect_cpp as run
    if a.cmd == 'track':
        r = run(*load_audio(a.audio), cfg)
        i = np.arange(len(r.beats))
        pos = i - np.maximum.accumulate(np.where(r.bar, i, 0))  # beats since bar 1
        lines = '\n'.join(f'{t:.3f}\t{p + 1}' for t, p in zip(r.beats, pos))  # can pass beats/bar when bar 1 moves later
        if a.out:
            open(a.out, 'w').write(lines + '\n')
        else:
            print(lines)
        print(f'# {len(r.beats)} beats, median {np.median(r.bpm[r.lock]) if r.lock.any() else 0:.1f} BPM, '
              f'locked {r.lock.mean():.0%}')
    elif a.cmd == 'eval':
        evaluate(a.root, cfg, a.annotations, a.json, a.jobs, a.engine, parse_fold(a.fold))
    elif a.cmd == 'train':
        from .train import Hyper, train
        h = Hyper()
        for pair in a.hyper:
            name, value = pair.split('=', 1)
            setattr(h, name, type(getattr(h, name))(eval(value)) if isinstance(getattr(h, name), tuple)
                    else type(getattr(h, name))(value))
        train(a.sources, a.out, h, parse_fold(a.fold), a.cache, a.jobs, a.threads)
    elif a.cmd == 'finetune':
        from .finetune import Tune, finetune
        t = Tune()
        for pair in a.tune:
            name, value = pair.split('=', 1)
            setattr(t, name, type(getattr(t, name))(value))
        finetune(a.base, a.sources, a.out, t, a.replay, a.cache, a.jobs, a.threads)
    elif a.cmd == 'compare':
        from .compare import compare
        raise SystemExit(0 if compare(a.root, cfg, a.jobs, a.tolerance) else 1)
    elif a.cmd == 'synth':
        write_set(a.root, a.count, a.seed)
    elif a.cmd == 'plot':
        from .plot import plot
        plot(detect(*load_audio(a.audio), cfg), a.out, load_beats(a.ref) if a.ref else None, a.start, a.dur)


if __name__ == '__main__':
    main()
