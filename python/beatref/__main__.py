import argparse

import numpy as np

from .config import Config
from .detector import detect
from .evaluate import evaluate, load_audio, load_beats
from .synth import write_set


def main():
    ap = argparse.ArgumentParser(prog='beatref', description='Reference beat tracker for the VCV Rack module.')
    ap.add_argument('--set', action='append', default=[], metavar='NAME=VALUE', help='override a Config setting (repeatable)')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('track', help='track a file and write its beats')
    p.add_argument('audio')
    p.add_argument('--out', help='beats file (default: print)')

    p = sub.add_parser('eval', help='score every annotated file under a directory')
    p.add_argument('root')
    p.add_argument('--annotations', help='directory of <stem>.beats files, if not beside the audio')
    p.add_argument('--json', help='write per-file and mean scores here')
    p.add_argument('--jobs', type=int, help='worker processes (default: one per CPU)')

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
    if a.cmd == 'track':
        r = detect(*load_audio(a.audio), cfg)
        lines = '\n'.join(f'{t:.3f}\t{(i - r.index[0]) % cfg.beats_per_bar + 1}' for t, i in zip(r.beats, r.index))
        if a.out:
            open(a.out, 'w').write(lines + '\n')
        else:
            print(lines)
        print(f'# {len(r.beats)} beats, median {np.median(r.bpm[r.lock]) if r.lock.any() else 0:.1f} BPM, '
              f'locked {r.lock.mean():.0%}')
    elif a.cmd == 'eval':
        evaluate(a.root, cfg, a.annotations, a.json, a.jobs)
    elif a.cmd == 'synth':
        write_set(a.root, a.count, a.seed)
    elif a.cmd == 'plot':
        from .plot import plot
        plot(detect(*load_audio(a.audio), cfg), a.out, load_beats(a.ref) if a.ref else None, a.start, a.dur)


if __name__ == '__main__':
    main()
