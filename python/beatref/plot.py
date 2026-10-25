"""A static preview of the module's display: band-coloured envelope, onsets and beat grid."""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot(result, out, ref=None, start=0.0, dur=None):
    r = result
    stop = start + dur if dur else r.times[-1]
    sel = (r.times >= start) & (r.times <= stop)
    t = r.times[sel]
    in_view = lambda ts: ts[(ts >= start) & (ts <= stop)]  # noqa: E731

    fig, (ax, at, ac) = plt.subplots(3, 1, figsize=(14, 7), sharex=True, height_ratios=[4, 1, 1])
    rgb = np.sqrt(r.bands[sel] / (r.bands[sel].sum(1, keepdims=True) + 1e-12))
    rgb = (rgb / rgb.max(1, keepdims=True).clip(1e-12))[:, [0, 1, 2]] * 0.85
    env = r.env[sel] / (r.env[sel].max() + 1e-12)
    ax.vlines(t, -env, env, colors=rgb, linewidth=1.2)
    ax.plot(t, r.act[sel] * 0.9, color='0.35', linewidth=0.7, label='activation')
    for b in in_view(r.beats):
        ax.axvline(b, color='0.2', linewidth=0.8, alpha=0.6)
    for b in in_view(r.bars):
        ax.axvline(b, color='0.0', linewidth=2.0)
    if ref is not None:
        ax.plot(in_view(ref), np.full(len(in_view(ref)), -1.05), '^', color='0.4', markersize=5, label='reference')
    ax.set_ylim(-1.15, 1.05)
    ax.set_yticks([])
    ax.legend(loc='upper right', fontsize=8)

    at.plot(t, r.bpm[sel], color='0.2')
    at.set_ylabel('BPM')
    ac.plot(t, r.conf[sel], color='0.2', label='confidence')
    ac.fill_between(t, 0, r.lock[sel], color='0.85', step='post', label='lock')
    ac.set_ylim(0, 1)
    ac.set_xlabel('time [s]')
    ac.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
