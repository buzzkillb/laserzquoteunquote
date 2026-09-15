#!/usr/bin/env python3
"""
BENCHMARK -- optimized targeting vs. the fly's brain
====================================================
Runs the SAME battlefield (identical seed => identical spawns & motion) under
each guidance brain:

    OPTIMIZED : pure engineering fire control (6-state Kalman + lead pursuit)
    FLY BRAIN : insect optic-lobe guidance (EMD/STMD saliency + LGMD looming +
                winner-take-all attention steering the turret)
    HYBRID    : fly attention gates the Kalman fire control (the product)

One command produces, polished and seed-round ready:
  benchmark.mp4           - split-screen head-to-head video (synced clocks)
  benchmark.png           - SF2-style VS end card + verdict
  benchmark_results.json  - full metrics per brain, per seed, + aggregate

Usage:
  python3 benchmark.py --creatures mosquitoes
  python3 benchmark.py --creatures mosquitoes,flies --seeds 7,8,9,10
"""
import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import sim3d
from sim3d import THREATS, SPECIES_ORDER, AREA, HEIGHT, glyph_ms

BG, PANE = "#04060d", "#0b1226"
CYAN, GOLD, RED, GREEN = "#22d3ee", "#fbbf24", "#f87171", "#4ade80"

MODES = [
    ("kalman", "OPTIMIZED", "#22d3ee",
     "KALMAN FIRE CONTROL // 6-state KF + lead pursuit"),
    ("fly", "FLY BRAIN", GREEN,
     "INSECT OPTIC LOBE // EMD+STMD, LGMD looming, WTA"),
    ("hybrid", "HYBRID", GOLD,
     "FLY ATTENTION x KALMAN // bio-gated fire control"),
]


# --------------------------------------------------------------- SF2 end card
def draw_sf2_card(results, seed, out_png):
    """Street Fighter 2 victory screen: chunky type, health-bar kills,
    WINNER banner, PERFECT!, INSERT COIN. Zero ambiguity about who won."""
    fig, fig2 = None, None
    fig = plt.figure(figsize=(12.8, 7.2))
    fig.patch.set_facecolor(BG)

    fig.text(0.5, 0.975, "HERE COMES A NEW CHALLENGER", color=GOLD,
             fontsize=12, family="monospace", ha="center", fontweight="bold")
    winner = max(results, key=lambda r: r["score"])
    rows = sorted(results, key=lambda r: -r["score"])

    # WINNER banner, SF2 red, character-select style
    fig.text(0.5, 0.925, "WINNER", color="#ff5d5d", fontsize=26,
             family="monospace", ha="center", fontweight="bold")
    fig.text(0.5, 0.885, f"{winner['title']}", color=winner["color"],
             fontsize=16, family="monospace", ha="center", fontweight="bold")
    fig.text(0.5, 0.855, winner["subtitle"], color="#8b9dc3", fontsize=9,
             family="monospace", ha="center")

    # VS podium: the three fighters with their kill health-bars
    y = 0.80
    fig.text(0.5, y, "P1        vs        P2        vs        CPU",
             color="#8b9dc3", fontsize=9, family="monospace", ha="center")
    y -= 0.05
    for r in rows:
        mark = ">>" if r is winner else "  "
        fig.text(0.30, y, f"{mark} {r['title']}", color=r["color"],
                 fontsize=12, family="monospace", ha="left",
                 fontweight="bold")
        bar = "\u2588" * int(round(16 * r["kills"] / max(1, r["targets"])))
        empty = "\u2591" * (16 - len(bar))
        fig.text(0.46, y, bar + empty, color=r["color"], fontsize=12,
                 family="monospace", ha="left")
        fig.text(0.66, y, f"{r['kills']:>2}/{r['targets']:<2}", color="#e2e8f0",
                 fontsize=11, family="monospace", ha="left")
        fig.text(0.705, y, f"{r['time']:6.1f}s", color="#8b9dc3",
                 fontsize=10, family="monospace", ha="left")
        fig.text(0.755, y, f"SCORE {r['score']:8.0f}", color=GOLD,
                 fontsize=10, family="monospace", ha="left")
        y -= 0.055

    # post-fight stats table (SF2 score tally style)
    y -= 0.01
    fig.text(0.5, y, "POST-FIGHT TALLY", color=CYAN, fontsize=10,
             family="monospace", ha="center", fontweight="bold")
    y -= 0.04
    hdr = (f"{'':<12}{'KILLS':<8}{'TIME':<9}{'SHOTS':<8}"
           f"{'ENERGY':<9}{'KILLS/SHOT':<11}")
    fig.text(0.5, y, hdr, color="#64748b", fontsize=9, family="monospace",
             ha="center")
    y -= 0.033
    for r in rows:
        eff = 100.0 * r["kills"] / max(1, r["shots"])
        line = (f"{r['title']:<12}{r['kills']:>3}/{r['targets']:<4}"
                f"{r['time']:>6.1f}s {r['shots']:>6}{r['energy']:>8.0f}J "
                f"{eff:>9.1f}%")
        fig.text(0.5, y, line, color=r["color"], fontsize=9.5,
                 family="monospace", ha="center")
        y -= 0.033

    # PERFECT banner if all brains cleared (all targets down)
    if all(r["cleared"] for r in results):
        fig.text(0.5, y - 0.005, "P E R F E C T !", color=CYAN, fontsize=17,
                 family="monospace", ha="center", fontweight="bold")

    # verdict ribbon
    speedup = (max(r["time"] for r in results)
               / max(0.1, min(r["time"] for r in results)))
    fig.text(0.5, 0.085,
             f"VERDICT // {winner['title']} TAKES THE ROUND  --  "
             f"{winner['kills']}/{winner['targets']} IN "
             f"{winner['time']:.1f}s  //  {speedup:.2f}x SPREAD",
             color=GOLD, fontsize=11, family="monospace", ha="center",
             fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.5", facecolor=PANE,
                       edgecolor=GOLD, alpha=0.95))
    fig.text(0.5, 0.038, f"ROUND 1  //  SEED {seed}  //  "
             f"{'+'.join(r['title'] for r in results)}",
             color="#64748b", fontsize=8.5, family="monospace", ha="center")
    fig.text(0.5, 0.015, "INSERT COIN TO CHALLENGE AGAIN",
             color="#475569", fontsize=8, family="monospace", ha="center")
    fig.savefig(out_png, dpi=140, facecolor=BG)
    plt.close(fig)


# ----------------------------------------------------------- split-screen MP4
def render_video(results, seed, creatures, out_mp4, max_fps=25):
    """Polished split-screen: one pane per brain, synced clock, live HUDs."""
    n = len(results)
    labels = {k: THREATS[k]["label"] for k in SPECIES_ORDER}
    colors = {k: THREATS[k]["color"] for k in SPECIES_ORDER}
    glyphs = {k: THREATS[k]["glyph"] for k in SPECIES_ORDER}

    fig = plt.figure(figsize=(12.8, 7.2))
    fig.patch.set_facecolor(BG)
    fig.text(0.5, 0.978, "TARGETING BENCHMARK \u2014 "
             "OPTIMIZED vs FLY BRAIN vs HYBRID", color="w", fontsize=13,
             family="monospace", ha="center", fontweight="bold")
    tgt_lbl = "+".join(THREATS[k]["label"] for k in (creatures or SPECIES_ORDER))
    fig.text(0.5, 0.947, f"{tgt_lbl.upper()} SWEEP  //  SEED {seed}  //  "
             "IDENTICAL BATTLEFIELD \u2014 SAME SPAWNS, SAME PHYSICS",
             color=CYAN, fontsize=8.5, family="monospace", ha="center")

    panes = []
    for i, r in enumerate(results):
        x0 = 0.015 + i * (0.97 / n)
        w = 0.97 / n - 0.025
        ax = fig.add_axes([x0, 0.37, w, 0.555], projection="3d")
        ax.set_facecolor(BG)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_facecolor(PANE); pane.set_alpha(0.28)
            pane.set_edgecolor("#164e63")
        ax.grid(True, color="#155e75", alpha=0.30, lw=0.4)
        ax.set_xlim(0, AREA); ax.set_ylim(0, AREA); ax.set_zlim(0, HEIGHT)
        ax.set_axis_off()
        ax.view_init(elev=22, azim=55)
        fig.text(x0 + (0.97 / n - 0.025) / 2, 0.925, r["title"],
                 color=r["color"], fontsize=12, family="monospace",
                 ha="center", fontweight="bold")
        hud = fig.text(x0 + 0.008, 0.865, "", fontsize=8, family="monospace",
                       va="top", color="#cbd5e1", zorder=10,
                       bbox=dict(boxstyle="round,pad=0.35", facecolor=PANE,
                                 edgecolor=r["color"], alpha=0.85))
        # per-species living bug plots (glyph + size = species identity)
        bug_plots, trail_plots = {}, {}
        for k in SPECIES_ORDER:
            ms = glyph_ms(THREATS[k]["size_m"]) * 0.8
            bug_plots[k] = ax.plot([], [], [], glyphs[k], color=colors[k],
                                   ms=ms, markeredgecolor="w",
                                   markeredgewidth=0.4, alpha=0.95,
                                   linestyle="")[0]
            trail_plots[k] = [ax.plot([], [], [], "-", color=colors[k],
                                      lw=0.8, alpha=0.22)[0]
                              for _ in range(THREATS[k]["count"])]
        beam_glow, = ax.plot([], [], [], color="#ff3d3d", lw=5, alpha=0.20,
                             solid_capstyle="round")
        beam_core, = ax.plot([], [], [], color="#ffcccc", lw=1.8,
                             solid_capstyle="round")
        burst_plots = [ax.plot([], [], [], "*", color="#ffd166", ms=12,
                               alpha=0.0, markeredgecolor="w",
                               linestyle="")[0] for _ in range(10)]
        pane = dict(ax=ax, hud=hud, bugs=bug_plots, trails=trail_plots,
                    beam=(beam_glow, beam_core), bursts=burst_plots,
                    res=r, x0=x0, w=w)

        # ---- NEURON OBSERVATORY (fly & hybrid panes) --------------------
        # Real retinotopic maps + multi-unit scope, driven by the SAME
        # FlyBrain tensors that steer that pane's turret. Kalman pane gets
        # its own telemetry readout for symmetry.
        if r["mode"] in ("fly", "hybrid"):
            bw = w * 0.44
            ax_sal = fig.add_axes([x0 + 0.004, 0.215, bw, 0.098])
            ax_sal.set_title("STMD SALIENCY // RETINOTOPIC 64x32",
                             fontsize=5.6, family="monospace", color=GREEN,
                             pad=1.5)
            im_sal = ax_sal.imshow(np.zeros((32, 64)), cmap="viridis",
                                   origin="lower", vmin=0, vmax=1.2,
                                   aspect="auto", interpolation="nearest")
            wta, = ax_sal.plot([], [], "o", ms=11, mfc="none", mec=GOLD,
                               mew=1.4)
            ax_sal.set_xticks([]); ax_sal.set_yticks([])
            for s in ax_sal.spines.values():
                s.set_color("#155e75")

            ax_emd = fig.add_axes([x0 + 0.004, 0.055, bw, 0.098])
            ax_emd.set_title("EMD MOTION ENERGY // 4-DIRECT. HS/VS ARRAY",
                             fontsize=5.6, family="monospace", color="#60a5fa",
                             pad=1.5)
            im_emd = ax_emd.imshow(np.zeros((32, 64)), cmap="inferno",
                                   origin="lower", vmin=0, vmax=2.0,
                                   aspect="auto", interpolation="nearest")
            ax_emd.set_xticks([]); ax_emd.set_yticks([])
            for s in ax_emd.spines.values():
                s.set_color("#155e75")

            ax_wave = fig.add_axes([x0 + w * 0.50, 0.055, w * 0.48, 0.26])
            ax_wave.set_title("MULTI-UNIT RECORDING // OPTIC LOBE \u2192 LGMD",
                              fontsize=5.6, family="monospace", color=CYAN,
                              pad=1.5)
            ax_wave.set_xlim(0, 160); ax_wave.set_ylim(-0.4, 4.4)
            ax_wave.set_facecolor(BG)
            ax_wave.set_xticks([]); ax_wave.set_yticks([])
            rows = [("EEG", 3.5, "#22d3ee"), ("LGMD", 2.4, "#f87171"),
                    ("STMD", 1.3, "#4ade80"), ("WTA", 0.2, "#fbbf24")]
            wave_lines = {}
            for nm, y0, col in rows:
                (ln,) = ax_wave.plot(np.arange(160), np.full(160, y0),
                                     color=col, lw=0.9)
                wave_lines[nm] = (ln, y0)
                ax_wave.text(1.5, y0 + 0.22, nm, fontsize=4.6,
                             family="monospace", color=col, alpha=0.85)
            for gy in (1.0, 2.1, 3.2):
                ax_wave.axhline(gy + 0.15, color="#155e75", lw=0.3, alpha=0.4)
            kill_txt = ax_wave.text(80, 4.15, "", fontsize=7,
                                    family="monospace", ha="center",
                                    color=GOLD, fontweight="bold")
            for s in ax_wave.spines.values():
                s.set_color("#155e75")
            pane.update(dict(ax_sal=ax_sal, im_sal=im_sal, wta=wta,
                             ax_emd=ax_emd, im_emd=im_emd,
                             ax_wave=ax_wave, wave_lines=wave_lines,
                             kill_txt=kill_txt, last_kills=0))
        else:
            tele = fig.text(x0 + 0.004, 0.33, "", fontsize=7.5,
                            family="monospace", va="top", color="#93c5fd",
                            bbox=dict(boxstyle="round,pad=0.4",
                                      facecolor=PANE, edgecolor="#1e3a5f",
                                      alpha=0.9))
            pane["tele"] = tele
        panes.append(pane)

    max_frames = max(len(r["frames"]) for r in results)

    def draw(fi):
        for i, p in enumerate(panes):
            r = p["res"]
            j = min(fi, len(r["frames"]) - 1)
            f = r["frames"][j]
            ax = p["ax"]
            # living creatures by species glyph
            for k in SPECIES_ORDER:
                pts = f["bugs"].get(k, [])
                if pts:
                    arr = np.array([q[0] for q in pts])
                    p["bugs"][k].set_data(arr[:, 0], arr[:, 1])
                    p["bugs"][k].set_3d_properties(arr[:, 2])
                else:
                    p["bugs"][k].set_data([], [])
                    p["bugs"][k].set_3d_properties([])
                trs = f["trails"].get(k, [])
                for ti, pl in enumerate(p["trails"][k]):
                    if ti < len(trs) and len(trs[ti]) > 1:
                        pl.set_data(trs[ti][:, 0], trs[ti][:, 1])
                        pl.set_3d_properties(trs[ti][:, 2])
                    else:
                        pl.set_data([], [])
                        pl.set_3d_properties([])
            # beam: turret -> aim point (only when firing); turret anchor is
            # the sim3d PanTiltTurret mount, same for every brain
            bg, bc = p["beam"]
            if f.get("firing"):
                tpos = (0.3, sim3d.AREA / 2, 1.5)
                for ln in (bg, bc):
                    ln.set_data([tpos[0], f["aim"][0]],
                                [tpos[1], f["aim"][1]])
                    ln.set_3d_properties([tpos[2], f["aim"][2]])
            else:
                for ln in (bg, bc):
                    ln.set_data([], [])
                    ln.set_3d_properties([])
            # kill bursts
            for bi, pl in enumerate(p["bursts"]):
                if bi < len(f["bursts"]):
                    q, age, _tk = f["bursts"][bi]
                    pl.set_markersize(6 + 20 * (age / 0.6))
                    pl.set_alpha(max(0.0, 1.0 - age / 0.6) * 0.9)
                    pl.set_data([q[0]], [q[1]])
                    pl.set_3d_properties([q[2]])
                else:
                    pl.set_alpha(0.0)
            # gentle synchronized cinematic orbit
            ax.view_init(elev=22 + 3 * math.sin(fi * 0.012),
                         azim=55 + fi * 0.25)
            eff = 100.0 * f["kills"] / max(1, f["shots"])
            p["hud"].set_text(
                f"t {f['t']:6.1f} s\n"
                f"kills {f['kills']:3d}/{r['targets']}\n"
                f"shots {f['shots']:5d}\n"
                f"energy {f['energy']:6.0f} J\n"
                f"eff {eff:5.1f} %")

            # ---- drive the NEURON OBSERVATORY with real brain tensors ----
            if "im_sal" in p:
                p["im_sal"].set_data(f["saliency"])
                att = f["attention"]
                if att is not None:
                    az, el, _sc = att
                    p["wta"].set_data(
                        [(az + math.pi) / (2 * math.pi) * 64],
                        [(el + math.pi / 2) / math.pi * 31])
                p["im_emd"].set_data(f["motion"])
                w = f["waves"]
                for nm, key, gain, cent in (
                        ("EEG", "eeg", 0.9, 0.5),
                        ("LGMD", "looming", 0.9, 0.0),
                        ("STMD", "neurons", 0.9, 0.0),
                        ("WTA", "attention", 0.9, 0.0)):
                    ln, y0 = p["wave_lines"][nm]
                    vals = np.asarray(w[key][-160:], dtype=float)
                    ln.set_ydata(y0 + (vals - cent) * gain)
                # kill-confirmed flash on the scope (decays over ~1 s)
                if f["kills"] > p["last_kills"]:
                    p["last_kills"] = f["kills"]
                    p["flash"] = 1.0
                fl = p.get("flash", 0.0)
                if fl > 0.02:
                    p["kill_txt"].set_text("\u2605 KILL CONFIRMED \u2605")
                    p["kill_txt"].set_alpha(fl)
                    p["flash"] = fl * 0.90
                else:
                    p["kill_txt"].set_text("")
            elif "tele" in p:
                p["tele"].set_text(
                    f"KALMAN TELEMETRY // PURE ENGINEERING\n"
                    f"az {math.degrees(f['az']):7.1f} deg   "
                    f"el {math.degrees(f['el']):6.1f} deg\n"
                    f"tracks {f['n_tracks']:3d}   heat {f['heat']*100:4.0f} %\n"
                    f"neurons used: 0 (boring)\n"
                    f"6-state KF + lead pursuit")
        return []

    ani = animation.FuncAnimation(fig, draw, frames=max_frames + 50,
                                  interval=40, blit=False)
    try:
        w = animation.FFMpegWriter(fps=max_fps, bitrate=4200)
        ani.save(out_mp4, writer=w)
        ok = True
    except Exception as e:
        print(f"[benchmark] MP4 failed ({e}); trying GIF fallback")
        try:
            ani.save(out_mp4.replace(".mp4", ".gif"),
                     writer=animation.PillowWriter(fps=15))
            ok = True
        except Exception as e2:
            print(f"[benchmark] render failed: {e2}")
            ok = False
    plt.close(fig)
    return ok


# ------------------------------------------------------------------ scoring
def score_run(s):
    """Arcade score: kills are points, waste is penalty, speed is bonus,
    precision is style points."""
    eff = 100.0 * s["kills"] / max(1, s["shots"])
    base = s["kills"] * 1000 - s["shots"] * 2 - s["energy"] * 0.5
    if s["cleared"]:
        base += max(0.0, (120.0 - s["duration_s"])) * 10.0
    return round(base + eff, 1)


def run_all(seeds, creatures, seconds):
    """Benchmark every brain on every seed, render-free (fast)."""
    per_seed = {}
    for seed in seeds:
        runs = {}
        for mk, title, color, subtitle in MODES:
            s = sim3d.run_sim(creatures=creatures, seconds=seconds, seed=seed,
                              brain_mode=mk, render=False)
            runs[mk] = dict(title=title, color=color, subtitle=subtitle,
                            mode=mk, kills=s["kills"],
                            targets=s["targets_selected"],
                            shots=s["shots"], energy=s["energy"],
                            time=s["duration_s"], cleared=s["cleared"],
                            score=score_run(s),
                            species_kills=s["species_kills"])
        per_seed[seed] = runs
        print(f"[seed {seed}]  " + "  ".join(
            f"{r['title']} {r['kills']}/{r['targets']} in {r['time']:.1f}s "
            f"(score {r['score']:.0f})" for r in runs.values()))
    return per_seed


def main():
    ap = argparse.ArgumentParser(
        description="Head-to-head targeting benchmark: OPTIMIZED vs FLY BRAIN")
    ap.add_argument("--creatures", default="mosquitoes",
                    help="comma-separated species to target, e.g. "
                         "mosquitoes,flies (or 'all')")
    ap.add_argument("--seeds", default="7,8,9,10",
                    help="comma-separated seeds; each seed = one round")
    ap.add_argument("--seconds", type=float, default=120.0,
                    help="upper bound per engagement")
    ap.add_argument("--video-seed", type=int, default=None,
                    help="seed to render as split-screen MP4 "
                         "(default: first seed)")
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()

    creatures = None
    if a.creatures and a.creatures.lower() != "all":
        creatures = [c.strip() for c in a.creatures.split(",") if c.strip()]
        bad = [c for c in creatures if c not in SPECIES_ORDER]
        if bad:
            ap.error(f"unknown species: {bad} (choose from {SPECIES_ORDER})")
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]

    # 1) benchmark all brains on all seeds (no rendering: fast)
    per_seed = run_all(seeds, creatures, a.seconds)

    # 2) aggregate verdict across rounds
    agg = {}
    for mk, title, color, subtitle in MODES:
        rs = [per_seed[s][mk] for s in seeds]
        wins = 0
        for s in seeds:
            best = max(per_seed[s].values(), key=lambda r: r["score"])
            if best["mode"] == mk:
                wins += 1
        agg[mk] = dict(title=title, wins=wins,
                       rounds=len(seeds),
                       avg_kills=round(np.mean([r["kills"] for r in rs]), 1),
                       avg_time=round(float(np.mean([r["time"] for r in rs])), 2),
                       avg_shots=round(float(np.mean([r["shots"] for r in rs])), 1),
                       avg_eff=round(100.0 * sum(r["kills"] for r in rs)
                                     / max(1, sum(r["shots"] for r in rs)), 1),
                       avg_energy=round(float(np.mean([r["energy"] for r in rs])), 0),
                       clears=sum(1 for r in rs if r["cleared"]))
    champ = max(agg.values(), key=lambda v: (v["wins"], -v["avg_time"]))

    # 3) render the showcase round: split-screen MP4 + SF2 end card
    vs_seed = a.video_seed if a.video_seed is not None else seeds[0]
    results = []
    for mk, title, color, subtitle in MODES:
        s = sim3d.run_sim(creatures=creatures, seconds=a.seconds,
                          seed=vs_seed, brain_mode=mk, render=False,
                          return_frames=True)
        frames = s.pop("frames")
        # slim the frames for memory: keep what the renderer draws
        slim = [dict(t=f["t"], bugs=f["bugs"], trails=f["trails"],
                     bursts=f["bursts"], firing=f["firing"],
                     aim=f["aim"], saliency=f["saliency"],
                     motion=f["motion"], attention=f["attention"],
                     looming=f["looming"], waves=f["waves"],
                     az=f["az"], el=f["el"], heat=f["heat"],
                     n_tracks=len(f["tracks"]),
                     kills=f["kills"], shots=f["shots"],
                     energy=f["energy"]) for f in frames]
        results.append(dict(title=title, color=color, subtitle=subtitle,
                            mode=mk, kills=s["kills"],
                            targets=s["targets_selected"],
                            shots=s["shots"], energy=s["energy"],
                            time=s["duration_s"], cleared=s["cleared"],
                            score=score_run(s), frames=slim))
    out_mp4 = os.path.join(a.outdir, "benchmark.mp4")
    out_png = os.path.join(a.outdir, "benchmark.png")
    mp4_ok = render_video(results, vs_seed, creatures, out_mp4)
    draw_sf2_card(results, vs_seed, out_png)

    # 4) persist results
    summary = dict(
        target=creatures or "all",
        seeds=seeds,
        per_seed={str(k): v for k, v in per_seed.items()},
        aggregate=agg,
        champion=dict(title=champ["title"], wins=f"{champ['wins']}/{len(seeds)}",
                      avg_time=champ["avg_time"], avg_eff=champ["avg_eff"],
                      clears=f"{champ['clears']}/{len(seeds)}"),
        showcase=dict(seed=vs_seed, mp4=out_mp4 if mp4_ok else None,
                      png=out_png),
    )
    out_json = os.path.join(a.outdir, "benchmark_results.json")
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"[benchmark] saved {out_mp4 if mp4_ok else '(mp4 failed)'} / "
          f"{out_png} / {out_json}")
    print(f"CHAMPION: {champ['title']} ({champ['wins']}/{len(seeds)} rounds, "
          f"avg {champ['avg_time']}s, {champ['avg_eff']}% kills/shot)")


if __name__ == "__main__":
    main()
