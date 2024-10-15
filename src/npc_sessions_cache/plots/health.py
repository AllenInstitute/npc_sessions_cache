from __future__ import annotations

import io

import tempfile
import pathlib
import datetime

import matplotlib.pyplot as plt
import matplotlib.figure
import upath
import npc_sessions


def get_file_created_time(path: upath.UPath) -> str:
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_path = pathlib.Path(tmpdirname) / 'test.png'
        tmp_path.write_bytes(path.read_bytes())
        return datetime.datetime.fromtimestamp(tmp_path.stat().st_ctime).strftime('%H:%M:%S')

def plot_surface_images(
    session: npc_sessions.DynamicRoutingSession,
) -> tuple[matplotlib.figure.Figure, ...]:
    ctime_to_fig = {}
    figs = []
    for p in (p for p in session.raw_data_paths if 'surface_image' in p.stem):
        try:
            ctime = get_file_created_time(p)
        except:
            ctime = None
        plt.figure()
        plt.imshow(plt.imread(io.BytesIO(p.read_bytes())))
        plt.title(f"{p.stem}\n{ctime or 'time unknown'}")
        plt.gca().axis('off')
        if ctime:
            ctime_to_fig[ctime] = plt.gcf()
        figs.append(plt.gcf())
    if len(figs) == len(ctime_to_fig):
        # sort figures by creation time, if available for all figures
        return tuple(ctime_to_fig[ctime] for ctime in sorted(ctime_to_fig))
    else:
        return tuple(figs)
