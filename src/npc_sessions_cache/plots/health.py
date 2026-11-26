from __future__ import annotations

import datetime
import io
import pathlib
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import npc_sessions
import PIL.Image
import upath


def get_file_created_time(path: upath.UPath) -> str:
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_path = pathlib.Path(tmpdirname) / "test.png"
        tmp_path.write_bytes(path.read_bytes())
        return datetime.datetime.fromtimestamp(tmp_path.stat().st_ctime).strftime(
            "%H:%M:%S"
        )


def plot_surface_images(
    session: npc_sessions.DynamicRoutingSession,
) -> tuple[plt.Figure, ...]:
    ctime_to_fig = {}
    figs = []
    for p in (p for p in session.raw_data_paths if "surface_image" in p.stem):
        try:
            ctime = get_file_created_time(p)
        except:
            ctime = None
        plt.figure()
        # sniff the format from content, not suffix: some surface images are BMPs
        # named .png (e.g. 810752_2025-10-07, 814666_2025-11-06)
        with PIL.Image.open(io.BytesIO(p.read_bytes())) as img:
            plt.imshow(np.asarray(img))
        plt.title(f"{p.stem}\n{ctime or 'time unknown'}")
        plt.gca().axis("off")
        if ctime:
            ctime_to_fig[ctime] = plt.gcf()
        figs.append(plt.gcf())
    if len(figs) == len(ctime_to_fig):
        # sort figures by creation time, if available for all figures
        return tuple(ctime_to_fig[ctime] for ctime in sorted(ctime_to_fig))
    else:
        return tuple(figs)
