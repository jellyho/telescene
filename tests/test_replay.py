from __future__ import annotations

import numpy as np

from telescene.recording import Recorder
from telescene.replay import ReplayServer


def take(tmp_path, tracking, quiet=None):
    rec = Recorder(tmp_path / "take.npz")
    for i, on in enumerate(tracking):
        rec.add(tracking=on, quiet=bool(quiet and i in quiet), worn=True, mirror=False, head=np.full(7, i, np.float32),
                body=np.full((24, 7), i, np.float32), grips=np.zeros((2, 7), np.float32), triggers=(0.0, 0.0, 0.0, 0.0))
    return rec.save()


def test_plays_rows_in_order_and_presses_a_where_tracking_changes(tmp_path):
    server = ReplayServer(take(tmp_path, [False, False, True, True, False]))
    frames = [server.latest_input() for _ in range(5)]
    assert [f.body[0, 0] for f in frames] == [0, 1, 2, 3, 4]
    assert [f.a_button for f in frames] == [False, False, True, False, True]
    assert [f.y_button for f in frames] == [False, True, False, False, False]
    assert server.done and server.latest_input() is None


def test_quiet_rows_are_silence(tmp_path):
    server = ReplayServer(take(tmp_path, [False] * 3, quiet={1}), record=False)
    assert server.latest_input() is not None and server.latest_input() is None and server.latest_input() is not None


def test_after_a_fall_it_presses_a_again_once_the_wait_is_over(tmp_path):
    server = ReplayServer(take(tmp_path, [True] * 8), record=False)
    assert server.latest_input().a_button  # row 0 turns tracking on
    server.latest_input()
    server.tracking_lost(wait_ticks=3)  # the session forced it off after row 1
    presses = [server.latest_input().a_button for _ in range(6)]
    assert presses == [False, False, False, True, False, False]  # rows 2-4 wait, row 5 resumes
