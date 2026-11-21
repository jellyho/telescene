from __future__ import annotations

import socket
import struct

import pytest

from telescene.progress import StartupProgress, pack_progress


def test_fraction_moves_with_time_and_stops_short_of_a_stage_end(tmp_path):
    p = StartupProgress(["a", "b"], port=0, history=tmp_path / "h.json", guess=10.0)
    t0 = p.since
    assert p.fraction(t0) == 0.0
    assert p.fraction(t0 + 5) == pytest.approx(0.25)
    assert p.fraction(t0 + 100) == pytest.approx(0.475)  # a ran long: held at 95 % of its share
    p.stage("b")
    assert p.fraction(p.since) == pytest.approx(0.5)
    assert p.text() == "b"
    p.done()
    assert p.fraction() == 1.0 and p.text() == "ready"
    again = StartupProgress(["a", "b"], port=0, history=tmp_path / "h.json")
    assert again.expected[0] < 1.0  # a took no time just now, so next time it is expected to be short
    again.done()


def test_the_port_answers_with_one_packet():
    p = StartupProgress(["starting Isaac"], port=0)
    port = p._sock.getsockname()[1]
    with socket.create_connection(("127.0.0.1", port), timeout=2) as s:
        data = b""
        while chunk := s.recv(256):
            data += chunk
    assert data[:4] == b"XPRG" and 0.0 <= struct.unpack("<f", data[4:8])[0] < 1.0
    assert data[9:9 + data[8]] == b"STARTING ISAAC"
    p.done()
    assert pack_progress(1.0, "x")[8] == 1
