# SPDX-License-Identifier: MIT
"""The PC end of the headset link: scene and input over TCP, poses out over UDP.

Nothing here may block the control loop. ``send_poses`` and ``latest_input`` are non-blocking socket
calls, and the only slow part, sending the scene, runs on its own thread. Input is also accepted over
UDP on the same port number, for a headset set to send it that way.
"""

from __future__ import annotations

import logging
import socket
import threading
import time

import numpy as np

from .wire import (BEACON, BEACON_PORT, INPUT_PORT, INPUT_SIZE, POSE_PORT, SCENE_PORT, SCENE_REQUEST_CHOOSE, SCENE_REQUEST_LIST,
                   SCENE_REQUEST_MIRROR, InputFrame, pack_poses, pack_scene_list, unpack_input)

logger = logging.getLogger(__name__)


class HeadsetServer:
    """Serve one scene blob to any client that connects, and hold the connection to detect restarts.

    Args:
        scenes: Names of scenes the headset may pick from its menu, and ``current_scene`` the one running;
            ``scene_groups``, if given, one group per scene, a tab in the menu. A pick sets
            ``requested_scene`` and stops serving; ``TeleopSession.run`` then returns, and whoever started this process
            restarts it with that scene.
    """

    def __init__(self, blob: bytes, port: int = SCENE_PORT, input_port: int = INPUT_PORT, pose_port: int = POSE_PORT,
                 beacon_port: int | None = BEACON_PORT, mirror_blob: bytes | None = None, scenes: list[str] | None = None,
                 current_scene: int = 0, scene_groups: list[str] | None = None):
        self._mirror_blob = mirror_blob
        self.scenes, self.current_scene = list(scenes or []), current_scene
        self.scene_groups = list(scene_groups) if scene_groups is not None else None
        pack_scene_list(self.scenes, 0, self.scene_groups)  # too many scenes, or groups that do not match, fail here
        self.requested_scene: int | None = None
        self._blob, self._pose_port, self._seq = blob, pose_port, 0
        self._input_addr: str | None = None
        self._last: InputFrame | None = None
        self._last_time = 0.0
        self._input_conn: socket.socket | None = None
        self._input_buffer = b""
        self._gap_max = 0.0
        self.packets_sent = self.packets_received = 0
        self._stop = threading.Event()
        self._tcp = socket.create_server(("0.0.0.0", port), reuse_port=False)
        self._tcp.settimeout(0.5)
        self.port = self._tcp.getsockname()[1]
        self._input_listen = socket.create_server(("0.0.0.0", input_port), reuse_port=False)
        self._input_listen.setblocking(False)
        self.input_port = self._input_listen.getsockname()[1]
        self._udp_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp_in.bind(("0.0.0.0", self.input_port))
        self._udp_in.setblocking(False)
        self._udp_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp_out.setblocking(False)
        self._thread = threading.Thread(target=self._serve, name="scene-server", daemon=True)
        self._thread.start()
        self._beacon_thread = None
        if beacon_port is not None:
            self._beacon_thread = threading.Thread(target=self._beacon, args=(beacon_port,), name="beacon", daemon=True)
            self._beacon_thread.start()

    def _beacon(self, port: int) -> None:
        """Once a second, so that an app started before or after this finds the PC. Not on the control path."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        while not self._stop.wait(1.0):
            try:
                sock.sendto(BEACON, ("255.255.255.255", port))
            except OSError:  # no route yet, or the network forbids broadcast: set the address on the headset
                pass
        sock.close()

    def _serve(self) -> None:
        conns: list[socket.socket] = []
        while not self._stop.is_set():
            try:
                conn, addr = self._tcp.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            try:
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                blob = self._blob
                conn.settimeout(0.2)
                try:
                    request = conn.recv(1)
                except TimeoutError:  # a client that asks for nothing gets the main scene
                    request = b""
                if request == SCENE_REQUEST_LIST:
                    conn.settimeout(2.0)
                    conn.sendall(pack_scene_list(self.scenes, self.current_scene, self.scene_groups))
                    conn.close()
                    continue
                if request == SCENE_REQUEST_CHOOSE:
                    conn.settimeout(2.0)
                    index = conn.recv(1)
                    if index and index[0] < len(self.scenes):
                        self.requested_scene = index[0]
                        conn.sendall(b"OK")
                        conn.close()
                        print(f"[telescene] the headset picked scene {self.scenes[index[0]]!r}: restarting", flush=True)
                        break  # serve nothing more from this process: the next scene comes from the restart
                    conn.sendall(b"NO")
                    conn.close()
                    continue
                if request == SCENE_REQUEST_MIRROR and self._mirror_blob is not None:
                    blob = self._mirror_blob
                conn.settimeout(30.0)
                conn.sendall(blob)
                conns = conns[-3:] + [conn]
                logger.info("scene sent to %s, %d bytes", addr[0], len(blob))
                print(f"[telescene] scene sent to {addr[0]}, {len(blob) / 1e6:.2f} MB", flush=True)
            except OSError as exc:
                logger.warning("scene send to %s failed: %s", addr[0], exc)
                conn.close()
        for conn in conns:
            conn.close()

    def _store_input(self, packet: bytes, addr: str) -> None:
        now = time.monotonic()
        if self._last_time:
            self._gap_max = max(self._gap_max, now - self._last_time)
        self._last, self._input_addr, self._last_time = unpack_input(packet), addr, now

    def _drop_input_conn(self) -> None:
        if self._input_conn is not None:
            self._input_conn.close()
        self._input_conn, self._input_buffer = None, b""

    def _drain_tcp_input(self) -> None:
        try:
            conn, addr = self._input_listen.accept()
        except BlockingIOError:
            pass
        else:  # the newest connection wins: an old one may be a headset that lost its link without closing
            self._drop_input_conn()
            conn.setblocking(False)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._input_conn = conn
            print(f"[telescene] input connected from {addr[0]}", flush=True)
        if self._input_conn is None:
            return
        while True:
            try:
                data = self._input_conn.recv(65536)
            except BlockingIOError:
                break
            except OSError:
                self._drop_input_conn()
                return
            if not data:
                self._drop_input_conn()
                return
            self._input_buffer += data
        count = len(self._input_buffer) // INPUT_SIZE
        if not count:
            return
        addr = self._input_conn.getpeername()[0]
        newest = self._input_buffer[(count - 1) * INPUT_SIZE:count * INPUT_SIZE]
        self._input_buffer = self._input_buffer[count * INPUT_SIZE:]
        self.packets_received += count
        try:
            self._store_input(newest, addr)
        except ValueError:  # out of step with the frames; only a fresh connection can recover from that
            logger.warning("bad input frame from %s, dropping the connection", addr)
            self._drop_input_conn()

    def take_input_gap_ms(self) -> float:
        """Longest gap between input arrivals since the last call. Resolution is one control tick."""
        gap, self._gap_max = self._gap_max, 0.0
        return gap * 1000

    def latest_input(self, max_age_s: float = 0.5) -> InputFrame | None:
        """Drain the sockets and return the newest frame, or None if the headset has gone quiet."""
        self._drain_tcp_input()
        while True:
            try:
                packet, addr = self._udp_in.recvfrom(4096)
            except BlockingIOError:
                break
            if len(packet) == INPUT_SIZE:
                self._store_input(packet, addr[0])
                self.packets_received += 1
        return self._last if time.monotonic() - self._last_time < max_age_s else None

    def send_poses(self, poses: np.ndarray, anchor, anchor_yaw: float, world_scale: float, haptics=(0.0, 0.0),
                   status: int = 0) -> None:
        """Poses go to whoever last sent input. Before that there is nobody to send to."""
        if self._input_addr is None:
            return
        self._seq += 1
        try:
            self._udp_out.sendto(pack_poses(self._seq, poses, anchor, anchor_yaw, world_scale, haptics, status),
                                 (self._input_addr, self._pose_port))
            self.packets_sent += 1
        except OSError:  # a full buffer drops this frame; the next one replaces it anyway
            pass

    def close(self) -> None:
        self._stop.set()
        self._tcp.close()
        self._thread.join(timeout=2.0)
        if self._beacon_thread is not None:
            self._beacon_thread.join(timeout=2.0)
        self._drop_input_conn()
        self._input_listen.close()
        self._udp_in.close()
        self._udp_out.close()
