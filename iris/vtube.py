"""Drive a Live2D avatar in VTube Studio.

Two things go over the plugin API:

- **Expressions.** Her replies carry emotion cues like [smug], parsed out before
  speech and, until now, thrown away. Each is matched to a hotkey in the loaded
  model by name.
- **Lip sync.** The usual approach routes speaker output through a virtual audio
  cable into VTube Studio's microphone input. This machine has no such cable and
  does not need one: the audio is already in the player's buffer, so the mouth
  parameter is driven straight from its level. No extra software, no device
  routing, and it cannot pick up room noise.

Nothing here may break a conversation. VTube Studio not running, the API
switched off, authentication refused, a model with no matching hotkeys -- each
of those leaves her talking normally with a still avatar.
"""

import json
import queue
import threading
import time
import uuid

from .config import Config
from .motion import IdleMotion

PLUGIN = "Iris"
DEVELOPER = "Iris local voice AI"

# Built-in input parameter: 0 is a closed mouth, 1 is wide open.
MOUTH = "MouthOpen"


class VTubeStudio:
    """A deliberately small synchronous client. One request at a time."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ws = None
        self.ok = False
        self.model = None
        self.hotkeys: dict[str, str] = {}       # lowercased name -> hotkeyID
        self.all_hotkeys: list[dict] = []       # every hotkey, unnamed ones too
        self.available: list[dict] = []         # expressions the model exposes
        self.expressions: dict[str, str] = {}   # emotion tag -> expression file
        self.active: str | None = None          # expression currently showing
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._mouth_thread = None
        self._cue_thread = None
        self._last_attempt = 0.0
        self.token_file = cfg.vtube_token

    # -- plumbing ----------------------------------------------------------
    def _send(self, message_type: str, data: dict | None = None,
              timeout: float | None = None) -> dict | None:
        """One request, one response. Returns None on any failure."""
        if self.ws is None:
            return None
        payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": uuid.uuid4().hex[:8],
            "messageType": message_type,
            "data": data or {},
        }
        try:
            with self._lock:
                self.ws.settimeout(timeout or self.cfg.vtube_timeout)
                self.ws.send(json.dumps(payload))
                reply = json.loads(self.ws.recv())
        except Exception as exc:
            self._fail(f"{message_type}: {exc}")
            return None
        if reply.get("messageType") == "APIError":
            detail = reply.get("data", {})
            print(f"  [vtube studio: {detail.get('message', detail)}]", flush=True)
            return None
        return reply.get("data", {})

    def _fail(self, why: str) -> None:
        if self.ok:
            print(f"  [vtube studio disconnected: {why}]", flush=True)
        self.ok = False
        self.ws = None

    # -- connection --------------------------------------------------------
    def connect(self, quiet: bool = False) -> bool:
        try:
            import websocket
        except ImportError:
            print("  [vtube studio: pip install websocket-client]", flush=True)
            return False

        url = f"ws://{self.cfg.vtube_host}:{self.cfg.vtube_port}"
        self._last_attempt = time.monotonic()
        try:
            self.ws = websocket.create_connection(url, timeout=self.cfg.vtube_timeout)
        except Exception:
            if not quiet:
                print(f"  [vtube studio not reachable at {url}; avatar off]",
                      flush=True)
            self.ws = None
            return False

        if not self._authenticate(quiet=quiet):
            self.ws = None
            return False

        self.ok = True
        # A restarted VTube Studio has no expression showing and may have loaded
        # a different model, so nothing about the old session carries over.
        self.active = None
        self._load_hotkeys(quiet=quiet)
        return True

    def ensure(self) -> None:
        """Reconnect if VTube Studio has come back, at most every few seconds.

        Restarting VTS used to mean restarting Iris: the first failed send tore
        the connection down and nothing ever tried again. Retrying is quiet by
        design -- the token is already stored, so a successful reconnect needs
        no interaction, and a failed one must not fill the transcript with
        noise while she is talking.
        """
        if self.ok or self._stop.is_set():
            return
        if time.monotonic() - self._last_attempt < self.cfg.vtube_reconnect_s:
            return
        if self.connect(quiet=True):
            print("  [vtube studio: reconnected]", flush=True)

    def _authenticate(self, quiet: bool = False) -> bool:
        token = None
        if self.token_file.exists():
            token = self.token_file.read_text(encoding="utf-8").strip() or None
        if token and self._try_token(token):
            return True
        if quiet:
            # An automatic retry must never pop a dialog the user did not ask
            # for. Without a usable token this is a job for the next start.
            return False

        # No token, or the stored one was rejected. Asking for a new one pops a
        # dialog in VTube Studio that a human has to accept, so it gets a long
        # timeout rather than the usual one.
        print("  [vtube studio: approve the plugin request in the VTS window]",
              flush=True)
        data = self._send(
            "AuthenticationTokenRequest",
            {"pluginName": PLUGIN, "pluginDeveloper": DEVELOPER},
            timeout=self.cfg.vtube_auth_timeout,
        )
        if not data or "authenticationToken" not in data:
            print("  [vtube studio: authentication refused; avatar off]", flush=True)
            return False
        token = data["authenticationToken"]
        self.token_file.write_text(token, encoding="utf-8")
        return self._try_token(token)

    def _try_token(self, token: str) -> bool:
        data = self._send(
            "AuthenticationRequest",
            {"pluginName": PLUGIN, "pluginDeveloper": DEVELOPER,
             "authenticationToken": token},
        )
        return bool(data and data.get("authenticated"))

    # -- expressions -------------------------------------------------------
    def _load_hotkeys(self, quiet: bool = False) -> None:
        """Discover expressions and hotkeys, and map emotion cues onto them.

        Expressions are driven through ExpressionActivationRequest, not through
        their hotkeys. The hotkeys are of type ToggleExpression, so firing
        [smug] and then [happy] would leave both switched on and her face would
        silently accumulate every emotion of the conversation. Explicit
        activate/deactivate is idempotent and always lands where intended.
        """
        data = self._send("HotkeysInCurrentModelRequest", {})
        if data:
            self.model = data.get("modelName")
            # Keyed by id, not name: this model has six unnamed animation
            # hotkeys, and a name-keyed dict silently collapses them into one.
            self.all_hotkeys = [
                {"id": h.get("hotkeyID"), "name": h.get("name", "").strip(),
                 "type": h.get("type", ""), "file": h.get("file", "")}
                for h in data.get("availableHotkeys", []) if h.get("hotkeyID")
            ]
            self.hotkeys = {
                h["name"].lower(): h["id"] for h in self.all_hotkeys if h["name"]
            }

        state = self._send("ExpressionStateRequest", {"details": True})
        self.available = [
            {"file": e.get("file", ""), "name": (e.get("name") or "").strip()}
            for e in (state or {}).get("expressions", [])
        ]

        # An emotion matches by configured name, then exact name, then substring,
        # so "Smug face" and "expression_smug" both work with nothing renamed.
        self.expressions = {}
        for tag in self.cfg.vtube_emotions:
            wanted = self.cfg.vtube_map.get(tag, tag).strip().lower()
            if not wanted:
                continue
            match = next(
                (e["file"] for e in self.available if e["name"].lower() == wanted),
                None,
            ) or next(
                (e["file"] for e in self.available
                 if wanted in e["name"].lower() or wanted in e["file"].lower()),
                None,
            )
            if match:
                self.expressions[tag] = match
        if quiet:
            return
        found = ", ".join(sorted(self.expressions)) or "none"
        print(f"  [vtube studio: model '{self.model}', {len(self.available)} "
              f"expressions, {len(self.all_hotkeys)} hotkeys, cues mapped: {found}]",
              flush=True)

    def _activate(self, expression_file: str, on: bool) -> None:
        self._send("ExpressionActivationRequest",
                   {"expressionFile": expression_file, "active": on})

    def express(self, tag: str) -> None:
        """Show one emotion, replacing whatever was showing before.

        'neutral' -- or a cue this model has no expression for -- clears her
        face rather than leaving the last one stuck on.
        """
        if not self.ok:
            return
        wanted = self.expressions.get(tag)
        if wanted == self.active:
            return
        if self.active:
            self._activate(self.active, False)
        self.active = None
        if wanted:
            self._activate(wanted, True)
            self.active = wanted

    def clear(self) -> None:
        """Drop any expression currently showing."""
        if self.ok and self.active:
            self._activate(self.active, False)
        self.active = None

    def trigger(self, name: str) -> bool:
        """Fire any hotkey by name -- used by scripts/avatar.py to try them out."""
        if not self.ok:
            return False
        hotkey = self.hotkeys.get(name.strip().lower())
        if not hotkey:
            return False
        return self._send("HotkeyTriggerRequest", {"hotkeyID": hotkey}) is not None

    # -- lip sync and motion -----------------------------------------------
    def inject(self, params: dict[str, float]) -> None:
        """Set several tracking parameters in one request.

        One message rather than one per parameter: this client is synchronous
        and does a full round trip per send, so a dozen separate injections
        would cost a dozen round trips and drop the frame rate through the
        floor. VTube Studio takes the whole list at once.

        `faceFound` stays False. It was worth checking -- injected values land
        identically whether it is set or not, measured against a live model --
        so the honest setting is the one that does not claim a face was seen.
        """
        if not self.ok or not params:
            return
        self._send(
            "InjectParameterDataRequest",
            {
                "faceFound": False,
                "mode": "set",
                "parameterValues": [
                    {"id": name, "value": value, "weight": 1.0}
                    for name, value in params.items()
                ],
            },
        )

    def set_mouth(self, value: float) -> None:
        self.inject({MOUTH: max(0.0, min(1.0, value))})

    def start_lipsync(self, player) -> None:
        """Follow the player's output level and move the mouth with it."""
        if self._mouth_thread is not None:
            return
        self._mouth_thread = threading.Thread(
            target=self._lipsync_loop, args=(player,), daemon=True
        )
        self._mouth_thread.start()

    def _lipsync_loop(self, player) -> None:
        period = 1.0 / self.cfg.vtube_lipsync_fps
        attack, release = self.cfg.vtube_mouth_attack, self.cfg.vtube_mouth_release
        smoothed = 0.0
        motion = IdleMotion(self.cfg) if self.cfg.vtube_motion else None
        started = time.perf_counter()
        while not self._stop.is_set():
            tick = time.perf_counter()
            if not self.ok:
                # This thread owns reconnection: it is the one that runs
                # whether or not she is talking. Poll slowly here rather than
                # spinning at the lip sync frame rate against a dead socket.
                self.ensure()
                smoothed = 0.0
                self._stop.wait(self.cfg.vtube_reconnect_s / 2)
                continue
            level = min(1.0, player.level * self.cfg.vtube_mouth_gain)
            # Asymmetric smoothing: open fast on a syllable, close slower, or the
            # mouth flutters on every glottal stop.
            rate = attack if level > smoothed else release
            smoothed += (level - smoothed) * rate
            frame = {MOUTH: max(0.0, min(1.0, smoothed))}
            if motion is not None:
                # Driven by the smoothed level, not the raw one, so her head
                # settles on the same curve her mouth closes on.
                frame.update(motion.frame(tick - started, smoothed))
            self.inject(frame)
            # Each update is a round trip, so sleeping a fixed period undershoots
            # the frame rate badly -- 20/s when asking for 30. Sleep the balance.
            time.sleep(max(0.0, period - (time.perf_counter() - tick)))

    # -- expression cues ---------------------------------------------------
    def start_cues(self, player) -> None:
        """Apply emotion cues as the clips carrying them start playing.

        Kept off the audio callback: an expression change is a websocket round
        trip, and blocking the device callback on one would glitch the audio.
        """
        if self._cue_thread is not None:
            return
        self._cue_thread = threading.Thread(
            target=self._cue_loop, args=(player,), daemon=True
        )
        self._cue_thread.start()

    def _cue_loop(self, player) -> None:
        while not self._stop.is_set():
            try:
                tag = player.cues.get(timeout=0.2)
            except queue.Empty:
                continue
            self.express(tag)

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        self._stop.set()
        if self.ws is not None:
            try:
                self.clear()          # do not leave her face stuck mid-emotion
                # ...nor her head frozen at whatever angle the last frame hit.
                rest = {MOUTH: 0.0}
                if self.cfg.vtube_motion:
                    rest.update(IdleMotion(self.cfg).rest())
                self.inject(rest)
                self.ws.close()
            except Exception:
                pass
        self.ok = False
        self.ws = None
