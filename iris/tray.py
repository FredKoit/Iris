"""A tray icon and global hotkeys, so she can be managed without the terminal.

Once she is floating on the desktop the console is usually buried, and the
things you most want to do -- shut her up, stop her talking to herself, get the
mouse back -- are exactly the ones you want while some other window has focus.
So the same four controls are on both a tray menu and a global hotkey.

Both halves are optional. `pystray` and `pynput` are not needed to hold a
conversation, and following the same rule as VTube Studio, a missing one prints
how to get it and is otherwise silently absent.
"""

import threading


# The icon says one thing: whether she can hear you right now.
#
#   solid dot     listening
#   hollow ring   push-to-talk is armed and the key is up
#   struck out    microphone deliberately off
#
# Drawn at 4x and downsampled, because ImageDraw does not antialias anything.
# At 64px the disc edge is visibly stepped and the slash has ragged sides, and
# the tray then scales that down again to 16 or 20 px, which sharpens the
# stepping rather than hiding it. Supersampling is the whole difference between
# this looking drawn and looking rendered.
_SIZE = 64
_SCALE = 4

_HER_PURPLE = (196, 122, 210, 255)
_OFF_GREY = (124, 124, 134, 255)
_SLASH = (240, 240, 245, 255)
_CLEAR = (0, 0, 0, 0)

# Rendered once each. A held push-to-talk key redraws on every press and
# release, and three fixed images do not need making twice.
_CACHE: dict[tuple[bool, bool], object] = {}


def _icon_image(muted: bool, waiting: bool = False):
    """A dot in her colour.

    `muted` wins: a microphone switched off deliberately is off whatever the
    talk key is doing, and both are true at once whenever she is muted from the
    tray while push-to-talk is armed. Resolving it here rather than in the
    drawing keeps that pair to one cached image instead of two identical ones.
    """
    key = (muted, waiting and not muted)
    if key not in _CACHE:
        _CACHE[key] = _render(muted, waiting)
    return _CACHE[key]


def _render(muted: bool, waiting: bool):
    from PIL import Image, ImageChops, ImageDraw

    size = _SIZE * _SCALE
    img = Image.new("RGBA", (size, size), _CLEAR)
    draw = ImageDraw.Draw(img)

    def at(*fractions):
        return tuple(f * size for f in fractions)

    disc = at(0.09, 0.09, 0.91, 0.91)
    draw.ellipse(disc, fill=_OFF_GREY if muted else _HER_PURPLE)

    if waiting and not muted:
        # Hollow, not dimmed: armed but not hearing anything. A ring keeps its
        # meaning when the tray scales it to 16 px, where a subtler treatment
        # -- a paler fill, a smaller dot -- just reads as the same icon again.
        draw.ellipse(at(0.32, 0.32, 0.68, 0.68), fill=_CLEAR)

    if muted:
        # ImageDraw writes pixels rather than compositing them, so drawing in
        # a fully transparent colour cuts a hole. The gap is laid down first
        # and wider than the stroke, which is what makes this read as a slash
        # over the dot rather than a scratch in it.
        #
        # Both are drawn well past the rim and then clipped back to the disc.
        # Left to overhang they read as a line laid across the icon instead of
        # a badge on it; stopped short of the rim they end in blunt square
        # caps, because ImageDraw has no cap style. Clipping avoids both.
        for fraction, colour in ((0.155, _CLEAR), (0.085, _SLASH)):
            draw.line(at(0.02, 0.98, 0.98, 0.02),
                      fill=colour, width=int(fraction * size))
        mask = Image.new("L", (size, size), 0)
        ImageDraw.Draw(mask).ellipse(disc, fill=255)
        img.putalpha(ImageChops.multiply(img.getchannel("A"), mask))

    return img.resize((_SIZE, _SIZE), Image.LANCZOS)


class PushToTalk:
    """Hold a key to open the microphone.

    pynput's `GlobalHotKeys` fires on the way down and says nothing at all on
    the way up, which is exactly half of what this needs, so the chord is
    tracked by hand off a raw listener. `HotKey.parse` plus `canonical` is the
    same pairing `GlobalHotKeys` uses internally, and it is what makes
    "<ctrl>+<alt>+t" match the ctrl_l the keyboard actually reports.

    Key repeat is free: the held keys are a set, and the callback only fires
    when the chord starts or stops being satisfied.
    """

    def __init__(self, chord: str, on_change):
        from pynput import keyboard

        self._wanted = set(keyboard.HotKey.parse(chord))
        self._down: set = set()
        self._held = False
        self._on_change = on_change
        self._listener = keyboard.Listener(
            on_press=self._press, on_release=self._release
        )

    def start(self) -> None:
        self._listener.start()

    def stop(self) -> None:
        try:
            self._listener.stop()
        except Exception:
            pass
        # Never leave her stuck open, or stuck deaf, because the key happened
        # to be down when this was torn down.
        if self._held:
            self._held = False
            self._on_change(False)

    def _press(self, key) -> None:
        self._down.add(self._listener.canonical(key))
        self._settle()

    def _release(self, key) -> None:
        self._down.discard(self._listener.canonical(key))
        self._settle()

    def _settle(self) -> None:
        held = self._wanted <= self._down
        if held != self._held:
            self._held = held
            try:
                self._on_change(held)
            except Exception as exc:
                print(f"  [push to talk: {exc}]", flush=True)


class Controls:
    """Tray menu and hotkeys over an `Iris`.

    Every action is a plain call onto the app, which owns the state. This class
    holds no truth of its own beyond what the menu is currently drawn as.
    """

    def __init__(self, iris, cfg):
        self.iris = iris
        self.cfg = cfg
        self.icon = None
        self.hotkeys = None
        self.talk = None
        self._thread = None

    # -- actions -----------------------------------------------------------
    def toggle_mic(self, *_args) -> None:
        self.iris.set_mic_off(not self.iris.mic_off.is_set())
        self._refresh()

    def toggle_idle(self, *_args) -> None:
        self.iris.set_idle_paused(not self.iris.idle_paused.is_set())
        self._refresh()

    def stop_speaking(self, *_args) -> None:
        self.iris.stop_speaking()

    def toggle_click_through(self, *_args) -> None:
        try:
            from .overlay import toggle_click_through
        except Exception as exc:            # not Windows, or user32 unavailable
            print(f"  [click-through unavailable: {exc}]", flush=True)
            return
        state = toggle_click_through()
        if state is None:
            print("  [click-through: VTube Studio window not found]", flush=True)
        else:
            print(f"  [clicks {'pass through' if state else 'land on her'}]",
                  flush=True)
        self._refresh()

    def quit(self, *_args) -> None:
        self.iris.quit()
        self.stop()

    # -- tray --------------------------------------------------------------
    def _menu(self):
        import pystray

        item = pystray.MenuItem
        return pystray.Menu(
            item(lambda _i: ("Unmute microphone" if self.iris.mic_off.is_set()
                             else "Mute microphone"), self.toggle_mic),
            item(lambda _i: ("Resume idle chatter"
                             if self.iris.idle_paused.is_set()
                             else "Pause idle chatter"), self.toggle_idle),
            item("Stop speaking", self.stop_speaking),
            item("Toggle click-through", self.toggle_click_through),
            pystray.Menu.SEPARATOR,
            item("Quit Iris", self.quit),
        )

    def _title(self) -> str:
        """The hover text. Worth saying which key, since the icon cannot."""
        if self.iris.mic_off.is_set():
            return "Iris - microphone off"
        if self.iris.waiting_to_talk:
            return f"Iris - hold {self.cfg.hotkey_talk} to speak"
        return "Iris - listening"

    def _refresh(self, menu: bool = True) -> None:
        """Redraw the icon, and by default the menu labels with it.

        `menu` is off for push-to-talk, which redraws on every press and
        release of a key somebody may be holding down: the labels cannot have
        changed, and rebuilding them at that rate is work for nothing.
        """
        if self.icon is None:
            return
        try:
            self.icon.icon = _icon_image(self.iris.mic_off.is_set(),
                                         self.iris.waiting_to_talk)
            self.icon.title = self._title()
            if menu:
                self.icon.update_menu()
        except Exception:
            pass          # a tray that will not redraw must not end a conversation

    def start(self) -> bool:
        """Bring up whichever parts are wanted and installed. True if any did.

        Push-to-talk is checked separately from `tray`, so `--no-tray
        --push-to-talk` gets the key without the icon.
        """
        started = False
        if self.cfg.tray:
            started |= self._start_tray()
            started |= self._start_hotkeys()
        if self.cfg.push_to_talk:
            started |= self._start_push_to_talk()
        # The icon is created before push-to-talk is armed, so it starts drawn
        # as listening whether or not she is. One redraw once everything is up
        # settles it, rather than leaving it wrong until the first keypress.
        self._refresh()
        return started

    def _on_talk(self, held: bool) -> None:
        """The talk key going down or up. Opens the gate, then shows it.

        The microphone moves first and the icon follows, so a slow redraw can
        never delay her hearing you.
        """
        self.iris.set_talk_held(held)
        self._refresh(menu=False)

    def _start_push_to_talk(self) -> bool:
        """Arm the talk key, and leave her listening normally if it will not.

        The failure matters more than usual here: with push-to-talk on and no
        key listener, the gate would never open and she would be silently,
        permanently deaf. So nothing is armed until the listener is actually
        running, and _update_mic() reads that rather than the config flag.
        """
        chord = (self.cfg.hotkey_talk or "").strip()
        if not chord:
            return False
        try:
            from pynput import keyboard                       # noqa: F401
            self.talk = PushToTalk(chord, self._on_talk)
            self.talk.start()
        except ImportError:
            print("  [push to talk: pip install pynput; still listening]",
                  flush=True)
            self.talk = None
            return False
        except Exception as exc:
            print(f"  [push to talk unavailable: {exc}; still listening]",
                  flush=True)
            self.talk = None
            return False
        self.iris.arm_push_to_talk(True)
        print(f"  [push to talk: hold {chord} to speak]", flush=True)
        return True

    def _start_tray(self) -> bool:
        try:
            import pystray                                    # noqa: F401
            from PIL import Image                             # noqa: F401
        except ImportError:
            print("  [tray icon: pip install pystray pillow]", flush=True)
            return False
        import pystray

        self.icon = pystray.Icon(
            "iris",
            _icon_image(self.iris.mic_off.is_set(), self.iris.waiting_to_talk),
            self._title(),
            menu=self._menu(),
        )
        # run() owns whatever thread it is on until stop(), so it gets its own.
        self._thread = threading.Thread(target=self._run_icon, daemon=True)
        self._thread.start()
        return True

    def _run_icon(self) -> None:
        try:
            self.icon.run()
        except Exception as exc:
            print(f"  [tray icon stopped: {exc}]", flush=True)

    # -- hotkeys -----------------------------------------------------------
    def _start_hotkeys(self) -> bool:
        try:
            from pynput import keyboard
        except ImportError:
            print("  [global hotkeys: pip install pynput]", flush=True)
            return False

        binding = {
            self.cfg.hotkey_mute: self.toggle_mic,
            self.cfg.hotkey_idle: self.toggle_idle,
            self.cfg.hotkey_stop: self.stop_speaking,
            self.cfg.hotkey_click_through: self.toggle_click_through,
        }
        binding = {k: v for k, v in binding.items() if k}
        if not binding:
            return False
        try:
            self.hotkeys = keyboard.GlobalHotKeys(binding)
            self.hotkeys.start()
        except Exception as exc:
            print(f"  [global hotkeys unavailable: {exc}]", flush=True)
            self.hotkeys = None
            return False
        print("  [hotkeys: " + ", ".join(
            f"{k} {v.__name__.replace('_', ' ')}" for k, v in binding.items()
        ) + "]", flush=True)
        return True

    def stop(self) -> None:
        if self.talk is not None:
            self.talk.stop()
            self.talk = None
            self.iris.arm_push_to_talk(False)
        if self.hotkeys is not None:
            try:
                self.hotkeys.stop()
            except Exception:
                pass
            self.hotkeys = None
        if self.icon is not None:
            try:
                self.icon.stop()
            except Exception:
                pass
            self.icon = None


def attach(iris, cfg) -> Controls | None:
    """Give `iris` a tray icon and hotkeys, if the packages are installed."""
    controls = Controls(iris, cfg)
    if not controls.start():
        return None
    iris.tray = controls
    return controls
