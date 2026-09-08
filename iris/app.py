"""The conversation loop that ties mic, brain, voice and speaker together."""

import queue
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
import time

from . import commands, timers
from .config import Config
from .llm import Brain, clean_for_speech
from .memory import CORRECTED, Memory
from .mic import MicListener
from .player import Player
from .stt import Transcriber
from .tts import Voice
from .vtube import VTubeStudio


class Iris:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cancel = threading.Event()

        print("loading voice...", flush=True)
        self.voice = Voice(cfg)
        print(f"loading whisper ({cfg.whisper_model})...", flush=True)
        self.stt = Transcriber(cfg)
        self.stt.warm()
        self.memory = Memory(cfg) if cfg.memory else None
        if self.memory:
            st = self.memory.stats()
            print(f"memory: {st['facts']} facts, {st['turns']} turns on record",
                  flush=True)
        self.screen = self._open_screen()
        self.timers = timers.Timers(cfg) if cfg.timers else None
        self.brain = Brain(cfg, self.memory, self.screen)
        self._consolidating = threading.Event()
        print(f"waking {cfg.ollama_model}...", flush=True)
        self.brain.warm()
        self.player = Player(cfg)
        self.listener = MicListener(
            cfg,
            on_speech_start=self._on_speech_start,
            on_speculate=self._speculate,
        )
        # One worker: transcriptions must not compete with each other for cores.
        self._stt_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")
        self._spec: tuple[int, Future] | None = None
        self._spec_lock = threading.Lock()
        self._hotwords = ""
        self.refresh_hotwords()
        self.vts = None
        if cfg.vtube:
            vts = VTubeStudio(cfg)
            if vts.connect():
                self.vts = vts

        self._last_heard = time.time()
        self._idle_wait = cfg.idle_after_s
        self._idle_streak = 0

        # Everything that wants the microphone shut, kept apart so they can
        # survive each other -- a reply must not reopen a microphone the user
        # muted from the tray, and neither may touch it while a push-to-talk
        # key is up. They are folded together in _update_mic(), which is the
        # only thing allowed to touch listener.muted.
        self.mic_off = threading.Event()      # deliberate, from the tray
        self._speaking = threading.Event()    # half-duplex, for one reply
        self._ptt_armed = threading.Event()   # push-to-talk actually running
        self._talk_held = threading.Event()   # ...and its key is down now
        self.idle_paused = threading.Event()
        self.quitting = threading.Event()
        self.tray = None
        self._overlay_stop = None

    # -- barge-in ----------------------------------------------------------
    def _on_speech_start(self) -> None:
        # The silence is broken the moment they open their mouth, not when the
        # transcript finally lands -- otherwise a long sentence can run past the
        # idle deadline and she starts talking over the end of it.
        #
        # Only the deadline is pushed back, not the streak or the backoff: the
        # VAD opens on a cough or a door too, and a full reset_idle() would let
        # background noise wind her back up to chattering every 45 s in a room
        # nobody is talking in.
        self._last_heard = time.time()
        if self.cfg.barge_in and self.player.busy:
            self.cancel.set()
            self.player.stop()
            print("  [interrupted]", flush=True)

    # -- the microphone gate -----------------------------------------------
    def _update_mic(self) -> None:
        """Open or shut the microphone from all the reasons it might be shut.

        Three of them, and they overlap: the tray mute is a deliberate switch
        that outlives any reply, half-duplex closes it for the length of one,
        and push-to-talk closes it whenever the key is not held. Setting the
        event directly from each is what let one of them undo another.
        """
        shut = (
            self.mic_off.is_set()
            or (self._speaking.is_set() and not self.cfg.barge_in)
            # Only once it is really running. A push-to-talk that failed to
            # start must leave her listening normally, not permanently deaf.
            or (self._ptt_armed.is_set() and not self._talk_held.is_set())
        )
        if shut:
            self.listener.muted.set()
        else:
            self.listener.muted.clear()

    def set_talk_held(self, held: bool) -> None:
        """The push-to-talk key went down, or came up."""
        if held:
            self._talk_held.set()
            self._update_mic()
            return
        # Letting go ends the turn. The listener is told to emit what it has
        # before the gate shuts, or the mute path would bin it.
        self.listener.finish()
        self._talk_held.clear()
        self._update_mic()

    def arm_push_to_talk(self, on: bool) -> None:
        """Called once the key listener is confirmed running, or has failed."""
        if on:
            self._ptt_armed.set()
        else:
            self._ptt_armed.clear()
        self._update_mic()

    def refresh_hotwords(self) -> None:
        """Bias the transcriber toward the names she has learned.

        Proper nouns are what a generic English model gets wrong, and the fact
        table already holds the ones that matter. Refreshed after every
        consolidation, so the longer she knows someone the better she hears them.
        """
        if self.memory is None or not self.cfg.hotwords:
            self._hotwords = "Iris"
            return
        words = ["Iris"] + [w for w in self.memory.keywords() if w != "Iris"]
        self._hotwords = " ".join(words)

    # -- speculative transcription ----------------------------------------
    def _speculate(self, seq: int, audio) -> None:
        """Called by the mic the moment a pause looks like the end of a turn.

        Whisper's encoder runs over a padded 30 s window whatever you said, so
        a two-word reply costs about as much as a long one. Starting that fixed
        cost during the silence we were going to wait out anyway hides nearly
        all of it.
        """
        future = self._stt_pool.submit(self.stt, audio, self._hotwords)
        with self._spec_lock:
            self._spec = (seq, future)

    def _transcribe(self, utt) -> tuple[str, bool]:
        with self._spec_lock:
            spec = self._spec
            self._spec = None
        timeout = self.cfg.stt_timeout_s
        try:
            if utt.speculated and spec is not None and spec[0] == utt.seq:
                return spec[1].result(timeout=timeout), True
            # Cold path goes through the same single worker: a stale speculation
            # may still be running, and one WhisperModel must not be used
            # concurrently.
            return self._stt_pool.submit(
                self.stt, utt.audio, self._hotwords
            ).result(timeout=timeout), False
        except TimeoutError:
            # There is one worker and no way to cancel what it is running, so a
            # wedged transcription cannot be recovered from here -- every turn
            # after this one will queue behind it and time out too. What the
            # timeout buys is that it says so, instead of hanging the loop in
            # silence with nothing to diagnose it from.
            print(f"  [transcription gave up after {timeout:.0f}s; "
                  f"restart her if this repeats]", flush=True)
            return "", False

    # -- one turn ----------------------------------------------------------
    def _say(self, phrases, label: str = "iris: ") -> None:
        """Synthesise and play a stream of phrases, interruptibly."""
        self.cancel.clear()
        self._speaking.set()
        self._update_mic()

        t0 = time.perf_counter()
        first_audio = None
        print(label, end="", flush=True)
        try:
            for phrase in phrases:
                spoken, tags = clean_for_speech(phrase)
                if not spoken:
                    continue
                audio = self.voice.say(spoken)
                if self.cancel.is_set():
                    break
                # The cue rides along with the audio instead of firing here.
                # Synthesis runs ahead of the speakers, so acting on it now
                # changed her face while an earlier phrase was still playing.
                cue = tags[0] if tags and self.vts is not None else None
                self.player.play(audio, cue=cue)
                if first_audio is None:
                    first_audio = time.perf_counter() - t0
                print(spoken + " ", end="", flush=True)
                if cue is not None:
                    print(f"[{self._cue_mark(cue)}] ", end="", flush=True)

            # Let her finish before the mic reopens, unless she was cut off.
            while self.player.busy and not self.cancel.is_set():
                time.sleep(0.02)
        finally:
            print(f"\n  [first audio {first_audio:.2f}s]" if first_audio else "")
            self._speaking.clear()
            if not self.cfg.barge_in:
                time.sleep(0.15)          # let the speakers settle
            self._update_mic()

    def respond(self, text: str) -> None:
        self._say(self.brain.reply(text, self.cancel))

    # -- timers ------------------------------------------------------------
    def handle_timer(self, text: str) -> bool:
        """Set, cancel or report a timer instead of replying.

        Runs before the memory commands, and has to: "forget the timer" is a
        cancellation, and the memory grammar reads it as an instruction and
        starts deleting facts about timers.
        """
        if self.timers is None:
            return False
        parsed = timers.parse(text)
        if parsed is None:
            return False
        action, span, label = parsed

        if action == timers.SET:
            if span > self.cfg.timer_max_s:
                reply = "That is further off than I will be running."
            else:
                self.timers.add(span, label)
                reply = self.timers.acknowledge(span, label)
                print(f"  [timer set: {timers.spoken(span)}"
                      + (f" -- {label}" if label else "") + "]", flush=True)
        elif action == timers.CANCEL:
            gone = self.timers.cancel_all()
            reply = "Gone." if gone else "You have not set one."
            if gone:
                print(f"  [timers cancelled: {gone}]", flush=True)
        else:
            reply = self.timers.remaining_line()

        # The acknowledgement is a fixed line, not a generated one, for the
        # reason the memory ones are: it has to be instant, and spending a
        # generation to say "fine, ten minutes" would put a second model run in
        # the middle of a turn.
        self.brain.history.append({"role": "user", "content": text})
        self.brain.history.append({"role": "assistant", "content": reply})
        self._say(iter([reply]))
        return True

    def _announce_timers(self) -> bool:
        """Say anything whose time is up. True if she did.

        Deliberately not gated on `idle_paused` or `mic_off`: those switch off
        her chatter and her ears, and neither is a reason to swallow something
        she was explicitly asked to say. It does wait for her own sentence and
        for theirs to finish, which costs a second and avoids talking over
        either.
        """
        if self.timers is None:
            return False
        if self.player.busy or self.listener.capturing.is_set():
            return False
        fired = self.timers.due()
        if not fired:
            return False
        line = self.timers.announce(fired)
        self.brain.history.append({"role": "assistant", "content": line})
        if self.memory is not None:
            self.memory.log("assistant", line)
        self._say(iter([line]), label="iris (timer): ")
        self._last_heard = time.time()
        return True

    # -- spoken memory commands -------------------------------------------
    def handle_command(self, text: str) -> bool:
        """Act on "remember that ..." / "forget ..." instead of replying.

        Returns True if it was a command, so the caller knows not to hand it to
        the brain. The acknowledgement is a fixed line rather than a generated
        one: it has to be instant and unambiguous, and spending a generation to
        say "got it" would put a second model run in the middle of a turn.
        """
        if self.memory is None:
            return False
        parsed = commands.parse(text)
        if parsed is None:
            return False
        kind, what = parsed

        if kind == commands.REMEMBER:
            fact = commands.third_person(what)
            status = self.memory.remember(fact, quote=text)
            verb = "corrected" if status == CORRECTED else "remembered"
            print(f"  [{verb}] {fact}", flush=True)
            self.refresh_hotwords()
            reply = ("Got it, I've changed that." if status == CORRECTED
                     else "Got it. I'll remember that.")
        else:
            gone = ([self.memory.forget_last()] if not what
                    else self.memory.forget_like(what))
            gone = [g for g in gone if g]
            for fact in gone:
                print(f"  [forgot] {fact}", flush=True)
            self.refresh_hotwords()
            reply = ("Forgotten." if gone
                     else "I don't have anything like that.")

        # Kept out of the turn log on purpose: the fact is already stored, and
        # letting the extractor read the instruction back would have it deriving
        # facts about the act of remembering.
        self.brain.history.append({"role": "user", "content": text})
        self.brain.history.append({"role": "assistant", "content": reply})
        self._say(iter([reply]))
        return True

    # -- desktop controls --------------------------------------------------
    def set_mic_off(self, off: bool) -> None:
        """Mute the microphone until told otherwise."""
        if off:
            self.mic_off.set()
        else:
            self.mic_off.clear()
        self._update_mic()
        print(f"  [mic {'muted' if off else 'live'}]", flush=True)

    def set_idle_paused(self, paused: bool) -> None:
        if paused:
            self.idle_paused.set()
        else:
            self.idle_paused.clear()
            self.reset_idle()
        print(f"  [idle chatter {'paused' if paused else 'on'}]", flush=True)

    def stop_speaking(self) -> None:
        self.cancel.set()
        self.player.stop()
        print("  [stopped]", flush=True)

    def quit(self) -> None:
        self.quitting.set()

    # -- speaking first ----------------------------------------------------
    def reset_idle(self) -> None:
        """Called whenever they say something: the silence has been broken."""
        self._last_heard = time.time()
        self._idle_wait = self.cfg.idle_after_s
        self._idle_streak = 0

    @property
    def idle_due(self) -> bool:
        return (
            self.cfg.idle
            and not self.idle_paused.is_set()
            and not self.mic_off.is_set()
            and self._idle_streak < self.cfg.idle_max_streak
            and not self.player.busy
            # Mid-utterance is not silence. Without this she can start an
            # unprompted remark while they are still mid-sentence, and in the
            # default half-duplex mode _say() then mutes the mic, which throws
            # away the rest of what they were saying.
            and not self.listener.capturing.is_set()
            and not self._consolidating.is_set()
            and time.time() - self._last_heard >= self._idle_wait
        )

    def speak_first(self) -> None:
        """Say something into the silence.

        Each unanswered remark backs off, and after `idle_max_streak` she stops
        entirely until spoken to -- an assistant that talks to an empty room on
        a fixed timer stops being company and becomes a noise source.
        """
        # idle_due was checked a few instructions ago and the gate can have
        # opened since. Look once more, as late as possible: after this line the
        # very next thing _say() does is mute the mic, so this is the last point
        # at which backing off still costs nothing. A remark not made is not a
        # remark ignored, so the streak is left alone.
        if self.listener.capturing.is_set():
            return
        self._idle_streak += 1
        self._say(self.brain.unprompted(self.cancel), label="iris (unprompted): ")
        self._last_heard = time.time()
        self._idle_wait *= self.cfg.idle_backoff

    def _cue_mark(self, tag: str) -> str:
        """How a cue is printed in the transcript.

        Printed where it sat in her speech, because it is stripped before
        synthesis and there would otherwise be no sign of it at all. A trailing
        "?" means the loaded model has no expression for that cue. The transcript
        is written as the text is generated; the avatar itself follows the audio,
        so the two are deliberately not in step.
        """
        return tag if tag in self.vts.expressions else f"{tag}?"

    # -- memory ------------------------------------------------------------
    def maybe_consolidate(self) -> None:
        """Distil recent turns into facts, off the conversation thread.

        Only ever between turns: it is a second generation on the same GPU, and
        running it during a reply would stall her mid-sentence.
        """
        if not self.brain.consolidation_due or self._consolidating.is_set():
            return
        self._consolidating.set()
        threading.Thread(target=self._consolidate, daemon=True).start()

    def _consolidate(self) -> None:
        try:
            for status, fact in self.brain.consolidate():
                verb = "corrected" if status == CORRECTED else "remembered"
                print(f"  [{verb}] {fact}", flush=True)
            self.refresh_hotwords()
        except Exception as exc:
            print(f"  [memory: consolidation failed: {exc}]", flush=True)
        finally:
            self._consolidating.clear()

    def shutdown_memory(self) -> None:
        """Catch the tail of the session before the process goes away."""
        if self.memory is None:
            return
        if self._consolidating.is_set():
            print("finishing up...", flush=True)
            while self._consolidating.is_set():
                time.sleep(0.05)
        try:
            if self.memory.stats()["pending"]:
                print("remembering...", flush=True)
                for status, fact in self.brain.consolidate():
                    verb = "corrected" if status == CORRECTED else "remembered"
                    print(f"  [{verb}] {fact}", flush=True)
        except Exception as exc:
            print(f"  [memory: consolidation failed: {exc}]", flush=True)
        self.memory.close()

    # -- modes -------------------------------------------------------------
    def run_voice(self) -> None:
        self.player.start()
        self._start_avatar()
        self._start_overlay_keeper()
        self.listener.start()
        self.reset_idle()
        print("\nlistening. ctrl+c to quit.\n", flush=True)
        try:
            while not self.quitting.is_set():
                try:
                    self._turn()
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    # The avatar is already forbidden from breaking a
                    # conversation; the brain was not, and it is the far more
                    # likely thing to go. Ollama being restarted, evicting the
                    # model or dropping a stream took the whole process down
                    # with a traceback -- and with it Whisper and Kokoro, which
                    # cost eight seconds to load again. One failed turn is a
                    # failed turn. She stays up and answers the next one, the
                    # way she does when VTube Studio goes away.
                    print(f"  [turn failed: {type(exc).__name__}: {exc}]",
                          flush=True)
                    self._recover()
        except KeyboardInterrupt:
            print("\nbye.")
        finally:
            self._shutdown()

    def _turn(self) -> None:
        """Wait for something to be said, and answer it. One iteration."""
        if self._announce_timers():
            return
        try:
            # A bare get() blocks Ctrl+C on Windows; poll instead.
            utt = self.listener.utterances.get(timeout=0.3)
        except queue.Empty:
            if self.idle_due:
                self.speak_first()
            return
        if self.player.busy and not self.cfg.barge_in:
            return
        self.reset_idle()
        t = time.perf_counter()
        text, hit = self._transcribe(utt)
        if not text:
            return
        tag = "prefetched" if hit else "cold"
        print(f"you: {text}   [stt {time.perf_counter() - t:.2f}s {tag}]",
              flush=True)
        if not self.handle_timer(text) and not self.handle_command(text):
            self.respond(text)
        # The silence starts when she stops talking, not when they did.
        self._last_heard = time.time()
        self.maybe_consolidate()

    def _recover(self) -> None:
        """Put the pieces back after a turn died partway through it.

        A failure between muting the microphone and unmuting it would otherwise
        leave her deaf for the rest of the session, which looks exactly like a
        crash and is harder to diagnose than one.
        """
        self.cancel.clear()
        self.player.stop()
        self._speaking.clear()
        self._update_mic()
        # A remark that failed is not a remark that was ignored, so the backoff
        # is not advanced -- but the deadline is, or she retries instantly and
        # turns a broken ollama into a spin loop.
        self._last_heard = time.time()

    def _start_avatar(self) -> None:
        if self.vts is None:
            return
        self.vts.start_lipsync(self.player)
        self.vts.start_cues(self.player)

    def _open_screen(self):
        """Start watching what has focus, if this machine can be asked.

        Same rule as the overlay: not Windows, or user32 missing, and she
        simply does not have this sense. It is never worth an error.
        """
        if not self.cfg.screen:
            return None
        try:
            from .screen import Watcher
        except Exception:
            return None
        watcher = Watcher(self.cfg)
        watcher.start()
        return watcher

    def _start_overlay_keeper(self) -> None:
        """Hold the avatar above other windows for as long as she is running."""
        if not self.cfg.overlay_keep:
            return
        try:
            from .overlay import STATE, start_keeper
        except Exception:
            return                    # not Windows, or user32 unavailable
        if not STATE.exists():
            return                    # overlay not applied; nothing to hold up
        self._overlay_stop = start_keeper(self.cfg.overlay_keep_s)
        print("  [holding the avatar on top]", flush=True)

    def _shutdown(self) -> None:
        if self._overlay_stop is not None:
            self._overlay_stop.set()
        if self.screen is not None:
            self.screen.stop()
        if self.tray is not None:
            self.tray.stop()
        self.listener.stop()
        self.player.close()
        self._stt_pool.shutdown(wait=False)
        if self.vts is not None:
            self.vts.close()
        self.shutdown_memory()

    def _read_lines(self, lines: queue.Queue) -> None:
        """Feed stdin into a queue. Its own thread, so the loop can poll.

        `input()` blocks until Return is pressed, which meant "Quit Iris" from
        the tray did nothing visible until the user typed something -- the one
        control most likely to be reached for when the terminal is buried is
        the one that appeared to be broken.
        """
        try:
            for line in sys.stdin:
                lines.put(line)
        except Exception:
            pass
        lines.put(None)            # stdin closed

    def run_text(self) -> None:
        """Same brain and voice, keyboard instead of a mic. Useful for tuning."""
        self.player.start()
        self._start_avatar()
        print("\ntext mode. blank line to quit.\n", flush=True)
        lines: queue.Queue = queue.Queue()
        threading.Thread(target=self._read_lines, args=(lines,),
                         daemon=True).start()
        try:
            while not self.quitting.is_set():
                print("you: ", end="", flush=True)
                line = None
                while not self.quitting.is_set():
                    try:
                        line = lines.get(timeout=0.3)
                        break
                    except queue.Empty:
                        # A timer set from here has to be able to go off here.
                        if self._announce_timers():
                            print("you: ", end="", flush=True)
                if line is None:                  # quit, or stdin closed
                    break
                text = line.strip()
                if not text:
                    break
                try:
                    if not self.handle_timer(text) and not self.handle_command(text):
                        self.respond(text)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    # Same rule as the voice loop: one failed turn is a failed
                    # turn, not the end of the session.
                    print(f"  [turn failed: {type(exc).__name__}: {exc}]",
                          flush=True)
                    self._recover()
                # The silence starts when she stops talking, not when they did.
                self._last_heard = time.time()
                self.maybe_consolidate()
        except (KeyboardInterrupt, EOFError):
            pass
        finally:
            print(flush=True)      # close off the dangling prompt
            if self.screen is not None:
                self.screen.stop()
            if self.tray is not None:
                self.tray.stop()
            self.player.close()
            self._stt_pool.shutdown(wait=False)
            if self.vts is not None:
                self.vts.close()
            self.shutdown_memory()
