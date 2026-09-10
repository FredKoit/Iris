"""Central tuning knobs for Iris. Edit here, not in the pipeline modules."""

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

# Audio ---------------------------------------------------------------------
MIC_RATE = 16000          # what Silero VAD and Whisper both expect
FRAME = 512               # 32 ms at 16 kHz -- Silero's native frame size
TTS_RATE = 24000          # Kokoro's output rate


@dataclass
class Config:
    # --- speech detection ---
    vad_threshold: float = 0.55        # speech probability to open the gate
    vad_release: float = 0.35          # drop below this to start the silence timer
    silence_ms: int = 500              # trailing silence that ends an utterance
    speculate_ms: int = 160            # start transcribing this early into it
    min_speech_ms: int = 250           # ignore blips shorter than this
    max_utterance_s: float = 20.0      # hard cap so a noisy room can't hang the loop
    preroll_ms: int = 300              # audio kept before the gate opened

    # --- transcription ---
    whisper_model: str = "base.en"     # tiny.en / base.en / small.en
    whisper_threads: int = 6           # physical cores, not threads
    whisper_lang: str = "en"
    # Ceiling on one transcription. Whisper pads to a 30 s window whatever it
    # is given, so the cost barely varies -- measured at 1.0 s for 4 s of audio
    # with base.en -- and anything near this is a hang, not a slow model. It
    # exists because the pool has a single worker and no way to cancel a task:
    # without a timeout a wedged call blocks the conversation loop in silence,
    # forever, with no output to diagnose it from.
    stt_timeout_s: float = 30.0

    # --- brain ---
    ollama_model: str = "qwen3:4b-instruct-2507-q4_K_M"
    ollama_host: str = "http://127.0.0.1:11434"
    # Nothing here stays 100% on this GPU -- qwen3 spills 27% to the CPU at
    # 2048 on its own. What 2048 buys is the least of it: measured with
    # `ollama ps`, 2048 is 3.2 GB at 27% CPU and 23.2 tok/s, and 3072 is 3.4 GB
    # at 31% and 18.8 tok/s. That is a fifth of her generation speed for a
    # kilobyte of context, on a card she is often sharing with a game, so the
    # room for memory comes out of history instead -- see max_history_turns.
    num_ctx: int = 2048
    temperature: float = 0.9           # personality wants some looseness
    top_p: float = 0.95
    # Short on purpose. Long-term continuity now lives in memory as distilled
    # facts, so verbatim history only has to cover the current exchange.
    max_history_turns: int = 6
    # That bounds how many turns are kept, not how big they are, and size is
    # what overruns the window: six ordinary turns are about 90 tokens, six
    # twenty-five-second rambles are 319. Measured with a full memory block,
    # the prompt reached 2077 of 2048 before a single token was generated --
    # and an overflow discards the *front* of the prompt, which is the persona,
    # not the stale turn at the back.
    #
    # So the history is not given a budget of its own. It gets whatever is left
    # of num_ctx once the system prompt is built and a reply is reserved, which
    # is the one form that cannot go stale the next time the persona changes.
    # See Brain._recent().
    reply_tokens: int = 160            # reserved for her answer, and its cap
    keep_alive: str = "60m"            # stop ollama evicting her between turns
    # Reasoning models emit a thinking block before answering. For a voice she
    # cannot afford it: qwen3:4b spent its whole budget thinking and said
    # nothing at all. None = do not send the flag (models without the feature
    # reject it), False = explicitly off.
    think: bool | None = None

    # --- voice ---
    tts_voice: str = "af_heart"
    # 1.0 is the natural rate. The old 1.1 was chosen to shave latency and it
    # reads as hurried, which is most of what makes a synthetic voice sound
    # synthetic. Costs about 0.08 s on the opening phrase.
    tts_speed: float = 1.0
    # Silence appended to each phrase. She is synthesised phrase by phrase, and
    # joined clips run 0.32 s shorter than the same text spoken in one go --
    # that missing pause is breath. Tune by ear with scripts/voices.py --tune.
    tts_pause_ms: int = 60
    tts_precision: str = "fp32"        # fp32 beats int8 on AMD Zen 3 (no VNNI)
    tts_threads: int = 4
    tts_voices: Path = field(default_factory=lambda: MODELS / "voices-v1.0.bin")

    # --- cloned voice ---
    # Kokoro supplies the words and the prosody; the reference recording
    # supplies the timbre. Costs ~0.5 s on the opening phrase and ~1.4 GB of
    # RAM. With this on, tts_voice matters much less -- it still shapes accent
    # and delivery, but the voice you hear is the reference.
    clone: bool = False
    clone_reference: Path = field(
        default_factory=lambda: ROOT / "samples" / "iris_reference.wav"
    )
    clone_model: str = "frothywater/kanade-12.5hz"
    clone_threads: int = 6

    @property
    def tts_model(self) -> Path:
        name = {"fp32": "kokoro-v1.0.onnx", "int8": "kokoro-v1.0.int8.onnx"}
        return MODELS / name[self.tts_precision]

    # --- avatar (VTube Studio) ---
    vtube: bool = True                 # silently off if VTS is not running
    vtube_host: str = "127.0.0.1"
    vtube_port: int = 8001
    vtube_timeout: float = 2.0
    vtube_auth_timeout: float = 60.0   # a human has to click Allow in VTS
    vtube_reconnect_s: float = 5.0     # retry interval after VTS goes away
    vtube_token: Path = field(default_factory=lambda: ROOT / ".vts_token")
    # Emotion cues she can emit, matched against the model's hotkey names.
    vtube_emotions: tuple = ("neutral", "happy", "smug", "sad", "angry",
                             "surprised", "thinking")
    # Explicit emotion -> expression name, for models whose expressions are not
    # named after English emotions. Checked before the substring match. Names
    # are matched case-insensitively, whole or partial.
    # Verify each one with: python scripts/avatar.py --try <expression name>
    #
    # For 'yiyi_chair'. A cue left out of here clears her face instead of
    # showing something that reads wrong, which is why sad, angry and surprised
    # are absent: nothing this model exposes says any of them plainly. The
    # remaining candidates are 'ase' (sweat) and 'cheek' (blush) -- both read
    # as flustered rather than as any one emotion, so they are left for you to
    # assign once you have watched them.
    #
    # 'cat_pos' is currently unreachable, and it is the model's fault rather
    # than the mapping's: qwen3:4b never emits [smug]. Not once in 160 sampled
    # replies, including a run on turns written to invite it -- a promise they
    # broke, a question they could answer themselves -- and including a persona
    # that called [smug] her ordinary cue. It labels those lines [thinking].
    # Left mapped because it costs nothing and a better model would use it.
    vtube_map: dict = field(default_factory=lambda: {
        "happy": "heart",       # heart eyes
        "smug": "cat_pos",      # cat mouth -- see above, never fires here
        "thinking": "megane",   # glasses -- carries most of her lines
    })
    vtube_lipsync_fps: int = 30
    vtube_mouth_gain: float = 6.0      # speech RMS is small; scale it to 0..1
    vtube_mouth_attack: float = 0.6    # open fast
    vtube_mouth_release: float = 0.25  # close slower, or the mouth flutters

    # --- idle motion ---
    # Nothing is tracking a face here, and VTube Studio's own idle animation is
    # off for this model, so without these she only moves her mouth.
    #
    # Amplitudes are in VTS input units, and the rig amplifies them by about
    # 1.2x on the way through: measured, FaceAngleX 25 lands at ParamAngleX
    # 29.8, which is the rig's limit. Small numbers here go a long way, and the
    # failure mode of large ones is a head that sweeps like a metronome.
    vtube_motion: bool = True
    vtube_motion_head: float = 5.0     # degrees of head wander
    vtube_motion_body: float = 2.0     # body lean, via FacePosition (see below)
    vtube_motion_speech: float = 0.8   # extra movement at full volume

    # Breathing. ParamBreath exists on the rig and nothing injectable reaches
    # it, so this rides on head pitch instead, where it does land -- the head
    # really does rise and fall a little with the chest. The old stand-in was a
    # body sway at 3 to 7 cycles a minute, which is not a rate anything
    # breathes at and, on this model, moved nothing anyway.
    vtube_breath_per_min: float = 15.0  # human resting is 12-20
    vtube_breath: float = 0.5          # degrees of head rise and fall

    # A beat is the small nod that lands on a stressed syllable. Without it
    # speech only scaled the idle wander up and down, so her head while talking
    # was the same shape as her head while silent -- correlation 0.96 between
    # the two, measured. This is the part that reads as saying something.
    vtube_beats: float = 1.0           # 0 turns them off
    vtube_beat_chance: float = 0.38    # of syllables that get one

    # Head tilt partly follows head turn. Every axis was an independent sine,
    # so turn, tilt and lean correlated at r = 0.00 with each other -- three
    # animations on one body rather than one body moving.
    vtube_head_tilt_follow: float = 0.25

    # How far she quiets down between postural shifts. People hold a position
    # and then change it; sines never rest, and she was below half a degree per
    # second only 18% of the time. 1.0 disables the settling.
    vtube_settle: float = 0.45

    # Brows, which nothing drove at all. They lift with an upward glance and
    # with the beats above. On this rig `Brows` reaches ParamBrowLForm and
    # ParamBrowRForm; BrowLeftY and BrowRightY reach nothing -- check yours
    # with `python scripts/avatar.py --probe`.
    #
    # Where neutral is. VTS declares this input as 0..1 with a default of 0,
    # but that is the bottom of the range, not the resting face -- this model
    # maps the whole input across the whole output, so the rest position is in
    # the middle. Swept live:
    #
    #   Brows 0.00 -> ParamBrowLForm -1.00      sending 0 pins her brows at
    #   Brows 0.25 -> ParamBrowLForm -0.50      one extreme and holds them
    #   Brows 0.50 -> ParamBrowLForm  0.00  <-  there, which is worse than
    #   Brows 0.75 -> ParamBrowLForm +0.50      never having driven them
    #   Brows 1.00 -> ParamBrowLForm +1.00
    #
    # Like the sign flags, this is a property of the model's mapping and not
    # something VTS will tell you. Set it wrong and her brows are stuck.
    vtube_brows_neutral: float = 0.5
    # How far above neutral they lift. Neutral + this must stay inside 0..1 or
    # the top of the movement is silently clamped.
    vtube_brows: float = 0.35
    # --- gaze ---
    # Eyes fixate and jump; they do not drift. See the long note in motion.py
    # for why, and for what the three couplings below are doing. Watch any of
    # this with `python scripts/avatar.py --eyes`.
    vtube_gaze: float = 0.3            # furthest the eyes look, of -1..1
    # Where "straight ahead" is on the vertical axis. Horizontal is centred on
    # zero and needs nothing; vertical is not, on this rig. Swept live:
    #
    #   EyeLeftY  0.00 -> ParamEyeBallY -1.04   <- pinned at the bottom stop
    #   EyeLeftY  0.25 -> ParamEyeBallY -0.54
    #   EyeLeftY  0.50 -> ParamEyeBallY -0.04   <- actually straight ahead
    #   EyeLeftY  1.00 -> ParamEyeBallY +0.96
    #
    # Sending values around zero, which is what a gaze model naturally produces,
    # held her eyes fully down and clamped there -- the whole vertical range
    # crushed against the stop, invisible, for as long as the gaze code has
    # existed. VTS declares this input as -1..1 with a default of 0 and the
    # model maps it otherwise, so only the rig can tell you. Zero is right for a
    # rig that centres it; check with `python scripts/avatar.py --probe`.
    vtube_gaze_y_neutral: float = 0.5
    vtube_gaze_hold: tuple = (0.35, 2.2)    # seconds held between saccades
    # How often the next look goes back to straight ahead -- at you, in effect.
    # Lower and she never quite meets your eye; higher and she stares.
    vtube_gaze_home: float = 0.45
    # Eyes hold their point while the head moves under them, as a fraction of
    # the gaze range at full head deflection. Also what settles her eyes back
    # towards centre once the head has caught up with a glance.
    vtube_gaze_vor: float = 0.35
    # How far the head follows a large glance, in head-wander units. 0 leaves
    # the eyes to do it alone, which reads as shifty.
    vtube_gaze_head: float = 1.2
    # Whether the head parameter points the opposite way to the eye one, per
    # axis. This is a property of the rig, and on 'yiyi_chair' the two axes do
    # not agree with each other -- measured against the live model, not guessed:
    #
    #   FaceAngleX +24  ->  ParamAngleX   +28.8      head and eye disagree
    #   EyeLeftX  +0.8  ->  ParamEyeBallX  -0.99     horizontally
    #
    #   FaceAngleY +24  ->  ParamAngleY   +40.9      and agree
    #   EyeLeftY  +0.8  ->  ParamEyeBallY  +1.59     vertically
    #
    # So the horizontal coupling has to be inverted and the vertical must not
    # be, which one flag cannot say. Both defaulted to "no flip", and the
    # horizontal one was wrong: the head turned *away* from every glance and
    # the eyes swung wider instead of settling. Driving the real motion and
    # reading the model's own parameters back, 0 of 7 glances had the head
    # follow before, 7 of 7 after.
    #
    # `python scripts/avatar.py --probe` measures this on any rig.
    vtube_gaze_flip_x: bool = True
    vtube_gaze_flip_y: bool = False
    # How far the lids close when she looks down, of fully shut. Wide open eyes
    # aimed at the floor is a specific thing that reads as a doll.
    #
    # Barely visible on this rig, and worth knowing before tuning it. The
    # EyeOpen response saturates almost immediately:
    #
    #   EyeOpenLeft 0.00 -> ParamEyeLOpen 0.01     blinks land correctly
    #   EyeOpenLeft 0.25 -> ParamEyeLOpen 0.83
    #   EyeOpenLeft 0.50 -> ParamEyeLOpen 0.99     anything above here reads
    #   EyeOpenLeft 1.00 -> ParamEyeLOpen 1.00     as simply open
    #
    # 0.25 leaves the lids at 0.75 at full downward gaze, which this curve
    # renders as wide open. Showing anything needs about 0.7, and that is a
    # squint -- left alone rather than tuned blind against one model's curve.
    vtube_lid_follow: float = 0.25

    vtube_blink: bool = True
    vtube_blink_every: tuple = (2.0, 6.0)   # seconds between blinks
    # Raised from 120: with the asymmetric profile in motion.py a blink is a
    # third closing and two thirds opening, and at 120 the opening half was two
    # frames at 30 fps. Human spontaneous blinks run 100-400 ms.
    vtube_blink_ms: int = 150
    # Blink rate roughly doubles while talking. It is one of the few
    # involuntary tells that tracks speech.
    vtube_blink_speech: float = 1.0

    # --- timers ---
    # "Remind me in ten minutes" is the thing people say out loud to a voice,
    # and it is one of the very few useful things she can do with no network.
    # They live in memory only: a countdown does not survive a restart, because
    # restoring one that spent most of its span in a process that was not
    # running is a worse answer than losing it.
    timers: bool = True
    # Past this she declines rather than pretending. She is a process somebody
    # closes, not a calendar, and a timer she will not be alive to fire is a
    # promise she cannot keep.
    timer_max_s: float = 12 * 3600

    # --- behaviour ---
    # Interrupting her mid-sentence needs headphones: on speakers her own voice
    # re-enters the mic and she interrupts herself on her first word. Off by
    # default because the built-in mic array and laptop speakers are the setup
    # most people start with; pass --barge-in once you have headphones on.
    barge_in: bool = False

    # --- situational awareness ---
    awareness: bool = True             # tell her the time, day and session length
    hotwords: bool = True              # bias the transcriber toward learned names

    # --- what is in front of her ---
    # The application that has focus, and how long it has held it. She floats
    # above it all day and without this cannot tell a game from a spreadsheet.
    # Windows only; silently off elsewhere. Nothing here leaves the machine.
    screen: bool = True
    screen_poll_s: float = 2.0         # two ctypes calls a tick; cheap
    # Her own windows. Time spent looking at her is not time spent on anything,
    # and clicking on her must not make her forget what they were doing.
    screen_ignore: tuple = ("VTube Studio",)
    # Shell surfaces that briefly take the foreground and are not something
    # anybody is doing: the tray flyout, the desktop itself, the alt-tab
    # overlay. Matched on the window *title*, because the process behind them
    # is explorer, and explorer is also File Explorer, which is a real
    # application somebody really is using.
    #
    # This is where "System tray overflow window. Explorer. Open about two
    # hours." came from -- she was asked what was on screen and told the truth
    # about a window that had stolen focus for a moment. Ignoring it keeps the
    # last real application in front instead, the same way her own window does.
    screen_ignore_titles: tuple = ("System tray overflow window", "Program Manager",
                                   "Task Switching", "Windows Shell Experience Host",
                                   "Snap Assist", "Search")
    # Applications she is told the name of but never the window title of. A
    # title is the most revealing single string on a machine -- documents,
    # browser tabs, message subjects -- and while none of it goes anywhere, a
    # vault is not something to read out loud even locally. Matched
    # case-insensitively on any part of the executable name.
    screen_private: tuple = ("1Password", "Bitwarden", "KeePass", "LastPass",
                             "Dashlane", "Proton Pass", "Enpass")
    screen_title_max: int = 90         # titles run long; she needs the gist
    # Below this she has only just switched to it, and saying how long someone
    # has been doing something for ninety seconds is not an observation.
    screen_settled_s: float = 300.0

    # --- speaking first ---
    idle: bool = True
    idle_after_s: float = 45.0         # silence before she says something herself
    idle_backoff: float = 1.7          # each unanswered remark waits longer
    idle_max_streak: int = 3           # then she gives up until spoken to

    # How sure Whisper has to be before something is allowed to become a
    # permanent fact, as opposed to merely being answered. The lower bar for
    # answering is in stt.py and stays where it is: no_speech_prob under 0.6
    # and avg_logprob over -1.0 on any one segment.
    #
    # Grounding cannot catch a misheard sentence, because it checks that a
    # fact's words appear in the transcript and they do -- the transcript is
    # simply wrong. "The user is in a setting with no after programs" came from
    # "it is not full of after programs that is to follow", which is a garbled
    # transcript that scored well enough to answer. Nothing downstream can tell
    # that from a real sentence; only Whisper knows, and it used to be the one
    # thing it was never asked.
    #
    # Watch the number in the `[stt ...]` line for a while before moving these:
    # clean close-mic speech sits near -0.2, and mumbling runs past -0.6.
    stt_clear_logprob: float = -0.5    # mean over the utterance's segments
    stt_clear_no_speech: float = 0.3   # worst segment, not the mean

    # --- memory ---
    memory: bool = True
    memory_db: Path = field(default_factory=lambda: ROOT / "memory.db")
    memory_max_facts: int = 40         # hard cap; weakest are pruned past this
    # How long it takes a fact to lose half its weight if it never comes up
    # again. Ranking on `hits` alone filled the prompt with whatever happened to
    # be echoed rather than whatever mattered, and it had a harder edge: once
    # forty facts had been reinforced even once, a newly learned one arrived
    # with a single hit, sorted last, and was pruned before it could ever be
    # said again. Nothing new could stick.
    #
    # Decay is measured from `updated`, so a fact that keeps coming up never
    # decays -- being mentioned again is what resets it. Something genuinely
    # dormant for a season does fall away, which is the deliberate cost.
    memory_half_life_days: float = 21.0
    memory_char_budget: int = 700      # ~175 tokens of the prompt, at most
    memory_dedup: float = 0.6          # word overlap above which two facts are one
    # Of a fact's content words, how many must appear in the sentence it is
    # credited to. Raised from 0.4 when matching started folding "likes" onto
    # "like": before that, half the true matches were missed by a plural, and
    # the bar had to be low to catch them. With the morphology handled, 0.4 let
    # "The user lives in Munich" through on the strength of "lives" alone --
    # inventing a city is the exact failure this check exists to stop, so a
    # majority of the fact's own words now have to be there.
    memory_grounding: float = 0.6
    # Subject overlap above which two facts are about the same thing. Lower than
    # memory_dedup on purpose: a correction shares the subject but is worded
    # differently ("likes coffee" / "does not like coffee"), so the wording test
    # never fires on it. Combined with disagreeing polarity, this is what makes
    # the newer statement replace the older instead of reinforcing it.
    memory_contradict: float = 0.5
    memory_forget: float = 0.4         # subject overlap for "forget about X"
    memory_min_turns: int = 3          # user turns needed before extracting
    consolidate_every: int = 6         # replies between fact-extraction passes

    # --- desktop controls ---
    # A tray icon and global hotkeys, for when she is floating over a game and
    # the terminal is three windows down. Both need packages that are not
    # required to hold a conversation (pystray + pillow, pynput); without them
    # this is silently off. pynput's chord syntax: <ctrl>+<alt>+m.
    tray: bool = True
    # VTube Studio re-asserts its own z-order, so "always on top" is dropped the
    # first time another window takes the foreground. Applied once it does not
    # stick; it has to be put back. Only ever acts when the overlay has actually
    # been applied (scripts/overlay.py leaves .overlay_state.json behind).
    overlay_keep: bool = True
    overlay_keep_s: float = 1.0
    # Hold a key to speak, instead of her answering everything she hears. Off
    # by default because it changes how she is used -- always listening is what
    # makes her a companion rather than a tool -- but with her floating over a
    # game it is the difference between a conversation with somebody else in
    # the room being private and it being addressed to her.
    #
    # A single key, not a chord: this one is held, not tapped. It is not
    # suppressed, so whatever else is bound to it still fires -- pick one your
    # game does not use.
    push_to_talk: bool = False
    hotkey_talk: str = "<f7>"
    hotkey_mute: str = "<ctrl>+<alt>+m"
    hotkey_idle: str = "<ctrl>+<alt>+i"
    hotkey_stop: str = "<ctrl>+<alt>+s"
    hotkey_click_through: str = "<ctrl>+<alt>+c"

    # --- doing things ---
    # The volume, the brightness, whatever is playing, and "open Spotify".
    # Recognised by a regex grammar rather than by the brain, for the reasons
    # in actions.py: a second generation mid-turn is the cost timers.py already
    # refuses, and a 2048-token context at temperature 0.9 is the wrong tool
    # for emitting a plan. Anything the grammar does not recognise is answered
    # normally, so this narrows what she says to and never what she says.
    #
    # Volume needs pycaw and comtypes; brightness needs a monitor that answers
    # WMI. Without either she says so in one line rather than pretending.
    actions: bool = True
    action_step: int = 10              # "turn it down", with no amount named
    # How long "it" keeps meaning the knob she last touched. Long enough for
    # "...actually, a bit more", short enough that "turn it up" an hour later
    # is a fresh sentence about something else entirely.
    action_context_s: float = 90.0
    # Brightness is a PowerShell subprocess and a WMI round trip -- the one
    # thing here that is not a library call. Bounded so a monitor that never
    # answers costs a turn instead of the session.
    action_timeout_s: float = 5.0

    persona_file: Path = field(default_factory=lambda: ROOT / "persona.txt")
    output_device: int | None = None   # None = system default
    input_device: int | None = None


def load_persona(cfg: Config) -> str:
    return cfg.persona_file.read_text(encoding="utf-8").strip()
