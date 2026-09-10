"""Idle motion, so she is never perfectly still.

There is no webcam here, so nothing drives her head. VTube Studio's own idle
animation is off for this model -- probed live, ParamAngleY sat frozen at 7.23
and ParamBreath at 0.00 -- which leaves a mannequin that only moves its mouth.
That reads as broken in a way that no amount of expression mapping fixes.

Head and body movement is synthesised from sums of sines at periods that share
no common multiple, so the cycle never visibly repeats. Random noise was the
obvious alternative and is wrong: sampled per frame it reads as a twitch, and
smoothed enough to stop twitching it stops looking deliberate. Slow overlapping
waves read as a person shifting their weight.

The eyes do not work that way and are modelled separately -- see the gaze
section below. Everything here is a function of (time, speech level) and of its
own private random state, so it can be inspected without VTube Studio running:
`python scripts/avatar.py --eyes --dry`.
"""

import math
import random

from .config import Config

# VTS input parameters, which it maps onto the model's own rig:
#   FaceAngle{X,Y,Z}   -> ParamAngle{X,Y,Z}      head turn, nod, tilt
#   EyeOpen{Left,Right}, Eye{Left,Right}{X,Y}    blink and gaze
#
# FacePosition{X,Y} was believed to reach ParamBodyAngle. Probed against the
# live model it reaches nothing at all: FacePositionX, Y and Z each move zero
# of the rig's 97 parameters. The body does move, but off the head -- FaceAngleX
# +24 carries ParamBodyAngleX to +7.9 on its own -- so a rig like this one gets
# its lean free from the head and BODY_X/BODY_Y below are inert.
#
# They are kept because the mapping is per-model and another rig may well honour
# them; `python scripts/avatar.py --probe` says which yours does. But on
# 'yiyi_chair', vtube_motion_body changes nothing you can see.
HEAD_YAW = "FaceAngleX"
HEAD_PITCH = "FaceAngleY"
HEAD_ROLL = "FaceAngleZ"
BODY_X = "FacePositionX"
BODY_Y = "FacePositionY"
EYES_OPEN = ("EyeOpenLeft", "EyeOpenRight")
# One channel, not two: `Brows` reaches ParamBrowLForm and ParamBrowRForm on
# this rig, while BrowLeftY and BrowRightY reach nothing at all.
BROWS = "Brows"
GAZE_X = ("EyeLeftX", "EyeRightX")
GAZE_Y = ("EyeLeftY", "EyeRightY")

# (period seconds, phase) pairs per axis. The periods are mutually prime-ish so
# the sum wanders instead of looping, and each axis is given its own set so the
# head does not sweep in a flat circle.
_WAVES = {
    HEAD_YAW:   ((11.0, 0.0), (7.3, 2.1)),
    HEAD_PITCH: ((9.4, 1.3), (5.9, 4.0)),
    HEAD_ROLL:  ((13.7, 0.7), (6.1, 3.4)),
    BODY_X:     ((17.0, 0.4), (10.3, 1.9)),
    BODY_Y:     ((19.0, 2.6), (8.7, 0.2)),
    GAZE_X[0]:  ((14.2, 1.1), (6.7, 5.0)),
    GAZE_Y[0]:  ((16.9, 3.3), (9.1, 1.7)),
}


def _drift(t: float, waves) -> float:
    """A slow wander in roughly -1..1."""
    return sum(math.sin(2 * math.pi * t / p + ph) for p, ph in waves) / len(waves)


def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return lo if v < lo else hi if v > hi else v


# -- gaze ---------------------------------------------------------------------
#
# Eyes do not drift. They fixate and they jump: a fifth of a second to a couple
# of seconds of near stillness, then a ballistic flick that is over in well
# under a tenth of a second. Smooth eye movement happens only when tracking
# something that is itself moving, which is not this. This module used to sweep
# both eyes along a sum of sines like everything else, and that is the specific
# thing that reads as sedated rather than as alive -- in a real face, eyes
# moving that slowly and that continuously mean something is wrong.
#
# So the model here is a state machine: pick a target, flick to it, hold it,
# repeat. Three couplings hang off it, and they matter as much as the saccades
# themselves, because eyes that move correctly but in isolation just look like a
# second animation playing over the first:
#
#   - the lids follow the eyes down
#   - the head goes after a large glance, late and only part of the way
#   - the eyes hold their point while the head moves under them
#
# The last two together produce the thing you actually recognise: the eyes lead,
# the head catches up, and the eyes slide back towards centre as it does.

# Duration of a saccade, for a jump of no distance and for one right across the
# range. Real saccades follow a "main sequence" -- larger means longer,
# sub-linearly. At 30 fps most of these resolve in one or two frames, which is
# exactly right: a saccade is meant to look like a jump.
_SACCADE_MIN = 0.03
_SACCADE_MAX = 0.10

# A jump past this fraction of the full range counts as looking *at* something
# rather than shifting slightly, and is what recruits the head and the blink.
# Measured against the amplitude draw below, 0.35 puts a head turn on about one
# saccade in eight -- roughly one every ten seconds. The obvious-looking 0.55
# fires once every three minutes, which is indistinguishable from never.
_BIG_SHIFT = 0.35
# How often a large gaze shift takes a blink with it. Real blinks cluster around
# gaze changes strongly enough that the two look wrong apart.
_SHIFT_BLINK = 0.4

# Vertical gaze travel as a fraction of horizontal: the smaller range, in an
# eye and in the rig. The lid coupling divides by it, so vtube_lid_follow means
# what it says at the bottom of her actual range rather than 70% of it.
_VERTICAL = 0.6

# Fixation is not stillness -- the eye wanders slightly around the point it is
# holding. As a fraction of the gaze range: small enough not to read as
# movement, large enough that a held gaze is not a freeze-frame.
_FIXATION_DRIFT = 0.05

# How long the head takes to react to a glance at all, and how long it takes to
# give up and drift back. The head is heavy and it does not commit.
_FOLLOW_RISE = 0.25
_FOLLOW_FALL = 1.2
# Peak of (1 - e^-a/rise) * e^-a/fall, so the configured gain means what it says.
_FOLLOW_Z = _FOLLOW_RISE / (_FOLLOW_RISE + _FOLLOW_FALL)
_FOLLOW_PEAK = (1.0 - _FOLLOW_Z) * math.exp(
    _FOLLOW_RISE * math.log(_FOLLOW_Z) / _FOLLOW_FALL
)

# Fraction of the moving part of a blink spent closing. The lid drops in about
# a third of the time it takes to lift again; a symmetric blink is the
# difference between a blink and a slow wink.
_BLINK_CLOSE = 0.35
# How many frame periods the eye is held fully shut in the middle of one. A real
# blink rests at the bottom for a few tens of milliseconds, and it has to here
# too for a duller reason: the bottom of a curve is a single instant, and at 30
# fps the frames either side of it can both land two thirds of the way down, so
# without a plateau some blinks render as a flutter. Above one frame period, so
# at least one frame is guaranteed to catch the eye actually closed.
_BLINK_SHUT_FRAMES = 1.2
_BLINK_SHUT_MAX = 0.35
# How often a blink is really two. Never doing it is its own tell.
_DOUBLE_BLINK = 0.12

# Time constants for the speech envelope: quick to believe she has started
# talking, slow to believe she has stopped.
_SPEECH_RISE = 0.25
_SPEECH_FALL = 2.0


# -- breathing ----------------------------------------------------------------
#
# Fraction of a breath spent drawing in. Breathing is not a sine: the inhale is
# the shorter half and the exhale trails, and at rest the ratio is roughly two
# to three. A symmetric wave at the same rate reads as a sigh over and over.
_BREATH_INHALE = 0.4
# The rate wanders. Nothing breathes to a metronome, and a fixed period is
# picked up surprisingly fast on something you are watching idle. Applied as a
# phase offset rather than a rate, so it stays a pure function of t and the
# phase can never jump backwards.
_BREATH_VARY = ((23.0, 1.4), (13.0, 0.3))


def _breath_shape(u: float) -> float:
    """-1 fully exhaled, +1 at the top of an inhale, for phase u in 0..1."""
    if u < _BREATH_INHALE:
        return -math.cos(math.pi * u / _BREATH_INHALE)
    return math.cos(math.pi * (u - _BREATH_INHALE) / (1.0 - _BREATH_INHALE))


def breath(t: float, per_min: float) -> float:
    """Where in a breath she is at time `t`, -1..1."""
    if per_min <= 0:
        return 0.0
    phase = t * per_min / 60.0 + 0.15 * _drift(t, _BREATH_VARY)
    return _breath_shape(phase % 1.0)


# -- speech beats -------------------------------------------------------------
#
# The nod that lands on a stressed syllable. Onsets are taken from the raw mouth
# level and not from the envelope -- the opposite of the amplitude coupling,
# and deliberately: the envelope exists to smooth the syllables away, and the
# syllables are exactly what a beat is looking for.
#
# Not every syllable gets one. Stress falls on maybe one in three, and a head
# that nods on all of them is a head keeping time rather than talking.
_BEAT_HI = 0.38          # rising through this, having been below _BEAT_LO,
_BEAT_LO = 0.18          # is an onset
_BEAT_REFRACTORY = 0.13  # no two beats closer than this
_BEAT_DUR = 0.30


def _beat_shape(u: float) -> float:
    """One nod: down and back, over u in 0..1. Zero at both ends."""
    return math.sin(math.pi * u) * (1.0 - u) * 2.0


# -- posture ------------------------------------------------------------------
#
# She holds a position, then changes it. Sums of sines never do: they are always
# mid-movement, and measured, the head was under half a degree per second only
# 18% of the time. So the wander is gated -- quiet while a posture is held, back
# to full while she shifts to the next one, with the held offset moving as she
# goes.
_HOLD = (4.0, 14.0)      # seconds settled in one position
_SHIFT = (0.6, 1.4)      # seconds spent moving to the next


def _saccade_shape(u: float) -> float:
    """Position through a saccade, 0..1, for normalised time u in 0..1.

    Smoothstep. A real saccade's velocity profile is a slightly skewed bell and
    this one is a symmetric parabola, which is a distinction no frame rate that
    can resolve a 40 ms movement at all will ever show.
    """
    return u * u * (3.0 - 2.0 * u)


def _blink_shape(u: float, shut_for: float = 0.0) -> float:
    """How shut the eye is, 0..1, for progress u through a blink.

    Asymmetric: smoothstep down over the first third of the movement, a beat
    held shut, then off the bottom quickly and settling open over the rest.
    """
    close = _BLINK_CLOSE * (1.0 - shut_for)
    if u < close:
        v = u / close
        return v * v * (3.0 - 2.0 * v)
    if u < close + shut_for:
        return 1.0
    v = (u - close - shut_for) / max(1e-6, 1.0 - close - shut_for)
    return (1.0 - v) ** 2


class IdleMotion:
    """Head, body, gaze and blink for an avatar nobody is tracking."""

    def __init__(self, cfg: Config, seed: int | None = None):
        self.cfg = cfg
        # Its own generator: blink and saccade timing must not depend on, or
        # perturb, the global random state that anything else might be using.
        self._rng = random.Random(seed)

        # Blink. `_blink_span` is per-blink, not a constant: identical blinks
        # back to back are as mechanical as identical intervals between them.
        self._blink_at = self._rng.uniform(*cfg.vtube_blink_every)
        self._blink_start = -99.0
        self._blink_span = cfg.vtube_blink_ms / 1000
        self._double = False           # the next blink is the second of a pair

        # Gaze. Starts centred, with the first saccade planned on frame one.
        self._from = (0.0, 0.0)
        self._to = (0.0, 0.0)
        self._saccade_at = 0.0
        self._saccade_for = _SACCADE_MIN
        self._hold_until = 0.0

        # The head's outstanding answer to the last large glance.
        self._follow = (0.0, 0.0)
        self._follow_at = -99.0

        # Speech envelope, and the timestamp it was last advanced at.
        self._speech = 0.0
        self._last_t: float | None = None

        # Speech beats: where the last nod started, how hard, and whether the
        # mouth has dropped far enough since for the next onset to count.
        self._beat_at = -99.0
        self._beat_size = 0.0
        self._beat_roll = 0.0
        self._armed = True

        # Posture: the offset she is currently holding, the one she is moving
        # to, and when this phase ends.
        self._posture_from = (0.0, 0.0, 0.0)
        self._posture_to = (0.0, 0.0, 0.0)
        self._posture_at = 0.0
        self._posture_until = 0.0
        self._shifting = False

    # -- speech ------------------------------------------------------------
    def speaking(self, t: float, level: float) -> float:
        """How much she is talking right now, as an envelope of `level`.

        `level` is the mouth-open value, which falls to zero between every
        syllable. Gaze and blink rate answer to being in the middle of saying
        something, not to the current phoneme, so they get the envelope of it
        rather than the thing itself.
        """
        dt = 0.0 if self._last_t is None else min(0.5, max(0.0, t - self._last_t))
        self._last_t = t
        tau = _SPEECH_RISE if level > self._speech else _SPEECH_FALL
        self._speech += (level - self._speech) * (1.0 - math.exp(-dt / tau))
        return _clamp(self._speech, 0.0, 1.0)

    # -- gaze --------------------------------------------------------------
    def _target(self, speech: float) -> tuple[float, float]:
        """Where to look next, both axes within -vtube_gaze..vtube_gaze."""
        cfg, rng = self.cfg, self._rng
        ecc = cfg.vtube_gaze

        # Back to roughly straight ahead -- at whoever she is talking to.
        # Speaking makes this *less* likely, not more: people look away while
        # they assemble a sentence and come back on the way out of it, and that
        # alternation is most of what makes a gaze belong to a conversation
        # rather than to a screensaver.
        if rng.random() < cfg.vtube_gaze_home * (1.0 - 0.5 * speech):
            return (_clamp(rng.gauss(0.0, 0.12 * ecc), -ecc, ecc),
                    _clamp(rng.gauss(0.0, 0.10 * ecc), -ecc, ecc))

        # Most saccades are small. The skewed draw puts the weight there and
        # leaves a thin tail of glances right across the room.
        amp = ecc * rng.random() ** 1.8
        angle = rng.uniform(0.0, 2.0 * math.pi)
        return (amp * math.cos(angle), amp * math.sin(angle) * _VERTICAL)

    def _plan(self, t: float, speech: float) -> None:
        """Choose the next target and how long the flick and the hold take."""
        cfg, rng = self.cfg, self._rng
        # A saccade is never interrupted, so the position at planning time is
        # always the last target.
        self._from = self._to
        self._to = self._target(speech)

        dx = self._to[0] - self._from[0]
        dy = self._to[1] - self._from[1]
        reach = max(1e-6, 2.0 * cfg.vtube_gaze)
        span = min(1.0, math.hypot(dx, dy) / reach)

        self._saccade_at = t
        self._saccade_for = _SACCADE_MIN + (_SACCADE_MAX - _SACCADE_MIN) * span

        lo, hi = cfg.vtube_gaze_hold
        # Skewed to the short end, stretched after a big jump -- she looks at
        # the thing she just looked at -- and compressed while she is talking,
        # when the eyes are working rather than resting.
        hold = lo + (hi - lo) * rng.random() ** 1.7
        hold *= (1.0 + 0.6 * span) / (1.0 + 0.7 * speech)
        self._hold_until = t + self._saccade_for + hold

        if span < _BIG_SHIFT:
            return
        self._follow = (dx / reach, dy / reach)
        self._follow_at = t
        # Not on top of a blink already running: retriggering would restart it
        # halfway and stretch it into a wince.
        if (cfg.vtube_blink and rng.random() < _SHIFT_BLINK
                and t - self._blink_start > self._blink_span):
            self._blink_at = min(self._blink_at, t + 0.02)

    def gaze(self, t: float, speech: float = 0.0) -> tuple[float, float]:
        """Where the eyes point at time `t`, before the head is accounted for.

        Fixate, flick, fixate. This also decides where the head is about to go
        and whether a blink comes along with the next jump, so it has to run
        before either of those is read.
        """
        if t >= self._hold_until:
            self._plan(t, speech)

        elapsed = t - self._saccade_at
        if elapsed < self._saccade_for:
            u = _saccade_shape(elapsed / self._saccade_for)
            x = self._from[0] + (self._to[0] - self._from[0]) * u
            y = self._from[1] + (self._to[1] - self._from[1]) * u
        else:
            x, y = self._to

        wander = _FIXATION_DRIFT * self.cfg.vtube_gaze
        return (x + wander * _drift(t, _WAVES[GAZE_X[0]]),
                y + wander * _drift(t, _WAVES[GAZE_Y[0]]))

    def head_follow(self, t: float) -> tuple[float, float]:
        """The head's late, partial answer to the last big glance, -1..1 each.

        Rises over a quarter of a second and then falls away over about a
        second: the head starts after the eyes have already arrived, never
        quite catches up, and gives the position back rather than holding it.
        """
        age = t - self._follow_at
        if age < 0.0 or age > 5.0 * _FOLLOW_FALL:
            return (0.0, 0.0)
        shape = ((1.0 - math.exp(-age / _FOLLOW_RISE))
                 * math.exp(-age / _FOLLOW_FALL) / _FOLLOW_PEAK)
        return (self._follow[0] * shape, self._follow[1] * shape)

    # -- speech beats ------------------------------------------------------
    def beat(self, t: float, level: float) -> tuple[float, float]:
        """The nod happening right now, as (pitch, roll) in head units.

        Fires on a rising mouth level, not on the envelope: the syllable is the
        thing being looked for. Each beat gets its own size and a small roll of
        its own sign, so a run of them does not read as a metronome.
        """
        cfg = self.cfg
        if cfg.vtube_beats <= 0:
            return (0.0, 0.0)

        if level < _BEAT_LO:
            self._armed = True
        elif (self._armed and level > _BEAT_HI
                and t - self._beat_at > _BEAT_REFRACTORY):
            self._armed = False
            if self._rng.random() < cfg.vtube_beat_chance:
                self._beat_at = t
                # Louder syllables get bigger nods, which is most of what makes
                # the timing look like it belongs to the sentence.
                self._beat_size = cfg.vtube_beats * (0.5 + 0.5 * level)
                self._beat_roll = self._rng.uniform(-0.5, 0.5)

        age = t - self._beat_at
        if age < 0.0 or age >= _BEAT_DUR:
            return (0.0, 0.0)
        shape = _beat_shape(age / _BEAT_DUR) * self._beat_size
        # Down, not up: a beat drops the chin and brings it back.
        return (-shape, shape * self._beat_roll)

    # -- posture -----------------------------------------------------------
    def posture(self, t: float) -> tuple[float, float, float, float]:
        """(wander gain, yaw, pitch, roll offsets) for the settling cycle."""
        cfg = self.cfg
        if t >= self._posture_until:
            self._plan_posture(t)

        span = max(1e-6, self._posture_until - self._posture_at)
        u = _clamp((t - self._posture_at) / span, 0.0, 1.0)
        if not self._shifting:
            # Held. Full stillness in the middle, easing out of and back into
            # the shifts either side so nothing starts or stops abruptly.
            edge = min(1.0, min(u, 1.0 - u) / 0.15)
            gain = 1.0 - (1.0 - cfg.vtube_settle) * (edge * edge * (3 - 2 * edge))
            return (gain,) + self._posture_to

        eased = u * u * (3.0 - 2.0 * u)
        offsets = tuple(a + (b - a) * eased
                        for a, b in zip(self._posture_from, self._posture_to))
        return (1.0,) + offsets

    def _plan_posture(self, t: float) -> None:
        rng = self._rng
        self._posture_at = t
        if self._shifting:
            self._shifting = False
            self._posture_until = t + rng.uniform(*_HOLD)
            return
        self._shifting = True
        self._posture_until = t + rng.uniform(*_SHIFT)
        self._posture_from = self._posture_to
        # Offsets are in the same units as the wander, and deliberately smaller
        # than it: this is settling into a slightly different position, not
        # striking a pose.
        self._posture_to = (rng.uniform(-0.5, 0.5),
                            rng.uniform(-0.4, 0.4),
                            rng.uniform(-0.4, 0.4))

    # -- blink -------------------------------------------------------------
    def _schedule_blink(self, t: float, speech: float) -> None:
        """Fix this blink's duration and the wait until the next one."""
        cfg, rng = self.cfg, self._rng
        base = cfg.vtube_blink_ms / 1000
        second = self._double
        self._double = (not second) and rng.random() < _DOUBLE_BLINK
        self._blink_span = base * (0.8 if second else rng.uniform(0.85, 1.25))
        if self._double:
            wait = rng.uniform(0.10, 0.20)
        else:
            lo, hi = cfg.vtube_blink_every
            # Talking roughly doubles the rate. Blink rate is one of the few
            # involuntary things that tracks speech, and holding it flat
            # through a sentence is what makes an idle loop look like a loop.
            wait = rng.uniform(lo, hi) / (1.0 + cfg.vtube_blink_speech * speech)
        self._blink_at = t + self._blink_span + wait

    def eye_openness(self, t: float, speech: float = 0.0) -> float:
        """1.0 open, dipping to 0.0 through a blink."""
        if not self.cfg.vtube_blink:
            return 1.0
        if t >= self._blink_at:
            self._blink_start = t
            self._schedule_blink(t, speech)
        elapsed = t - self._blink_start
        if 0.0 <= elapsed < self._blink_span:
            shut_for = min(_BLINK_SHUT_MAX, _BLINK_SHUT_FRAMES
                           / self.cfg.vtube_lipsync_fps / self._blink_span)
            return 1.0 - _blink_shape(elapsed / self._blink_span, shut_for)
        return 1.0

    # -- everything else ---------------------------------------------------
    def frame(self, t: float, level: float = 0.0) -> dict[str, float]:
        """The parameters to inject at time `t`, given how loudly she is talking.

        Speech widens the movement rather than adding a separate gesture: people
        move more while making a point and settle when they stop, and that
        coupling is most of what separates alive from idling.
        """
        cfg = self.cfg
        level = _clamp(level, 0.0, 1.0)
        speech = self.speaking(t, level)
        # The envelope, not the raw level -- the same distinction gaze and blink
        # already make, and for the same reason. `level` is the mouth parameter,
        # which swings between nothing and wide open at syllable rate, and
        # multiplying the head's slow drift by it amplitude-modulates the head
        # at 4-5 Hz. Measured, that was 14.3 deg/s of average head movement
        # against 3.6 with the ripple removed: not a nod, not a gesture, just a
        # tremor proportional to wherever the drift happened to be. Widening on
        # the envelope is what the line above actually describes.
        gain = 1.0 + cfg.vtube_motion_speech * speech
        head, body = cfg.vtube_motion_head * gain, cfg.vtube_motion_body * gain

        # Gaze first: it is what decides where the head is going next.
        gaze_x, gaze_y = self.gaze(t, speech)
        # Sign of the head parameter against the eye one, per axis: on this rig
        # they disagree horizontally and agree vertically. See the config.
        sign_x = -1.0 if cfg.vtube_gaze_flip_x else 1.0
        sign_y = -1.0 if cfg.vtube_gaze_flip_y else 1.0
        follow_x, follow_y = self.head_follow(t)

        # Settling gates the wander; the offsets are the position being held.
        wander, off_yaw, off_pitch, off_roll = self.posture(t)
        beat_pitch, beat_roll = self.beat(t, level)
        air = breath(t, cfg.vtube_breath_per_min)

        yaw = (wander * _drift(t, _WAVES[HEAD_YAW]) + off_yaw
               + sign_x * cfg.vtube_gaze_head * follow_x)
        pitch = 0.6 * (wander * _drift(t, _WAVES[HEAD_PITCH]) + off_pitch
                       + sign_y * cfg.vtube_gaze_head * follow_y)
        # Tilt goes partly with the turn. Left to its own sine it correlated
        # with the turn at r = 0.00, which is two animations, not one neck.
        roll = (wander * 0.5 * _drift(t, _WAVES[HEAD_ROLL]) + off_roll
                + cfg.vtube_head_tilt_follow * yaw)

        out = {
            # Breath and beats are in degrees already, so they go on after the
            # speech gain rather than being widened by it: she does not breathe
            # deeper because she is mid-sentence, and a beat is its own size.
            HEAD_YAW:   head * yaw,
            HEAD_PITCH: head * pitch + cfg.vtube_breath * air
                        + cfg.vtube_motion_head * 0.12 * beat_pitch,
            HEAD_ROLL:  head * roll + cfg.vtube_motion_head * 0.12 * beat_roll,
            # Both of these reach nothing on 'yiyi_chair' -- see the note at the
            # top. They are still driven correctly for rigs that map them: the
            # lean goes with the turn, and the rise and fall is the breath.
            BODY_X:     body * (wander * _drift(t, _WAVES[BODY_X])
                                + cfg.vtube_head_tilt_follow * yaw),
            BODY_Y:     body * 0.5 * air,
        }

        # Vestibulo-ocular: the eyes hold their point while the head moves
        # under them. This is also what walks the eyes back towards centre as
        # the head catches up with a glance -- head_follow above pushes the
        # head one way and this pulls the eyes the other, so the two couplings
        # together produce eyes-lead-head-follows out of nothing but their
        # signs. Get a sign wrong and they swing out together instead, which
        # is the failure mode the two flip flags exist for.
        vor = cfg.vtube_gaze_vor * cfg.vtube_gaze
        gaze_x = _clamp(gaze_x - sign_x * vor * _clamp(yaw, -1.5, 1.5))
        gaze_y = _clamp(gaze_y - sign_y * vor * _clamp(pitch, -1.5, 1.5))

        opened = self.eye_openness(t, speech)
        if cfg.vtube_gaze > 0.0:
            # Lids follow the eyes down. Looking at the floor with the eyes
            # still wide open is one of the specific things that reads as a
            # doll rather than as a person. Upward gaze widens them a little in
            # a real face, but that is mostly brow, and there is no brow here.
            below = _clamp(gaze_y / (cfg.vtube_gaze * _VERTICAL), -1.0, 0.0)
            opened *= 1.0 + cfg.vtube_lid_follow * below
        opened = _clamp(opened, 0.0, 1.0)

        # Brows lift with an upward glance and with the emphasis of a beat.
        # The lid coupling below covers looking down; this is the other half of
        # it, and the half that needs a brow to exist.
        #
        # Measured from neutral, not from zero: on this rig zero is the bottom
        # of the brow's travel, so a lift of "none" sent as 0 pins them at one
        # extreme and holds them there. The two reasons are summed and capped
        # before scaling, so the top of the movement cannot exceed the knob.
        lift = 0.0
        if cfg.vtube_gaze > 0.0:
            up = _clamp(gaze_y / (cfg.vtube_gaze * _VERTICAL), 0.0, 1.0)
            lift = _clamp(up + 0.6 * max(0.0, -beat_pitch), 0.0, 1.0)
        out[BROWS] = _clamp(cfg.vtube_brows_neutral + cfg.vtube_brows * lift,
                            0.0, 1.0)

        for eye in EYES_OPEN:
            out[eye] = opened
        for eye in GAZE_X:
            out[eye] = gaze_x
        for eye in GAZE_Y:
            # Offset only on the way out. Everything above -- the lid coupling,
            # the brows, the vestibulo-ocular term -- works in a space where
            # zero means straight ahead, and only the rig disagrees.
            out[eye] = _clamp(cfg.vtube_gaze_y_neutral + gaze_y)
        return out

    def rest(self) -> dict[str, float]:
        """Everything centred, eyes open. Sent on the way out so she is not
        left frozen at whatever angle the last frame happened to land on."""
        out = {k: 0.0 for k in (HEAD_YAW, HEAD_PITCH, HEAD_ROLL,
                                BODY_X, BODY_Y)}
        out.update({k: 0.0 for k in GAZE_X})
        out.update({k: self.cfg.vtube_gaze_y_neutral for k in GAZE_Y})
        # Not zero: zero is one end of the brow's travel, not the middle of it.
        out[BROWS] = self.cfg.vtube_brows_neutral
        out.update({k: 1.0 for k in EYES_OPEN})
        return out
