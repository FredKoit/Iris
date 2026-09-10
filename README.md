# Iris

A local voice AI you talk to out loud. Speech in, personality, speech out, no
account and no network. Everything runs on this laptop.

Built and measured on an RTX 3050 Ti Laptop (4 GB) + Ryzen 5 5600H. Every
component is free and open source.

## What runs where

The 4 GB of VRAM is built around the language model. Everything else runs on
the CPU, which is why this fits at all — though the model she ships with,
qwen3, does not quite fit whole itself: 27% of it spills to CPU too. `llama3.2`
fits with room to spare and is faster; see "The brain", further down, for why
qwen3 is the default anyway.

| stage         | what                                      | where                 | measured                |
| ------------- | ----------------------------------------- | --------------------- | ----------------------- |
| endpointing   | Silero VAD v6 (ships with faster-whisper) | CPU                   | ~1 ms/frame             |
| transcription | faster-whisper `base.en`, int8            | CPU, 6 threads        | 0.70 s, **hidden**      |
| brain         | `qwen3:4b-instruct-2507` via Ollama       | GPU + 27% CPU, 3.2 GB | 0.56 s to first phrase  |
| speech        | Kokoro-82M fp32 ONNX                      | CPU, 4 threads        | 0.6 s opener (RTF 0.43) |

Run `python scripts/latency.py` to reproduce the component timings on your own
machine.

### Why transcription is free

Whisper's encoder runs over a padded 30-second window no matter how little you
said, so a one-word reply costs about as much as a long one: measured here,
"Enough." took 0.67 s and a 25-second ramble took 1.44 s. It is a fixed cost,
not a per-second one.

That fixed cost is spent during silence we were going to wait out anyway. When a
pause reaches `speculate_ms` (160 ms), the mic hands the audio to Whisper
immediately; if you start talking again the guess is thrown away, and if you
don't, the transcript is already in hand when the endpoint fires.

Measured from end of speech to transcript: **2.21 s sequential, 1.47 s
speculative** — and the two paths produce identical text. In the integration
test the transcript was ready in 0.00 s.

Look for `prefetched` or `cold` in the `[stt ...]` line to see which path a turn
took.

## Setup

Ollama and the Python 3.12 venv are already installed here. From scratch:

```
python -m uv venv --python 3.12 .venv
python -m uv pip install --python .venv/Scripts/python.exe \
    sounddevice numpy faster-whisper ollama kokoro-onnx
python scripts/fetch_models.py                    # Kokoro weights, ~350 MB
ollama pull qwen3:4b-instruct-2507-q4_K_M          # the default brain
```

Whisper downloads itself on first run.

The tray icon, global hotkeys, push-to-talk, and the VTube Studio avatar are
all optional. Each one is silently absent with a one-line message if its
package is missing, so none of this is required just to talk to her:

```
python -m uv pip install --python .venv/Scripts/python.exe \
    pystray pillow pynput websocket-client
```

Reaching the volume needs two more. Without them she still hears "turn it
down", and still says she cannot:

```
python -m uv pip install --python .venv/Scripts/python.exe pycaw comtypes
```

## Use

```
python run.py               talk to her
python run.py --text        type instead, same voice and brain
python run.py --barge-in    interrupt her mid-sentence (headphones only)
```

Every other flag:

| flag                                | effect                                           |
| ----------------------------------- | ------------------------------------------------ |
| `--push-to-talk`                    | only listen while a key is held — see below      |
| `--no-idle`                         | never speak first; only answer when spoken to    |
| `--idle-after SECONDS`              | how long she waits before speaking first         |
| `--no-tray`                         | no tray icon and no global hotkeys               |
| `--no-screen`                       | do not tell her which application is in front    |
| `--no-timers`                       | do not act on "remind me in ten minutes"         |
| `--no-actions`                      | do not touch volume, brightness, media or apps   |
| `--model`, `--voice`, `--whisper`   | override the config for one run                  |
| `--input-device`, `--output-device` | a mic or speaker by index, from `--list-devices` |
| `--list-devices`, `--list-voices`   | print what is available                          |

### Push to talk

By default she answers everything she hears, which is the right behaviour for a
companion and the wrong one when somebody else is in the room. `--push-to-talk`
holds the microphone shut until you hold a key:

```
python run.py --push-to-talk          hold F7 to speak
```

The key is `hotkey_talk`, F7 by default. A single key rather than a chord,
because this one is held rather than tapped, and it is **not** suppressed -- so
whatever else is bound to it still fires. Pick one your game does not use.

Letting go ends the turn immediately. The microphone gate used to have only two
states, and muting it _discarded_ whatever was buffered -- correct for
half-duplex, where the thing being thrown away is her own voice coming back
through the speakers, and exactly wrong here, where it would throw away the
sentence you just said. `MicListener.finish()` is the other half: emit what is
buffered, then reset. It is checked before the mute, because releasing the key
sets both and the other order loses the audio.

It also means you do not sit through the trailing-silence timer to prove you
have stopped talking, which is `silence_ms` -- half a second -- off every turn.

If `pynput` is missing she stays listening normally rather than going silently
and permanently deaf, which is the failure that actually matters here.

The tray icon follows the key, so there is somewhere to look when you are not
sure whether she can hear you:

| icon | meaning |
|---|---|
| solid dot | listening |
| hollow ring | push-to-talk is armed and the key is up |
| struck through | microphone deliberately off, from the tray or the hotkey |

A deliberate mute wins over the key: holding it while muted leaves her muted.
The icon is redrawn on every press and release but the menu is not, since no
label can have changed and somebody may be holding the key down. Hovering gives
the same thing in words, including which key -- the icon cannot say `<f7>`.

### One gate, three owners

Three separate things want the microphone shut and they have to survive each
other: the tray mute is a deliberate switch that outlives any reply, half-duplex
closes it for the length of one, and push-to-talk closes it whenever the key is
up. Each used to set and clear `listener.muted` directly, which is what let a
reply finishing reopen a microphone the user had muted from the tray.

They are now separate flags folded together in `Iris._update_mic()`, the only
thing allowed to touch the gate. Push-to-talk is read from whether the key
listener is actually running, not from the config flag, so a listener that
failed to start cannot leave her unable to hear anything at all.

### Headphones vs speakers

On speakers her own voice re-enters the microphone. She then hears herself,
interrupts herself, and transcribes herself in a loop. So the default is
half-duplex: the mic is deaf while she talks. Put headphones on and pass
`--barge-in` to cut her off mid-sentence, which is most of what makes her feel
alive.

## Tuning

Everything lives in [iris/config.py](iris/config.py).

- **Too slow?** `whisper_model = "tiny.en"` halves transcription (0.75 s → 0.39 s),
  but since speculation already hides that cost it now buys you very little.
  Leave it on `base.en` and keep the accuracy.
- **She answers before you finish a thought?** Raise `speculate_ms`, or set it
  above `silence_ms` to turn speculation off entirely. It never changes _when_
  she replies — only whether the transcript is ready by then — so this is a CPU
  knob, not a behaviour one.
- **She cuts you off mid-sentence?** Raise `silence_ms`.
- **She responds to the fridge?** Raise `vad_threshold` toward 0.7.
- **Do not switch TTS to int8.** The int8 export is 3x _slower_ than fp32 here:
  Zen 3 has no VNNI, so the quantised kernels fall back to slow paths.
  Measured: RTF 1.27 int8 vs 0.43 fp32.

## Knowing when she is

A language model has no clock. Asked how long we had been talking, she answered
"probably around last night's pizza dinner"; asked the day, "Tuesday...
probably Thursday". [awareness.py](iris/awareness.py) puts the time, the day,
how long this session has run and how long since the last one into the system
prompt on every turn.

She uses it both ways: asked directly she now answers "It's 16:23" and
"Saturday", and unprompted at 03:12 after a week away she opened with "finally
having some conversation after hours".

One catch worth knowing if you edit the persona: it used to say she cannot know
things she has no way of knowing, and a 3B model resolved that against the clock
and went back to "I'm not really sure". The persona now carves out an explicit
exception. If she starts claiming ignorance about the time again, that
contradiction is why.

## Knowing what you are doing

She floats above whatever else is on the screen and, until now, had no idea what
that was -- a companion pinned over a game that cannot tell a game from a
spreadsheet is unmoored in the same way a companion with no clock is. She is now
told which application has focus and how long it has held it today.

It costs one line in the system prompt:

```
On screen in front of them: "ELDEN RING" (up about 2 hours today)
```

and it turns "...Hey." into "...Hey. Still on ELDEN RING. Been at it two hours."

**This replaces invention, it does not add it.** Measured over sixteen bare
"hey"s with no watcher running at all, she already made the screen up -- "Still
on Word.", "Still on Notepad.", "The screen is warm." A model given a persona
that lives on your laptop will talk about your laptop whether or not it knows
anything. With the watcher on, she used the real application in 15 of 16.

Identity is taken from the process, not the window title. A title changes every
time you switch file or browser tab, so tracking on it would restart the clock a
hundred times an hour and never notice somebody had been in the same editor all
evening. The totals are cumulative rather than the current run, for the same
reason: people alt-tab constantly, and "you have had that game up for two hours"
is the true and interesting statement where "in front for forty seconds" is
neither. Her own window is excluded, so clicking on her does not make her forget
what you were doing.

### It is a label, not a sentence

The line is written as a parenthesised label on purpose. The first version was
prose -- "They have had it up for about two hours so far today" -- and she handed
the clause straight back, saying "that's how long you've had it open today" out
loud. A well-formed sentence in the prompt is ready to speak. A label has to be
reworded before it can be said at all, and rewording it is the point.

The prohibition at the end of the block had to be extended too. The screen line
gave her one more thing to read out, and "never read the clock out unless they
ask" was not enough on its own; naming each failure individually was. She also
has to be told not to narrate it -- unprompted, she produced "No other actions
detected. Screen remains locked on that window", which is a monitoring readout,
not an observation. That is now forbidden outright and did not recur in sixteen
samples.

Restructuring the _clock_ lines the same way was tried and did not help: 1 of 16
against 0 of 16, which is noise. The prohibition did the work there, so the clock
lines were left as they were.

Retried later under a bait built to detect it -- 01:20, a script of silences to
fill, where she volunteers the time far more than in ordinary conversation --
and it still did not help: 3 of 50 as a sentence against 6 of 50 as a label. The
original call stands. Volunteering the clock is the one prohibition with no
structural fix found; it runs between 6% and 30% of replies depending on how
much silence there is to fill, and the prohibition is all that holds it down.

### Shell windows are not something you are doing

This came out of her own logs:

> "System tray overflow window. Explorer. That's all. Whatever."

Forbidden, and not really her fault -- she was asked what was on screen and told
the truth about a window that had taken the foreground for a moment. The tray
flyout, the desktop, the alt-tab overlay: all of them steal focus, and none of
them is a thing anybody is doing.

`screen_ignore` already handled this for her own window, but it matches on the
process, and the process behind all of those is `explorer` -- which is also File
Explorer, which somebody really is using. So `screen_ignore_titles` matches on
the title instead, and an ignored reading leaves the last real application in
front rather than replacing it.

Baited with an ugly title at 01:20, reading the title out word for word fell
from 25-30% of replies to 10%, and what is left is her answering the direct
question "what's on my screen", where quoting it is a defensible answer.

`persona.txt` carries the carve-out, for the same reason the clock needed one --
told flatly that she cannot see, a 3B model refuses to use what it has been
given. Asked outright whether she can see your screen she now answers "I know
which application is in front of you", which is true, brief and in character.

### What it can see, and what it never sends

Nothing leaves the laptop. Ollama is on 127.0.0.1 and the prompt is never written
to disk -- window titles reach memory only if she says one out loud, and the
extractor grounds facts against your words alone, so nothing here can become a
stored fact on its own.

Window titles are still the most revealing single string on a machine, so:

| knob               | what it does                                                                |
| ------------------ | --------------------------------------------------------------------------- |
| `screen`           | off entirely; `python run.py --no-screen` for one session                   |
| `screen_private`   | told the application's name, never its title. Password managers by default  |
| `screen_ignore`    | not counted at all. Her own window by default                               |
| `screen_title_max` | titles run long; she needs the gist                                         |
| `screen_settled_s` | below this she has only just switched, and a duration is not an observation |

Windows only -- it is all user32/kernel32. Elsewhere she simply does not have
this sense, the same way the overlay is silently absent.

## Speaking first

She does not wait to be spoken to. After `idle_after_s` of silence (45 by
default) she says something unprompted, drawing on memory and the clock:

```
iris (unprompted): It's 16:32 on a Saturday and we're just getting to the
                   good parts of the day.
iris (unprompted): It's ridiculously late for someone who studies computer
                   science.
```

Each unanswered remark waits longer than the last -- 45 s, then 77 s, then
130 s -- and after three she gives up until spoken to. Something that talks to
an empty room on a fixed timer stops being company and becomes a noise source.
She also stays quiet while she is already talking and while a consolidation pass
is running.

`--no-idle` turns it off; `--idle-after 20` makes her pushier.

### The nudge is the whole trick

She is prompted with a stage direction that is never stored in history, so she
does not learn that stage directions are a thing people say. Every clause in it
is load-bearing, and the first draft got it wrong: written in the third person
("they have gone quiet"), she answered in the third person too --

> "They've been staring out that window for hours and I'm starting to think
> Pixel has taken over their life completely."

-- narrating the user like a documentary, and inventing a room she cannot see.
Rewritten to address them as "you", with an explicit reminder that she has no
eyes, it produces the lines above instead. If she starts narrating you again,
that is the string to look at.

## The avatar

She drives a Live2D model in VTube Studio: expressions from her emotion cues,
and lip sync from her own audio. Start VTube Studio, turn the API on (gear icon,
port 8001), and run her -- the first connection pops an approval dialog in VTS,
after which the token is kept in `.vts_token` and it connects silently.

```
python scripts/avatar.py            what the model exposes and how cues map
python scripts/avatar.py --sweep    fire every hotkey in turn to see what each does
python scripts/avatar.py --try NAME fire one hotkey
python scripts/avatar.py --mouth    lip sync test, no speech
python scripts/avatar.py --probe    which VTS input moves what, and which are inverted
python scripts/avatar.py --eyes     gaze and blink, drawn as a trace you can read
python scripts/avatar.py --say "hi" speak a line with the mouth following
```

`--eyes` takes `--talk` to drive it as if she were speaking, and `--dry` to
print the trace without VTube Studio running at all.

If VTube Studio is not running, the API is off, or approval is refused, she
talks normally and the avatar is skipped. It is never allowed to break a
conversation.

### Where she is looking

Nothing tracks a face here, so her head, body and eyes are synthesised. The head
and body wander along sums of sines at periods that never line up. The eyes do
not, and that distinction is the whole thing: real eyes fixate and jump, holding
a point for anywhere from a fifth of a second to a couple of seconds and then
flicking to the next one inside a tenth of a second. Smooth eye movement happens
only when tracking something that is itself moving. Eyes that slide are the
single most recognisable tell of an idle animation, because in a real face they
mean something is medically wrong.

Three couplings hang off the saccades, and they carry as much of it as the
saccades do -- eyes that move correctly but in isolation just look like a second
animation playing over the first:

- **The lids follow the eyes down.** Wide open eyes aimed at the floor read as a
  doll.
- **The head chases a large glance,** late and only part of the way. About one
  saccade in eight is big enough to recruit it.
- **The eyes hold their point while the head moves under them,** which is the
  vestibulo-ocular reflex, and is also what settles her eyes back towards centre
  once the head has caught up.

Head and body carry four more things, each of which was measured before and
after:

| | before | after |
|---|---|---|
| breathing | 3-7 per minute, on a channel that moved nothing | **15/min** on head pitch, 40% inhale |
| speech | idle wander scaled up, r=0.96 with silence | **beats** on ~1 syllable in 3, r=0.58 |
| turn vs tilt | r = 0.00, three animations on one body | **r = 0.41** |
| ever still | under 0.5 deg/s **18%** of the time | **29%**, holding a posture then shifting |

The last two produce the thing you actually recognise -- eyes first, head after,
eyes easing back -- out of nothing but their signs. Which means the signs have
to be right, and the sign of `FaceAngleX` against `EyeLeftX` is a property of the
rig that could not be checked without watching the model. Run `--eyes` and look:
if her head turns _away_ from what she just glanced at, or her eyes swing wide
instead of settling, flip `vtube_gaze_flip_x` in the config.

Speech drives all of it. She looks away more while assembling a sentence and
back at you on the way out of one, holds each point for less time, and blinks
about twice as often -- measured, 16 blinks a minute idle against 31 while
talking, against 15-20 and roughly double for a person.

### Zero is not always the middle

Three parameters on this rig map their whole input range across their whole
output range, which puts neutral at 0.5 rather than 0. Sending the zero a model
naturally produces pins them at one end:

| input | 0.00 | 0.50 | 1.00 |
|---|---|---|---|
| `EyeLeftY` → `ParamEyeBallY` | **-1.04** (clamped) | -0.04 | +0.96 |
| `Brows` → `ParamBrowLForm` | **-1.00** (clamped) | 0.00 | +1.00 |

Her eyes had been **looking fully down, hard against the stop, for as long as
the gaze code has existed** -- the entire vertical range crushed flat and
invisible. `vtube_gaze_y_neutral` and `vtube_brows_neutral` are where the middle
actually is. VTS declares both inputs with a default of 0, so nothing short of
sweeping the rig reveals this.

The same sweep says `EyeOpen` saturates by 0.5 of its input, so the lid-follow
coupling is nearly invisible here -- documented in the config rather than tuned
blind against one model's curve.

### What the rig actually listens to

`--probe` also answers a question that had been guessed wrong in the other
direction. `FacePositionX/Y/Z` were believed to reach `ParamBodyAngle`; on this
model they reach **nothing at all** -- zero of the rig's 97 parameters move.
Everything `vtube_motion_body` drives has been inert.

The body does move, just not from there: `FaceAngleX +24` carries
`ParamBodyAngleX` to `+7.9` on its own, so the lean comes free with the head
turn. `BODY_X`/`BODY_Y` are kept because the mapping is per-model and another rig
may honour them, but on `yiyi_chair` that knob changes nothing you can see.

Also unreachable: `ParamBreath` exists on the model and sits at 0.00 forever.
Nothing injectable touches it, so what the code calls breathing is a slow body
sway standing in for it -- and on this rig, one that does not move either.

### Floating her on the desktop

```
python scripts/overlay.py            transparent, borderless, always on top
python scripts/overlay.py --click    ...and clicks pass through to what is behind
python scripts/overlay.py --move 1200 300 --size 500 700
python scripts/overlay.py --restore  put the window back
```

Two things have to be set in VTube Studio first, because its API exposes neither
(`BackgroundListRequest` and `UIElementVisibilityRequest` both come back
"Unknown messageType"):

1. **A solid background colour.** Pure green is the usual choice. The default
   dark background is not quite uniform -- sampling four corners gave #191A1B,
   #18191A and #121213 -- and colour keying needs an exact match, so it would
   leave speckles.
2. **Hide the VTS interface.** The UI is drawn inside the same window, so it
   stays visible otherwise.

Then `python scripts/overlay.py --key 00FF00`.

Nothing is captured or re-rendered. VTube Studio's own window gets
`WS_EX_LAYERED` and a colour key, so Windows composites the background away;
the model stays perfectly in sync at no runtime cost. Verified on this machine:
a pixel over the background read `0x1B1A19` before and `0x3C3B3A` after, i.e.
it was showing the window behind.

Two things to know. Colour keying is binary, so anti-aliased edges keep a thin
fringe of the key colour, and anything in the model that is _exactly_ the key
colour goes transparent too -- pick a colour the model does not use. And
borderless means no title bar to drag, so move her with `--move` or
`--restore` first.

### Lip sync without a virtual audio cable

The usual recipe routes speaker output through VB-Cable into VTube Studio's
microphone input. That is not needed here: the audio is already sitting in the
player, so [player.py](iris/player.py) publishes the RMS of each block it sends
to the speakers and [vtube.py](iris/vtube.py) drives `MouthOpen` from it at
30 fps. No extra software, no device routing, and it cannot pick up room noise.

`vtube_mouth_gain` is 6.0, which is calibrated rather than guessed: over a real
spoken line her mouth then averages 0.40, reaches 0.84 on loud syllables,
saturates only 3% of the time, and closes fully through the 16% of blocks that
are silence. Gain 8 clips 16% of the time and looks like shouting.

Opening is faster than closing (`vtube_mouth_attack` 0.6, `vtube_mouth_release`
0.25) -- with symmetric smoothing the mouth flutters on every glottal stop.

### Expressions, and why not hotkeys

Her cues -- `[smug]`, `[angry]` -- are mapped to the model's expressions and
shown as she speaks. They print inline in the transcript, since they are
stripped before synthesis and there would otherwise be no sign anything moved:

```
you: iris: Nope, [angry] just tired of waiting for you to show up.
```

The obvious implementation is to trigger the matching hotkey, and it is wrong.
Those hotkeys are of type `ToggleExpression`, so `[smug]` followed by `[happy]`
leaves **both** switched on and her face silently accumulates every emotion of
the conversation. `ExpressionActivationRequest` sets a named expression on or
off explicitly, so each cue deactivates the previous one and lands where
intended. `[neutral]`, and any cue this model has no expression for, clears her
face rather than leaving the last one stuck.

### Models whose hotkeys are not in English

`vtube_map` in [config.py](iris/config.py) maps each cue to an expression by
name; without it, matching falls back to the cue name itself. The model here
names its expressions in Chinese, so the defaults are set accordingly:

| cue       | expression |                        |
| --------- | ---------- | ---------------------- |
| happy     | 爱心眼     | heart eyes             |
| smug      | 叉腰       | hands on hips          |
| angry     | 瞪眼       | glare                  |
| sad       | 羞涩       | bashful                |
| surprised | 抬手       | raised hand            |
| thinking  | 消除高光   | eye highlights removed |

Those translations are a guess from the names. Run `--sweep` to see what each
one actually does and rewrite the map to taste. The model also has six unnamed
`TriggerAnimation` hotkeys that nothing is bound to yet.

## Memory

She remembers you between sessions, two ways: automatically, from a background
extractor, and directly, when you tell her to. Everything lives in `memory.db`
next to the code; delete it and she forgets you completely.

Two tables. `turns` is the raw log of everything said. `facts` is what actually
reaches the prompt: short third-person statements about you, deduplicated and
reinforced when they come up again.

Every six replies, a background pass reads the turns she has not processed yet
and distils them into facts. It runs off the conversation thread and only
between turns -- it is a second generation on the same GPU, and running it
during a reply would stall her mid-sentence. It also runs once on exit, so the
tail of a session is not lost.

```
python scripts/memory.py           what she knows, strongest first
python scripts/memory.py log 20    the last 20 things said
python scripts/memory.py forget 3  delete one fact
python scripts/memory.py wipe      start over
```

Watch for `[remembered] ...` lines while you talk -- that is a consolidation
pass landing.

### Telling her directly

Consolidation is free to decide a sentence was not worth keeping, and it only
runs every six replies. That is the wrong shape for an instruction:

```
you:  remember that I hate mushrooms
iris: Got it. I'll remember that.
you:  forget about the mushrooms
iris: Forgotten.
```

[commands.py](iris/commands.py) recognises these directly and acts immediately,
without waiting for the extractor. The hard part is telling an instruction from
ordinary conversation -- "I remember going there" and "note taking is hard" are
not commands, and "forgetting the milk was my fault" is not a request to delete
something about milk. A trailing "?" is read differently in each direction:
"can you remember my name?" is almost always a real question and is left alone,
but "can you forget about the microwave?" is never a question about her
capabilities, and being unable to hear it once let a subject she had invented
run for twenty turns after being asked twice to drop it.

A bare "forget that" means the thing she just learned; a bare "forget it" is a
shrug and does nothing, since wiping memory because someone dropped a subject
would be an unpleasant surprise. What you say is rewritten to the third person
before it is stored -- "I hate mushrooms" becomes "The user hates mushrooms" --
so it reads like every other fact instead of a pasted quote.

### What the extractor gets wrong

**It invents entire people.** Given a two-line transcript that said only "my
name is Fred and I have a cat called Pixel", llama3.2 produced a complete
profile: a flat in downtown Portland, a career in freelance graphic design, an
art gallery website, hiking, craft beers, a weekly meeting with a business
partner. Every one of those was fabricated. A small model asked to produce a
list of facts will produce a list of facts whether or not it has any, and no
amount of prompting reliably stops it.

Two defences, both in [llm.py](iris/llm.py):

- **Grounding.** A fact is kept only if its content words actually appear in the
  transcript (`memory_grounding = 0.4`). Fabrications score near zero; real
  facts score high. On the real hallucinations above this rejected 8 out of 8
  while keeping 4 out of 4 genuine facts -- including "studying computer science
  at university" from a transcript that only said "uni".
- **A minimum.** Extraction needs `memory_min_turns` (3) user turns. Below that
  the turns stay pending and are picked up next session, so nothing is lost --
  short sessions are exactly where the invention happens.

**It attributes her own words to you.** "The user is not a nighttime creature"
came from one of her jokes, not from anything anyone said about themselves. And
**it turns questions into facts**: "is it late?" became "The user is not late."
Both are fixed by grounding against the user's own _statements_ only --
assistant turns and anything ending in a question mark are excluded from the
evidence, though they stay in the transcript so the model still has context.
Verified: a session of nothing but questions now learns nothing at all, while a
session of statements still learns correctly.

It also **records the absence of information as a fact** ("The user lives in a
location unknown.") and **says the same thing twice in different words**
("building a local voice AI called Iris" / "has a recurring plan to build a
voice AI"). Those get a phrase blocklist and word-overlap deduplication
(`memory_dedup = 0.6`) respectively.

If a wrong fact does land, `python scripts/memory.py forget <id>` removes it.
Check what she believes now and then.

Invented emotion tags are also stripped before anything is written to history or
memory. Feeding `[muffled growl of frustration]` back into the prompt teaches
her to produce more of them.

### What grounding cannot catch

Grounding asks whether a fact's words appear in the transcript. A **misheard**
transcript passes that test perfectly, because the words really are there --
the transcript is simply not what anyone said. This was a real stored fact:

> `The user is in a setting with no after programs.`
> from "it is not full of after programs that is to follow"

Nothing downstream of the microphone can question that. Only Whisper knows, and
it was the one thing it was never asked: `stt.py` computed `no_speech_prob` and
`avg_logprob`, collapsed them into "is this speech at all", and threw the
numbers away.

**So there are two bars now, and remembering is the higher one.** Answering a
half-heard sentence costs one bad reply; believing it costs every session after
it. `Heard.clear` is the second bar, `stt_clear_logprob` (-0.5 mean) and
`stt_clear_no_speech` (0.3 worst segment) set it, and only memory reads it --
she answers an unclear turn exactly as she would a clean one, because asking
someone to repeat themselves over a confidence score is worse company than
occasionally misunderstanding them.

An unclear turn still goes into the transcript, so the extractor reads it as
context. It is only barred from being the *evidence*, which is the same
treatment questions and her own replies already get. With the turn as evidence
the misheard fact grounds; with it as context only, it does not.

The number is printed in the `[stt ...]` line, because choosing a threshold for
your own microphone and room means watching what your clean speech actually
scores. On this machine `selftest.py` measures **-0.25** for a clean sentence,
so the bar at -0.5 has room under it.

Spoken `remember ...` commands are deliberately **not** gated. Those are an
instruction rather than an inference, she says back what she stored, and you are
right there to correct her.

### Intentions are not facts

> `The user plans to watch videos.`
> from "yeah I know, i am going to watch some videos now"

True for an evening, kept for good, holding a slot in a forty-fact store. An
intention said out loud is nearly always about the next hour, so `_EPHEMERAL`
drops facts that open with one -- "plans to", "is going to", "intends to",
"will".

It costs the occasional real one, and that is the trade. A standing intention
tends to get phrased as a state once it is underway -- "is building a voice AI",
"is studying computer science" -- and those pass untouched.

### What she brings up, and what falls away

Facts were ranked on `hits` alone. Two things were wrong with that.

Only about sixteen of forty facts fit `memory_char_budget`, so the ordering
decides what she actually knows -- and ranking on reinforcement filled that
budget with whatever happened to get echoed rather than whatever mattered. The
second was sharper: once forty facts had each been reinforced even once, a newly
learned fact arrived with a single hit, sorted below all of them, and was pruned
before it could ever come up again. **Nothing new could stick.** Demonstrated,
with a five-fact cap and five reinforced facts already stored:

| ranking           | does a fact learned today survive pruning? |
| ----------------- | ------------------------------------------ |
| `hits` alone      | no                                         |
| `hits` with decay | yes                                        |

The score is now `hits x 0.5 ^ (age / memory_half_life_days)`, age measured from
`updated`. That last detail is what makes it safe: `updated` is set every time a
fact is confirmed, so anything that keeps coming up never decays at all. Ordering
by it inverts a store like this one --

```
 1.81  x2  The user is called Paul
 0.98  x1  The user is building a voice AI
 0.80  x3  The user once talked about pizza
 0.40  x4  The user mentioned the weather twice
```

-- where `hits` alone put the weather first and the name third.

The cost is real and worth stating: something genuinely dormant for a season
does eventually fall out of a store capped at forty. `scripts/memory.py` prints
the score next to each fact, so what is on its way out is visible before it goes.

### Why the context window is only 2048

Memory needs prompt space, so the obvious move is a bigger context. It does not
work on 4 GB. Measured with `ollama ps`:

| num_ctx | model size | placement    | first phrase |
| ------- | ---------- | ------------ | ------------ |
| 2048    | 2.3 GB     | **100% GPU** | 0.29 s       |
| 3072    | 2.8 GB     | 15% CPU      | 0.40 s       |
| 4096    | 2.9 GB     | 20% CPU      | 0.37 s       |

Anything above 2048 pushes layers onto the CPU, and that costs more than the
extra context is worth. The room for memory came out of history instead:
`max_history_turns` dropped from 12 to 6, which is the right trade now that
continuity is carried by distilled facts rather than a verbatim transcript.

Measured prompt with a full memory block and six turns of history: **1818
tokens**, leaving 230 for a reply that is capped at 160.

`max_history_turns` bounds how many turns are kept, not how big they are, and
size is what overruns the window: six ordinary turns cost about 90 tokens, six
twenty-five-second rambles 319. With a full memory block that reached **2077 of
2048 before a single token was generated** -- and an overflow discards the front
of the prompt, which is the persona, rather than the stale turn at the back.

So the history has no budget of its own. `Brain._recent()` hands it whatever is
left of `num_ctx` once the system prompt is built and `reply_tokens` is
reserved. There is no tokenizer on this side of the API, so it is counted in
characters at 4.0 per token -- measured at 4.06 over the persona, 4.09 over the
situation block and 4.4 to 5.0 over speech, and the low end is used for both
halves of the subtraction, which is wrong in the safe direction twice. Six
ordinary turns survive intact alongside a full memory block; six rambles are cut
to the most recent five.

Raising `num_ctx` was measured rather than assumed, and rejected: with
`ollama ps`, 2048 is 3.2 GB at 27% CPU and 23.2 tok/s, 3072 is 3.4 GB at 31%
and 18.8 tok/s. A fifth of her generation speed, and more VRAM pressure on a
card she is usually sharing with whatever is in front.

If you want 4096 anyway, quantise the KV cache on the Ollama server rather than
buying it with CPU offload -- set `OLLAMA_FLASH_ATTENTION=1` and
`OLLAMA_KV_CACHE_TYPE=q8_0` in your environment and restart Ollama. Untested
here.

## Timers

The one genuinely useful thing she can do with no network, and the thing people
actually say out loud to a voice.

```
you:  set a timer for ten minutes
iris: Fine. 10 minutes.
you:  remind me in twenty minutes to take the pizza out
iris: Right. 20 minutes. Take the pizza out.
      ...
iris: That's your 20 minutes. Take the pizza out.
```

"How long is left on the timer" and "cancel the timer" work too. `--no-timers`
switches the whole thing off.

Durations are parsed rather than pattern-matched a phrase at a time: every
quantity in the string is summed, so "one hour thirty minutes" needs no rule of
its own. Two shapes do need one. A plural fraction with its own count --
"three quarters of an hour" -- reads as 3.25 taken word by word, and the words
it consumes have to be blanked out afterwards or the same string is _also_
counted as a whole "an hour". And a bare fraction with no unit at all -- "an
hour and a half" -- means the unit in front of it.

**A command needs a duration or it is not a command.** "Set a timer" and "timer
for the pasta" fall through to the brain untouched. Without that rule she would
have to hold a half-finished command open across turns, and a voice that is
silently waiting for the rest of your sentence is worse than one that just
answers you.

Two orderings matter, and both are load-bearing:

- Timers are checked **before** the memory commands. "Forget the timer" is a
  cancellation, and the memory grammar reads it as an instruction and starts
  deleting facts about timers.
- Firing waits for her own sentence and for yours to finish, but is **not**
  gated on the tray mute or on paused idle chatter. Those switch off her ears
  and her chatter; neither is a reason to swallow something she was explicitly
  asked to say.

What she says is a fixed line, not a generated one, for the reason the memory
acknowledgements are fixed: a timer is a utility, one that waffles for two
sentences before saying what it was for is worse than one that does not, and a
generation that fails at the moment the timer fires is the single failure this
feature cannot afford. `spoken()` is also deliberately not `humanize()` -- a
timer set for an hour is "an hour", not "about an hour". Rounding is right for
"how long have we been talking" and wrong for the one thing here whose whole
point is that it is exact.

Timers live in memory only. A countdown does not survive a restart, because
restoring one that spent most of its span in a process that was not running is
a worse answer than losing it. Past `timer_max_s` (twelve hours) she declines
rather than promising something she will not be alive to do.

## Doing things

She floats above this desktop all day and `screen.py` already tells her what is
on it. This is the other half of that: the four things people say out loud to
something sitting on their machine.

```
you:  turn the volume down
iris: 60.
you:  a bit more
iris: 50.
you:  actually set it to 70
iris: 70.
you:  dim the screen
iris: 85.
you:  skip this song
iris: Next.
you:  open spotify
iris: Opening spotify.
```

The volume, the brightness, whatever owns the media session, and anything with
a Start Menu shortcut. `--no-actions` switches the whole thing off.

### It is a grammar, not a plan

This is ported from an assistant that does it the other way round: obvious
commands hit a deterministic router, and anything ambiguous goes to a second
model call that returns a JSON plan, which is then validated and executed. The
router came across. The planner deliberately did not.

Three reasons, and they are the same three `timers.py` gives for answering "set
a timer for ten minutes" with a fixed line instead of a generated one:

- **The context window is 2048**, for the reasons in its own section above. A
  tool guide describing thirty tools does not fit beside the persona, the
  memories and what is on screen, and everything it would displace is what
  makes her her.
- **She runs at temperature 0.9.** That is tuned for a personality that
  wanders, and it is the worst setting there is for emitting valid JSON.
- **It would be a second generation inside one turn.** Spending a whole model
  run to work out that "pause" means pause costs more than the pause is worth,
  and it is the exact expense the timer acknowledgements exist to avoid.

So it is regex, wrapped in the same `POLITE`/`TRAILING` pair the timer and
memory grammars use -- which is why "okay Iris, could you turn the volume down
please" and "volume down" are one pattern rather than two. Anything it does not
recognise falls through to the brain untouched, which is the right answer to
"can you turn my life around" rather than a gap in the feature.

### A remark is not an instruction

The hard part is not recognising commands. It is *not* recognising everything
else, because a false positive here is not a bad sentence -- it is her reaching
over and changing the machine because of something you said about it.

"The screen is at fifty percent" contains a brightness noun, a number and a
preposition, and shares every content word with "set the screen to fifty
percent". "The volume is at maximum" is a statement about a knob that reads
exactly like an order to move it.

What separates them is that an instruction *opens* like one. A single gate at
the top of the knob grammar requires the sentence to begin with a verb that
does something, a bare comparative ("louder"), or a noun aimed straight at a
value ("volume to 30"). It is the same trick the memory grammar uses to leave
"I remember going there" alone, and it does most of the work in this file.

One word is deliberately not a command. Bare **"stop"**, said out loud to
something that is talking, means stop talking -- so stopping the music has to
say "the music".

Bare **"mute"** was the second, on the reasoning that it might mean her
microphone, and that was wrong. Held back, it produced this:

```
you:  Mute.
iris: Volume at zero.
you:  Mute.
iris: Mute is already active.
you:  No, it's not
iris: Volume is at 57. Muting would bring it to zero. You said "no", so it
      remains at 57.
you:  Mute it now.
iris: Volume at 57. Mute command issued. Final state: zero. Done.
```

Nothing was muted at any point. The microphone reading it was being protected
from does not exist as a code path -- her mic is a tray switch and a hotkey,
never a spoken command -- so there was no second interpretation to be ambiguous
with, only a first one being declined. And a grammar that declines a real
instruction does not leave a polite silence behind it. It hands the sentence to
the one component in this process that will confidently make an answer up,
which is the exact failure the whole file exists to prevent.

So bare "mute" and "unmute" are commands now, and "mute your mic" is the one
phrasing still left alone -- she genuinely cannot do that, and muting the
speakers instead would be another confident wrong answer.

The same turn showed up a second gap. `TRAILING`, inherited from the memory
grammar, absorbs a trailing "please" and nothing else, because nobody has ever
said "remember that now". Instructions to *do* something carry the adverb
constantly -- "mute it now", "open spotify now" -- and in the launch pattern it
was worse than a miss: the lazy name group swallowed it and went looking for an
application called "spotify now".

### She has to say it before she does it

Muting the system mutes *her*. Kokoro comes out of the same speakers, so
applying the mute and then speaking the acknowledgement means the
acknowledgement is never heard: she appears to ignore you, and the machine goes
quiet at the same moment, which is exactly as unsettling as it sounds.

So `Desk.do()` returns a line and, for that one case, something to run once the
line has been *played*. `_say()` already blocks until the speakers drain, so
"Muted." is out in the room before anything silences the room. Setting the
volume to zero takes the same path, being the same thing under another name.
Turning it up while muted unmutes, because "turn it up" has never meant
anything else.

### Only things that already exist can be opened

Every word reaching this file came from a microphone by way of a transcriber
whose job is to guess. Handing an unresolved guess to `os.startfile` or `Popen`
would let a mishearing name a path or a binary, so the launch grammar resolves
against an index -- the Start Menu, the desktop, and a small curated table for
the applications nobody calls by their executable name -- and returns nothing
at all when the name is not in it. There is no fallback past the index, and
"open the pod bay doors" gets a personality answer instead of an error.

The two kinds of index entry rank opposite ways round, which took a bug to
notice. A protocol handler beats a shortcut: `spotify:` reaches the installed
application however it was installed, where a Start Menu entry of that name may
point at a stale updater. A bare executable name loses to one: `discord` is on
PATH only if you are lucky, and a shortcut actually found on this machine beats
a guess about this machine.

### What it costs

| stage                                   | measured |
| --------------------------------------- | -------- |
| parse, on every utterance and usually a miss | 0.06 ms |
| volume, via pycaw                       | under 1 ms |
| media key, via user32                   | under 1 ms |
| brightness, via WMI                     | 1.1 s    |

Three of those are free. Brightness is not, and cannot be: there is no ctypes
call for panel brightness the way there is for volume, so it is a WMI method,
and reaching WMI means starting a PowerShell -- 0.85 s on this machine before
it has done anything at all. Reading the current level and writing the new one
as two subprocesses cost 2.4 s; folding both into one script that reads, adds
and writes brought it to 1.1 s, and there is no further to go without a
different interface.

It is also the one that silently did nothing. A failed WMI method call is a
*non-terminating* error: PowerShell prints it to stderr and then exits 0, so
the first version of this checked the return code, saw success, and said
"eighty" in a pleasant voice while changing absolutely nothing.
`$ErrorActionPreference` turns that into an exit code, and `do()` turns an exit
code into her admitting it -- which is the entire point, because a companion
that claims to have done something is worse than one that says it cannot.

## Hearing names she has learned

`faster-whisper` accepts `hotwords`, and memory already stores the words a
generic English model gets wrong -- your name, your pets, your projects. The
fact table is fed straight back into the transcriber, so the longer she knows
you the better she hears you.

Measured A/B on the same audio:

|       | without                                          | with                                   |
| ----- | ------------------------------------------------ | -------------------------------------- |
| clean | "I study at Strathmore University"               | unchanged                              |
| noisy | "I study at **Straffnell** University"           | "I study at **Strathmore** University" |
| noisy | "Read here, talking to **IMS** about JavaScript" | "talking to **Iris** about JavaScript" |

On clean audio it changes nothing. On noisy audio it recovers the names, though
it does not fix general noise robustness -- one test line stayed garbled either
way. No downside, and it costs nothing.

## Her voice can be cloned (currently off)

Kokoro cannot be taught a new voice. Its voicepacks are style vectors from a
training pipeline that was never released, and the only custom voice it supports
natively is a blend of the 54 it ships with. So the timbre is changed _after_
synthesis: Kokoro supplies the words and the prosody, and the Kanade zero-shot
voice-conversion model re-voices the result to match a reference recording.

`clone` is **off** by default: she speaks as Kokoro's `af_heart`. Set it to
`True` to use the cloned voice instead. The reference is
`samples/iris_reference.wav` -- any 3-10 second clip of the target voice works,
and `clone_reference` points at it.

Measured both ways, same machine, idle GPU. The uncloned column was taken on
`af_bella`, and holds for any voicepack: they are all the same 82M model with a
different style vector, so the timings and the memory do not move between them.

|                      | kokoro        | cloned  |
| -------------------- | ------------- | ------- |
| TTS real-time factor | **0.50-0.56** | 0.76    |
| opening phrase       | **0.57 s**    | ~1.3 s  |
| startup              | **2.2 s**     | ~18 s   |
| RAM                  | **~0.8 GB**   | ~2.2 GB |

The torch and Kanade dependencies stay installed either way, so flipping `clone`
back to `True` costs nothing but the load time.

### The one optimisation that makes it affordable

`KanadeModel.voice_conversion()` re-encodes the reference on every single call,
which on a 9.8 s reference is 1.12 s of wasted work per phrase. It is only a
wrapper over encode+decode, so [cloning.py](iris/cloning.py) encodes the
reference **once** at startup and reuses the embedding:

| phrase                | as shipped | reference cached |
| --------------------- | ---------- | ---------------- |
| "Oh, absolutely not." | 1.59 s     | **0.48 s**       |
| 3.3 s of speech       | 2.01 s     | 0.88 s           |
| 6.0 s of speech       | 3.19 s     | 1.81 s           |

Conversion runs at RTF 0.29 and Kokoro at 0.43, so together 0.76 -- still under
1.0, which is what matters: she does not fall behind mid-reply once she has
started talking.

### What it costs

Measured, with cloning on versus off:

|                     | Kokoro only | cloned      |
| ------------------- | ----------- | ----------- |
| time to first audio | ~1.4 s      | **~2.2 s**  |
| RAM                 | ~0.8 GB     | **~2.2 GB** |
| startup             | ~9 s        | ~18 s       |

That is the trade: about 0.8 s of extra latency and 1.4 GB of RAM for a voice
that is not one of Kokoro's 54. `clone = False` reverses it.

### Two things about the model

**It is stochastic.** The same sentence converts slightly differently every
time -- running identical code twice gives waveforms differing by up to 0.68
peak. That is the decoder, not a bug. Quality varies a little between takes.

**It can fail without warning.** A conversion error returns her Kokoro voice for
that phrase rather than dropping it or crashing the loop: a hole in the middle
of a sentence is worse than a change of timbre. The first three failures print.

### Choosing her voice

With cloning on, `tts_voice` still shapes accent and delivery but no longer
decides the timbre. Turn cloning off before auditioning. Kokoro ships 54 voices,
28 of them English:

```
python scripts/voices.py                  every English voice, one file each
python scripts/voices.py --female         just the American female ones
python scripts/voices.py --tune af_heart  speed and pause variants of one
python scripts/voices.py --say "..."      audition on your own line
```

Files land in `samples/voices/` and `samples/tune/`. Play them, then set
`tts_voice` in [config.py](iris/config.py).

The samples are synthesised **phrase by phrase and joined**, exactly as the live
loop does it. Auditioning a whole sentence in one call flatters the result and
is not what you will hear.

Two settings matter as much as the voice:

- `tts_speed` is 1.0, the natural rate. It used to be 1.1, chosen to shave
  latency, and a hurried delivery is most of what makes a synthetic voice sound
  synthetic. Going back to 1.0 costs about 0.2 s on the opening phrase.
- `tts_pause_ms` (60) is silence appended to each phrase. Joined clips run
  0.32 s shorter than the same sentence spoken in one go, and that missing time
  is breath. Kokoro already leaves ~40 ms of lead and ~85 ms of tail, so a
  little goes a long way -- too much sounds sedated.

### Her personality

[persona.txt](persona.txt) is the whole character, reloaded on every start. The
one line doing real latency work is the instruction to open with a two-to-five
word reaction — that gives the chunker an early comma to cut on, so she starts
speaking before the model has finished thinking.

Emotion cues like `[smug]` are stripped before synthesis by `clean_for_speech`,
and the first one reaches `vts.express()`, which sets that expression on the
Live2D model -- see [The avatar](#the-avatar). The persona asks for a cue on
every line rather than allowing one, because only an explicit cue fires an
expression change: a line without one leaves whatever was on her face there.

Asking is not enough on its own. Measured over five ten-turn conversations,
**17 replies in 50 still arrived with no cue at all** -- history keeps the
spoken words only, so she reads a dozen of her own untagged lines and copies
them. `Iris._say` therefore falls back to `neutral` when a reply's *opening*
phrase carries none. Only the opening one: a later phrase sending neutral would
clear her face mid-sentence, which is the same mistake as firing the cue at
synthesis time instead of at playback.

**The menu is four cues, not seven.** It was `[neutral] [happy] [smug] [sad]
[angry] [surprised] [thinking]`, and the three that were dropped are the three
this rig has no expression for anyway. Fewer options is worth real accuracy: no
cue at all fell from 38% of replies to 26%, and the share carrying a cue the rig
can actually show rose from 38% to 56%, over 50 replies per arm.

**`[smug]` never fires, and that is the model.** Not once in 160 sampled
replies. Not with concrete triggers spelled out -- a promise they broke, a
question they could answer themselves, the third hour on the same file -- and
not with a persona that called it her ordinary cue rather than a rare one. It
labels those lines `[thinking]` instead. A 4B model appears not to recognise its
own tone as smug, and no wording tested moves it, so `cat_pos` in `vtube_map` is
mapped to something that will not arrive. It is left mapped because it costs
nothing and a better model would use it.

Putting the cue back into history was measured as the alternative and rejected.
It does fix the omission outright, 17 in 50 to none, and the invented tags the
stripping guards against never reappeared -- `clean_for_speech` filters to the
allowed set, so "[muffled growl]" cannot get back in. But `[angry]` went from
never to 12 replies in 50, and stayed there whether the loud cues were held out
of history or rewritten to neutral first; a visibly live cue slot is apparently
enough on its own to make a 4B model exercise the whole range. Replies also got
longer, 2.8 sentences to 3.0. The fallback costs none of that.

### The last line of the prompt

The persona is not the last thing she reads, and that turns out to decide
whether the shortest rule in it is obeyed. `CLOSING` in [iris/llm.py](iris/llm.py)
restates the two-sentence rule as the final block, after the situation and the
memory. persona.txt has carried that rule from the beginning -- twelve lines
in, with six hundred tokens of context read after it.

Measured over five eight-turn conversations, she broke it on **38 turns out of
40**, averaging 4.5 sentences and getting longer as the conversation went on.
With the same instruction closing the prompt: **15 of 40**, averaging 2.7. The
replies that invented a warm screen or a dark room went with the tail she no
longer had room to write, five occurrences to none.

Position is doing the work here, not wording. A longer closing line that also
forbade reading the clock out was measured and dropped: three volunteered times
in thirty-five turns against four, which is nothing, and it cost length.

### The prose is load-bearing

It does not look it. Tightening persona.txt from 1434 tokens to 1251 -- writing
the same rules in fewer words, removing none of them -- broke three things at
once:

| | before the trim | after |
| --- | --- | --- |
| mean reply length | 2.6 sentences | 3.6 |
| "can you see my screen" answered properly | 6/6 | 0/3 — "Yes." |
| emitted a chemical formula when pushed | 0/6 | 3/3 |

That last one is `CO2 + H2O -> C6H12O6 + O2`, the exact string the formula rule
exists to prevent, and it came back the moment the rule lost its closing "Say it
in words or do not say it". Restoring the file restored all three. If the
persona has to shrink, measure each cut -- the redundant-looking sentences are
carrying the ones next to them.

### Examples are a menu

Three examples of a flat reaction -- `"...Cool." "Whatever." "If you say so."` --
read to a 4B model as a list to choose from, and it chooses one. In the logged
conversations from before the closing line existed, "Whatever." closed a quarter
of her replies and was by a wide margin the most repeated thing she said.

Most of that was the missing closing line rather than the examples. The tic
lived in a trailing third sentence, and once there was no room for one it fell
from **25% of replies to 7%**, measured over 72 replies weighted towards the
silences and flat reactions that produce it.

The remainder was the menu. Replacing the three quoted strings with a
description of the register, plus an instruction never to reach for the same one
twice in a row, took it to **1 in 72** -- and raised the variety of her short
phrases from 75 distinct to 82, so it did not simply flatten her. Keeping the
quotes and adding the same instruction made it slightly worse: the strings are
what she copies, not the sentence around them.

### Measuring it

That regression went unnoticed for two runs because the samples were being read
and thrown away. `scripts/behaviour.py` keeps them:

```
python scripts/behaviour.py                     three conversations
python scripts/behaviour.py --json before.json  save a run
python scripts/behaviour.py --compare before.json
python scripts/behaviour.py --persona try.txt   measure a variant
```

It drives the real `Brain` -- `_system()` builds the prompt, `_recent()` trims
the history, `reply()` and `unprompted()` produce the text -- so it cannot drift
away from what ships. The clock is pinned to a fixed afternoon, since she is
written to drift with the hour and two runs at different times are not
comparable. The fixture window is titled `motion.py - Visual Studio Code`, which
is the only way to separate naming the application (asked for) from reading the
title out word for word (forbidden); with a game called ELDEN RING those are the
same string.

It reports length, missing cues and which cues never fire, her most repeated
stock phrase, the two prohibitions, both hand-tuned carve-outs, and the context
budget. Everything but the budget is a rate sampled at temperature 0.9: a couple
of points is noise, half is not. The budget is arithmetic, and it is the only
thing here that exits non-zero.

Re-running the trim above against a saved baseline prints what two runs of
reading samples had missed:

```
    mean sentences per reply   3.75   (WAS 2.29)
    replies over two           17/24  (71%)   (WAS 7)
    formula                    2/3 correct   (was 3)
```

## The brain

`qwen3:4b-instruct-2507-q4_K_M`. Measured against llama3.2 on an idle GPU, same
persona, same memory, same six prompts, with the cloned voice in the loop:

|                     | qwen3-instruct     | llama3.2             |
| ------------------- | ------------------ | -------------------- |
| placement           | 3.2 GB, 27% on CPU | 2.3 GB, **100% GPU** |
| generation          | 23.4 tok/s         | **75.4 tok/s**       |
| llm to first phrase | 0.56 s             | **0.17 s**           |
| to first audio      | **1.60 s**         | 1.96 s               |
| full reply          | 1.47 s             | **0.41 s**           |
| reasoning aloud     | 0/6                | 0/6                  |

qwen3 is three times slower per token and still reaches audio no later, because
with cloning on the opening phrase has to be synthesised _and_ converted before
she makes a sound -- so how short her opener is matters more than how fast the
tokens arrive. The gap is inside the noise either way; treat them as comparable
on latency and pick on writing. Swap with `run.py --model llama3.2`.

### Do not use plain `qwen3:4b`

The default tag is a reasoning model and cannot be made to work here. Every
lever was tried:

| attempt                          | result                                                             |
| -------------------------------- | ------------------------------------------------------------------ |
| `think=True`                     | zero speakable output; the whole budget goes in the thinking block |
| `think=False`                    | the reasoning moves into the **content** channel instead           |
| `/no_think` in the system prompt | no effect                                                          |
| `/no_think` in the user turn     | it reasons about the token itself                                  |
| `num_predict` 200 / 600 / 1200   | 9 s / 33 s / **92 s**, still reasoning                             |

At 1200 tokens it was still counting words in candidate replies and had produced
nothing to say. It also recites the persona back verbatim, averaging 904
characters against llama3.2's 51 -- all of which would be spoken aloud.
`qwen3:4b-instruct-2507` is a different model that only has a non-thinking mode,
which is why it works.

`Brain._resolve_thinking()` asks Ollama for the model's capabilities and sets
`think=False` for anything reporting `thinking`, so the silent zero-output
failure cannot happen again. The flag stays unset for models without the
feature, which reject it.

### Exact tags must match exactly

Model resolution once matched on base name, so `qwen3:4b-instruct-2507-q4_K_M`
silently ran as `qwen3:4b` -- a different model with different behaviour --
because both begin "qwen3". A tagged name now only ever matches itself; only a
bare name like `llama3.2` may resolve by base, to `llama3.2:latest`.

An 8B model at Q4 needs ~5 GB and is not worth considering on this card.

### One bad turn is a bad turn, not the end

The avatar has never been allowed to break a conversation. The brain was, and it
is the far likelier thing to go: ollama being restarted, evicting the model, or
dropping a stream mid-reply took the whole process down with a traceback -- and
with it Whisper and Kokoro, which cost about eight seconds to load again.

A failed turn is now reported and the loop carries on, the same way she
reconnects when VTube Studio comes back. Recovery also reopens the microphone,
because a failure between muting it and unmuting it would otherwise leave her
deaf for the rest of the session, which looks exactly like a crash and is harder
to diagnose than one.

Transcription has a ceiling (`stt_timeout_s`, 30 s) for a narrower reason. The
pool has a single worker and no way to cancel what it is running, so a wedged
call cannot be recovered from -- but it can say so instead of hanging the loop
in silence with nothing to diagnose it from.

## Layout

```
run.py              CLI entry point
persona.txt         who she is
iris/config.py      every tunable knob
iris/awareness.py   the time, the day, how long you have been talking
iris/screen.py      which application is in front, and for how long
iris/vad.py         streaming Silero wrapper (stateful, unlike the bundled one)
iris/mic.py         capture -> utterances, with pre-roll, hangover, speculation
iris/stt.py         faster-whisper, the hallucination filter, and the two bars
iris/llm.py         Ollama streaming, phrase chunker, emotion tags, extraction
iris/commands.py    spoken "remember"/"forget" grammar, shared with timers.py
iris/timers.py      "remind me in ten minutes", spoken durations
iris/actions.py     the volume, the brightness, the music, opening things
iris/memory.py      SQLite facts and turn log, deduplication
iris/tts.py         Kokoro
iris/cloning.py     Kanade voice conversion, reference encoded once
iris/player.py      output queue with instant barge-in
iris/motion.py      idle head and body, and the gaze model below
iris/vtube.py       VTube Studio client: expressions, lip sync, parameter injection
iris/overlay.py     restyles the VTube Studio window so she floats on the desktop
iris/tray.py        tray icon, global hotkeys, push-to-talk key listener
iris/app.py         the conversation loop
scripts/selftest.py every stage, no microphone needed
scripts/latency.py  the timing table above
scripts/behaviour.py what she says, measured -- run it after editing persona.txt
scripts/memory.py   inspect, edit and wipe what she knows
```

## Not done yet

- **Memory is only as good as a 3B extractor.** She records what you tell her
  directly and misses what you imply. A larger model would do better; so would
  a second extraction pass that revises old facts rather than only adding new
  ones and letting the weak ones decay.
- **She still cannot be interrupted on speakers.** `barge_in` needs headphones,
  because her own voice re-enters the microphone and she interrupts herself on
  her first word. Cancelling the echo would need her output and the mic input
  compared, which is a real filter and a real piece of work.
- **There is no wake word.** `--push-to-talk` covers the case where somebody
  else is in the room, but it wants a hand free. A wake word would not, at the
  cost of a model running on every frame.
- **A window changing is a better reason to speak up than a timer.** The
  watcher already knows when it happens; idle chatter still runs on a clock.
