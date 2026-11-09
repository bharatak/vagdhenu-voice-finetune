# Recording guide (one page)

The model learns **everything** in your recordings: your voice, but also the room, the distance,
the pauses. Keeping these the same every day is what makes the result sound like you.

## Setup (same every day)
- **iPhone Voice Memos:** Recording mode **Mono**, Audio Quality **Lossless**.
- iPhone on a stand, **~20 cm** from your mouth, slightly below it. Same room, same spot, same distance.
- Fan and AC **off**. A room with curtains, a bed or cushions has less echo.
- *Optional:* a tanpura/śruti drone in **headphones only**, never audible in the recording. It keeps the pitch
  from drifting. Not needed: chanting relaxed in your usual puja rhythm matters more.
- **Warm up first:** chant a śloka or two *before* pressing record, so the recording starts settled.
- Open the day's script: `python -m mv.script_page <sarga> --open`.

## While chanting
1. Start recording, stay **silent for 3 seconds**, then begin.
2. Chant **only the lines on screen**, in order. No saṅkalpa, colophon or talking mid-file.
3. Each pāda in one breath: **no gaps between words**. Pause **only at daṇḍas**:
   short at `।`, a little longer at `॥`.
4. Skip greyed-out **⚠ skip** verses. Their source text is damaged.
5. **If you stumble, stay silent for 3 seconds, then restart that line.** Don't say "sorry" or go back further.
6. Natural pārāyaṇa pace and pitch. Hold long vowels, sound visargas fully, keep retroflexes crisp.

## After
- One sarga per file. AirDrop it to the Mac and rename it to `myvoice/inbox/sarga_NN.m4a`
  (for example `sarga_05.m4a`).
- If you started partway through, add the starting verse: `sarga_05_v40.m4a`
  (and use `--start 40` for the script).
- If you see a text error on screen, note it. Corrections go in `data/text/fixes.json`.
