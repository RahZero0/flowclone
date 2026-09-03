# FlowClone

Minimal local Wispr Flow clone for macOS. Hold **Right ⌥ (Option)**, speak, release — the transcript pastes into whatever app you're typing in. Only UI is a menu-bar icon: 🎤 idle, 🔴 recording, ⏳ transcribing, ⚠️ error.

All transcription is local (whisper.cpp `small.en`, Metal-accelerated). Nothing leaves your Mac.

## Setup

```bash
cd flowclone
python3 -m venv .venv && source .venv/bin/activate   # or: uv venv && uv pip install -r requirements.txt
pip install -r requirements.txt
python flowclone.py
```

First run downloads the ggml model (~466MB) to `~/Library/Application Support/pywhispercpp/models/`.

### Permissions (one-time)

Add your terminal (and/or `.venv/bin/python3`) to BOTH lists, then fully restart the terminal:

- **System Settings → Privacy & Security → Accessibility**
- **System Settings → Privacy & Security → Input Monitoring**
- **Microphone** — macOS prompts on first recording

## Usage

Hold Right Option → speak → release. Text pastes at your cursor.

Menu bar (🎤):

- **Status** — live state / last error
- **Hot load model** — on: model resident in RAM (~500MB), instant. Off: loads per dictation (~0 idle RAM, +1–2s).
- **Paste last transcript**
- **Test paste** — click it, then click a text field; verifies the paste path alone

## Troubleshooting

- Logs: `~/.flowclone.log` (also printed in the terminal).
- Hotkey does nothing → set `"debug_keys": true` in `~/.flowclone.json`, press your key, read its name from the log, put it in `"hold_key"`.
- "audio is silent" → check mic selection in System Settings → Sound → Input, and mic permission.

## Config (`~/.flowclone.json`)

`model`: `base.en` (lighter, ~200MB) or `small.en-q5_1` (quantized). `hold_key`: any pynput key name (`cmd_r`, `f13`, ...).

## RAM footprint

| Mode | Idle | During dictation |
|---|---|---|
| Hot load, small.en | ~600MB | ~700MB |
| On demand, small.en | ~80MB | ~700MB |
| On demand, base.en | ~80MB | ~350MB |
