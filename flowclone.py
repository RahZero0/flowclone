#!/usr/bin/env python3
"""FlowClone — minimal local Wispr Flow clone for macOS.

Hold RIGHT OPTION -> speak -> release -> transcript is pasted into the
focused app. Menu-bar icon is the only UI:
  🎤 idle   🔴 recording   ⏳ transcribing   ⚠️ error (see menu)

STT: whisper.cpp via pywhispercpp (Metal-accelerated on Apple Silicon).
Logs: ~/.flowclone.log (also printed to the terminal).
"""

import gc
import json
import logging
import os
import subprocess
import threading
import time

import numpy as np
import rumps
import sounddevice as sd
from pynput import keyboard
from pynput.keyboard import Controller, Key

CONFIG_PATH = os.path.expanduser("~/.flowclone.json")
LOG_PATH = os.path.expanduser("~/.flowclone.log")
DEFAULTS = {
    "model": "small.en",  # any pywhispercpp/ggml name: base.en, small.en, small.en-q5_1 ...
    "hot_load": True,  # keep model in RAM
    "hold_key": "alt_r",  # right Option; e.g. "cmd_r", "f13", "ctrl_r"
    "sample_rate": 16000,
    "restore_clipboard": True,
    "debug_keys": False,  # log every key press (to find your key's name)
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_PATH)],
)
log = logging.getLogger("flowclone")


def load_config():
    cfg = dict(DEFAULTS)
    try:
        with open(CONFIG_PATH) as f:
            cfg.update(json.load(f))
    except (OSError, ValueError):
        pass
    return cfg


def save_config(cfg):
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(cfg, f, indent=2)
    except OSError as e:
        log.error("could not save config: %s", e)


def pbpaste():
    try:
        return subprocess.run(["pbpaste"], capture_output=True, text=True).stdout
    except OSError:
        return ""


def pbcopy(text):
    subprocess.run(["pbcopy"], input=text, text=True)


class FlowClone(rumps.App):
    def __init__(self):
        super().__init__("🎤", quit_button=None)
        self.cfg = load_config()
        self.model = None
        self.model_lock = threading.Lock()
        self.recording = False
        self.chunks = []
        self.chunks_lock = threading.Lock()
        self.stream = None
        self.last_transcript = ""
        self.last_error = ""
        self.kbd = Controller()  # for sending Cmd+V

        self.status_item = rumps.MenuItem("Status: starting…")
        self.hot_item = rumps.MenuItem(
            "Hot load model (uses RAM)", callback=self.toggle_hot
        )
        self.hot_item.state = self.cfg["hot_load"]
        self.menu = [
            self.status_item,
            rumps.MenuItem(f"Hold {self.cfg['hold_key']} to dictate"),
            rumps.MenuItem(f"Model: {self.cfg['model']}"),
            None,
            self.hot_item,
            rumps.MenuItem("Paste last transcript", callback=self.paste_last),
            rumps.MenuItem(
                "Test paste (types 'FlowClone works!')", callback=self.test_paste
            ),
            None,
            rumps.MenuItem("Quit", callback=self.quit_app),
        ]

        if self.cfg["hot_load"]:
            threading.Thread(target=self._preload, daemon=True).start()
        else:
            self.set_status("idle (model loads on demand)")

        self.listener = keyboard.Listener(
            on_press=self.on_press, on_release=self.on_release
        )
        self.listener.start()
        log.info("started; hold key = %s", self.cfg["hold_key"])

    # ---- status / errors ----
    def set_status(self, text):
        self.status_item.title = f"Status: {text}"

    def fail(self, msg, exc=None):
        self.last_error = msg
        log.error("%s%s", msg, f": {exc}" if exc else "", exc_info=bool(exc))
        self.title = "⚠️"
        self.set_status(f"ERROR — {msg}")
        try:
            rumps.notification("FlowClone", "Error", msg)
        except Exception:
            pass

    # ---- model ----
    def _preload(self):
        try:
            self.set_status("loading model…")
            self.ensure_model()
            self.set_status("ready — hold key and speak")
        except Exception as e:
            self.fail("model load failed (see ~/.flowclone.log)", e)

    def ensure_model(self):
        with self.model_lock:
            if self.model is None:
                from pywhispercpp.model import Model

                log.info("loading model %s …", self.cfg["model"])
                t0 = time.time()
                # kwargs vary across pywhispercpp versions; degrade gracefully
                for kwargs in (
                    {"n_threads": 4, "print_progress": False, "print_realtime": False},
                    {"n_threads": 4},
                    {},
                ):
                    try:
                        self.model = Model(self.cfg["model"], **kwargs)
                        break
                    except TypeError:
                        continue
                if self.model is None:
                    raise RuntimeError("could not construct pywhispercpp Model")
                log.info("model loaded in %.1fs", time.time() - t0)
            return self.model

    def release_model(self):
        with self.model_lock:
            self.model = None
        gc.collect()
        log.info("model released")

    # ---- hotkey ----
    def key_name(self, key):
        if isinstance(key, Key):
            return key.name
        return getattr(key, "char", None)

    def on_press(self, key):
        name = self.key_name(key)
        if self.cfg.get("debug_keys"):
            log.info("key pressed: %r (name=%s)", key, name)
        if name == self.cfg["hold_key"] and not self.recording:
            self.start_recording()

    def on_release(self, key):
        if self.key_name(key) == self.cfg["hold_key"] and self.recording:
            self.stop_and_transcribe()

    # ---- audio ----
    def start_recording(self):
        with self.chunks_lock:
            self.chunks = []
        sr = self.cfg["sample_rate"]

        def cb(indata, frames, t, status):
            if status:
                log.warning("audio status: %s", status)
            with self.chunks_lock:
                self.chunks.append(indata.copy())

        try:
            self.stream = sd.InputStream(
                samplerate=sr, channels=1, dtype="float32", callback=cb
            )
            self.stream.start()
            self.recording = True
            self.title = "🔴"
            log.info("recording started")
        except Exception as e:
            self.fail("microphone open failed — check mic permission", e)

    def stop_and_transcribe(self):
        self.recording = False
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
        except Exception as e:
            log.warning("stream close: %s", e)
        self.stream = None
        self.title = "⏳"
        threading.Thread(target=self._transcribe_worker, daemon=True).start()

    def _transcribe_worker(self):
        try:
            with self.chunks_lock:
                chunks, self.chunks = self.chunks, []
            if not chunks:
                log.info("no audio captured")
                return
            audio = np.concatenate(chunks).flatten().astype(np.float32)
            dur = len(audio) / self.cfg["sample_rate"]
            log.info(
                "captured %.2fs of audio (peak %.3f)", dur, float(np.abs(audio).max())
            )
            if dur < 0.3:
                return
            if float(np.abs(audio).max()) < 0.001:
                self.fail("audio is silent — is the right mic selected / permitted?")
                return

            self.set_status("transcribing…")
            model = self.ensure_model()
            segments = model.transcribe(audio)
            text = " ".join(s.text.strip() for s in segments).strip()
            log.info("raw transcript: %r", text)

            if not self.cfg["hot_load"]:
                self.release_model()

            text = text.strip()
            if not text:
                self.set_status("heard nothing intelligible")
                return

            self.last_transcript = text
            self.paste_text(text)
            self.set_status(
                f"pasted: {text[:40]}…" if len(text) > 40 else f"pasted: {text}"
            )
        except Exception as e:
            self.fail("transcription failed (see ~/.flowclone.log)", e)
        finally:
            if self.title != "⚠️":
                self.title = "🎤"

    # ---- paste ----
    def paste_text(self, text):
        old = pbpaste() if self.cfg["restore_clipboard"] else None
        pbcopy(text)
        time.sleep(0.15)  # let the pasteboard settle
        try:
            # pynput synthetic Cmd+V — uses the same Accessibility trust as the listener
            with self.kbd.pressed(Key.cmd):
                self.kbd.press("v")
                self.kbd.release("v")
            log.info("pasted %d chars via Cmd+V", len(text))
        except Exception as e:
            log.warning("pynput paste failed (%s), trying osascript", e)
            r = subprocess.run(
                [
                    "osascript",
                    "-e",
                    'tell application "System Events" to keystroke "v" using command down',
                ],
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                self.fail(
                    f"paste failed — grant Automation/Accessibility ({r.stderr.strip()})"
                )
        if old is not None:
            time.sleep(0.5)
            pbcopy(old)

    def paste_last(self, _):
        if self.last_transcript:
            self.paste_text(self.last_transcript)

    def test_paste(self, _):
        # Click a text field within 3 seconds, then it types.
        def worker():
            time.sleep(3)
            self.paste_text("FlowClone works!")

        threading.Thread(target=worker, daemon=True).start()
        self.set_status("click a text field — pasting in 3s")

    # ---- settings ----
    def toggle_hot(self, item):
        item.state = not item.state
        self.cfg["hot_load"] = bool(item.state)
        save_config(self.cfg)
        if self.cfg["hot_load"]:
            threading.Thread(target=self._preload, daemon=True).start()
        else:
            self.release_model()
            self.set_status("idle (model loads on demand)")

    def quit_app(self, _):
        try:
            self.listener.stop()
        except Exception:
            pass
        rumps.quit_application()


if __name__ == "__main__":
    FlowClone().run()
