#!/usr/bin/env python3

from __future__ import annotations
import os, sys, json, time, wave, random, hashlib, subprocess, shutil
import re
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional

# ---- dependencies check (friendly) ----
def require(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False

if not require("numpy") or not require("torch"):
    print("Missing dependencies. Activate your venv, then install:")
    print("  pip install numpy torch")
    sys.exit(1)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def hr():
    print("-" * 72)

def prompt(s: str, default: Optional[str] = None) -> str:
    if default is not None:
        r = input(f"{s} [{default}]: ").strip()
        return r if r else default
    return input(f"{s}: ").strip()

def prompt_int(s: str, default: int, minv: int = 0, maxv: int = 10**9) -> int:
    while True:
        r = input(f"{s} [{default}]: ").strip()
        if not r:
            return default
        try:
            v = int(r)
            if v < minv or v > maxv:
                raise ValueError()
            return v
        except Exception:
            print(f"Please enter an integer in range [{minv}, {maxv}].")

def prompt_float(s: str, default: float, minv: float = -1e9, maxv: float = 1e9) -> float:
    while True:
        r = input(f"{s} [{default}]: ").strip()
        if not r:
            return default
        try:
            v = float(r)
            if v < minv or v > maxv:
                raise ValueError()
            return v
        except Exception:
            print(f"Please enter a number in range [{minv}, {maxv}].")

def prompt_bool(s: str, default: bool) -> bool:
    d = "Y/n" if default else "y/N"
    while True:
        r = input(f"{s} ({d}): ").strip().lower()
        if not r:
            return default
        if r in ("y","yes","1","true","t"):
            return True
        if r in ("n","no","0","false","f"):
            return False
        print("Please enter y or n.")

def normpath(p: str) -> str:
    p = p.strip().strip('"').strip("'")
    return os.path.normpath(p)

def fslash(p: str) -> str:
    return normpath(p).replace("\\", "/")

def exists_dir(p: str) -> bool:
    return p and os.path.isdir(p)

def ensure_dir(p: str):
    if p:
        os.makedirs(p, exist_ok=True)


def sanitize_filename(name: str, max_len: int = 80) -> str:
    if name is None:
        name = ""
    s = str(name)
    s = s.strip()
    
    s = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", s)
    
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        s = "alias"
    
    s = s.rstrip(" .")
    if len(s) > max_len:
        s = s[:max_len].rstrip(" .")
    return s

def sha1_short(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:12]

def safe_rmtree(path: str):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)

def safe_remove(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def wav_info(path: str) -> Tuple[int, int, int, int]:
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        n = wf.getnframes()
    return sr, ch, sw, n

def read_wav_mono_float(path: str) -> Tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        n = wf.getnframes()
        raw = wf.readframes(n)

    if sw == 1:
        x = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        x = (x - 128.0) / 128.0
    elif sw == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        x = (b[:, 0].astype(np.int32) |
             (b[:, 1].astype(np.int32) << 8) |
             (b[:, 2].astype(np.int32) << 16))
        x = np.where(x >= 2**23, x - 2**24, x).astype(np.float32) / (2**23)
    elif sw == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sampwidth={sw}")
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return np.nan_to_num(x).astype(np.float32), sr

def write_wav_pcm16_mono(path: str, x: np.ndarray, sr: int):
    ensure_dir(os.path.dirname(path))
    x = np.nan_to_num(x.astype(np.float32))
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())

def resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    ratio = sr_out / float(sr_in)
    n_out = int(round(len(x) * ratio))
    if n_out <= 1:
        return np.zeros((1,), dtype=np.float32)
    xp = np.linspace(0.0, 1.0, num=len(x), endpoint=True)
    xq = np.linspace(0.0, 1.0, num=n_out, endpoint=True)
    return np.interp(xq, xp, x).astype(np.float32)

def ms_to_samples(ms: float, sr: int) -> int:
    return int(round((ms / 1000.0) * sr))


def stft_logmag(x: np.ndarray, n_fft: int, hop: int, win: int) -> np.ndarray:
    x = x.astype(np.float32)
    w = np.hanning(win).astype(np.float32)
    pad = n_fft // 2
    xpad = np.pad(x, (pad, pad), mode="reflect")
    if len(xpad) < win:
        xpad = np.pad(xpad, (0, win - len(xpad)), mode="edge")
    n_frames = 1 + (len(xpad) - win) // hop
    frames = np.lib.stride_tricks.as_strided(
        xpad,
        shape=(n_frames, win),
        strides=(xpad.strides[0] * hop, xpad.strides[0]),
        writeable=False
    )
    frames = frames * w[None, :]
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    mag = np.abs(spec).astype(np.float32) + 1e-7
    logmag = np.log(mag).astype(np.float32)
    return logmag.T  # (F,T)

def center_crop_or_pad(a: np.ndarray, F: int, T: int) -> np.ndarray:
    f0, t0 = a.shape
    if f0 < F:
        a = np.pad(a, ((0, F - f0), (0, 0)), mode="edge")
    elif f0 > F:
        a = a[:F, :]
    f0, t0 = a.shape
    if t0 < T:
        a = np.pad(a, ((0, 0), (0, T - t0)), mode="edge")
    elif t0 > T:
        start = (t0 - T) // 2
        a = a[:, start:start + T]
    return a.astype(np.float32)

def resolve_path(dataset_dir: str, p: str) -> str:
    if not p:
        return p
    p2 = p.replace("/", os.sep).replace("\\", os.sep)
    if os.path.isabs(p2):
        return p2
    return os.path.normpath(os.path.join(dataset_dir, p2))


@dataclass
class OtoEntry:
    wav: str
    alias: str
    offset: float
    consonant: float
    cutoff: float
    preutter: float
    overlap: float

def read_text_auto(path: str) -> str:
    for enc in ("utf-8", "shift_jis", "cp932"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            continue
    with open(path, "rb") as f:
        data = f.read()
    return data.decode("utf-8", errors="replace")

def parse_oto(oto_path: str) -> List[OtoEntry]:
    txt = read_text_auto(oto_path)
    out: List[OtoEntry] = []
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        if "=" not in line:
            continue
        left, right = line.split("=", 1)
        wav = left.strip()
        parts = right.split(",")
        if len(parts) < 6:
            continue
        alias = parts[0].strip()
        try:
            offset = float(parts[1])
            consonant = float(parts[2])
            cutoff = float(parts[3])
            preutter = float(parts[4])
            overlap = float(parts[5])
        except Exception:
            continue
        out.append(OtoEntry(wav=wav, alias=alias, offset=offset, consonant=consonant, cutoff=cutoff, preutter=preutter, overlap=overlap))
    return out

import re as _re
NOTE_RE = _re.compile(r'^(?P<n>[A-Ga-g])(?P<acc>(#|b)?)?(?P<oct>[0-8])$')

def normalize_note_token(tok: str):
    tok = (tok or "").strip()
    if not tok:
        return None
    m = NOTE_RE.match(tok)
    if not m:
        return None
    n = m.group("n").upper()
    acc = m.group("acc")
    octv = m.group("oct")
    return f"{n}{acc}{octv}"

def detect_pitch_from_wav_relpath(wav_rel: str):
    parts = (wav_rel or "").replace("\\", "/").replace("\\", "/").split("/")
    for seg in parts:
        p = normalize_note_token(seg)
        if p:
            return p
    return None


def gather_multipitch_oto_entries(voicebank_dir: str, oto_relpath: str) -> List[OtoEntry]:

    all_entries: List[OtoEntry] = []
    try:
        subdirs = [d for d in os.listdir(voicebank_dir) if os.path.isdir(os.path.join(voicebank_dir, d))]
    except Exception:
        return all_entries

    for sd in sorted(subdirs):
        oto_path = os.path.join(voicebank_dir, sd, oto_relpath)
        if not os.path.exists(oto_path):
            continue
        ents = parse_oto(oto_path)
        
        fixed: List[OtoEntry] = []
        for e in ents:
            wav_rel = e.wav.replace("\\", "/")
            if "/" not in wav_rel:
                wav_rel = f"{sd}/{wav_rel}"
            else:
        
                pass
            fixed.append(OtoEntry(
                wav=wav_rel,
                alias=e.alias,
                offset=e.offset,
                consonant=e.consonant,
                cutoff=e.cutoff,
                preutter=e.preutter,
                overlap=e.overlap
            ))
        all_entries.extend(fixed)

    return all_entries

def end_from_cutoff(total_ms: float, cutoff: float) -> float:
    return (total_ms + cutoff) if cutoff < 0 else (total_ms - cutoff)

def compute_length_ms(total_ms: float, offset: float, cutoff: float) -> int:
    end = end_from_cutoff(total_ms, cutoff)
    length = max(50.0, end - max(0.0, offset))
    return int(round(length))

def slice_target_simple(src_wav: str, dst_wav: str, offset_ms: float, cutoff_ms: float, min_ms: float = 120.0):
    x, sr = read_wav_mono_float(src_wav)
    total_ms = (len(x) / float(sr)) * 1000.0
    start_ms = max(0.0, float(offset_ms))
    end_ms = end_from_cutoff(total_ms, float(cutoff_ms))
    # clamp and ensure minimum length
    end_ms = min(total_ms, max(end_ms, start_ms + float(min_ms)))
    s0 = max(0, min(len(x), ms_to_samples(start_ms, sr)))
    s1 = max(0, min(len(x), ms_to_samples(end_ms, sr)))
    if s1 <= s0 + 8:
        s0, s1 = 0, len(x)
    write_wav_pcm16_mono(dst_wav, x[s0:s1].copy(), sr)

def render_with_resampler(resampler: str, cmd_template: str, in_wav: str, out_wav: str,
                          pitch: str, velocity: int, flags: str, offset: int, length: int, consonant: int,
                          cutoff: float, volume: int, modulation: int, tempo: float, extras: str,
                          dry_run: bool, log_fn):
    cmd = cmd_template.format(
        resampler=resampler,
        in_wav=in_wav,
        out_wav=out_wav,
        pitch=pitch,
        velocity=velocity,
        flags=flags,
        offset=offset,
        length=length,
        consonant=consonant,
        cutoff=cutoff,
        volume=volume,
        modulation=modulation,
        tempo=tempo,
        extras=extras,
    )
    log_fn(f"CMD: {cmd}")
    if dry_run:
        return 0
    p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        log_fn(f"ERROR rc={p.returncode}")
        try:
            log_fn(p.stderr.decode("utf-8", errors="replace")[:2000])
        except Exception:
            pass
    return p.returncode


class ShardDataset(Dataset):
    def __init__(self, cache_root: str):
        man_path = os.path.join(cache_root, "manifest.json")
        if not os.path.exists(man_path):
            raise FileNotFoundError(f"Missing manifest: {man_path}")
        with open(man_path, "r", encoding="utf-8") as f:
            self.man = json.load(f)
        self.cache_root = cache_root
        self.shards = self.man["shards"]
        self.index = []
        for si, s in enumerate(self.shards):
            for j in range(int(s["count"])):
                self.index.append((si, j))
        random.shuffle(self.index)
        self._loaded_si = None
        self._X = None
        self._Y = None

    def __len__(self) -> int:
        return len(self.index)

    def _load_shard(self, si: int):
        if self._loaded_si == si:
            return
        p = os.path.join(self.cache_root, self.shards[si]["path"])
        z = np.load(p, allow_pickle=False)
        self._X = z["X"]
        self._Y = z["Y"]
        self._loaded_si = si

    def __getitem__(self, idx: int):
        si, j = self.index[idx]
        self._load_shard(si)
        x = self._X[j].astype(np.float32)
        y = self._Y[j].astype(np.float32)
        x = x - float(x.mean())
        y = y - float(y.mean())
        return torch.from_numpy(x), torch.from_numpy(y)

class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 3):
        super().__init__()
        p = k // 2
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, padding=p),
            nn.GroupNorm(num_groups=min(8, c_out), num_channels=c_out),
            nn.SiLU(),
            nn.Conv2d(c_out, c_out, k, padding=p),
            nn.GroupNorm(num_groups=min(8, c_out), num_channels=c_out),
            nn.SiLU(),
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.block = ConvBlock(c_in, c_out)
        self.pool = nn.AvgPool2d(2, 2)
    def forward(self, x):
        h = self.block(x)
        return self.pool(h), h

class Up(nn.Module):
    def __init__(self, c_in: int, c_skip: int, c_out: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.block = ConvBlock(c_in + c_skip, c_out)
    def forward(self, x, skip):
        x = self.up(x)
        dh = skip.shape[-2] - x.shape[-2]
        dw = skip.shape[-1] - x.shape[-1]
        if dh or dw:
            x = F.pad(x, (0, max(0, dw), 0, max(0, dh)))
            x = x[:, :, :skip.shape[-2], :skip.shape[-1]]
        return self.block(torch.cat([x, skip], dim=1))

class STFTTimbreUNet(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()
        self.inp = ConvBlock(1, base)
        self.d1 = Down(base, base * 2)
        self.d2 = Down(base * 2, base * 4)
        self.mid = ConvBlock(base * 4, base * 4)
        self.u2 = Up(base * 4, base * 4, base * 2)
        self.u1 = Up(base * 2, base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)
    def forward(self, x):
        x0 = self.inp(x)
        x1, s1 = self.d1(x0)
        x2, s2 = self.d2(x1)
        xm = self.mid(x2)
        xu2 = self.u2(xm, s2)
        xu1 = self.u1(xu2, s1)
        return self.out(xu1)

def save_ckpt(path: str, model: nn.Module, opt: torch.optim.Optimizer, step: int, best: float, cfg: dict):
    ensure_dir(os.path.dirname(path))
    torch.save({"model_state": model.state_dict(), "opt_state": opt.state_dict(), "step": step, "best": best, "cfg": cfg}, path)


DEFAULT_CFG_NAME = "last.json"

@dataclass
class WizardCfg:
    # dataset build
    voicebank_dir: str = ""
    dataset_dir: str = ""
    voicebank: str = "Luminoid"
    oto_relpath: str = "oto.ini"
    resampler_exe: str = ""
    cmd_template: str = "\"{resampler}\" \"{in_wav}\" \"{out_wav}\" {pitch} {velocity} \"{flags}\" {offset} {length} {consonant} {cutoff} {volume} {modulation} !{tempo} \"{extras}\""
    flags: str = "g0B0H0P86"
    tempo: float = 195.0
    velocity: int = 100
    volume: int = 100
    modulation: int = 0
    pitches_csv: str = "C4"
    multipitch_auto: bool = True 
    dry_run: bool = False
    limit: int = 0

    format_before_build: bool = True
    wipe_cache_models_for_bank: bool = False


    min_target_ms: float = 120.0

    
    n_fft: int = 1024
    hop: int = 256
    win: int = 1024
    patch_T: int = 96
    max_seconds: float = 1.2
    shard_size: int = 512
    no_resample: bool = False

    
    epochs: int = 6
    batch_size: int = 16
    lr: float = 2e-4
    base: int = 32
    print_every: int = 25


    wrapper_dir: str = ""
    real_resampler: str = ""
    model_root: str = ""
    log_file: str = ""
    strength: float = 0.55

def load_last_cfg(path: str) -> WizardCfg:
    if not os.path.exists(path):
        return WizardCfg()
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return WizardCfg(**d)
    except Exception:
        return WizardCfg()

def save_last_cfg(path: str, cfg: WizardCfg):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def format_dataset(cfg: WizardCfg):
    ds = normpath(cfg.dataset_dir)
    if not exists_dir(ds):
        print("Dataset dir not set or does not exist. Run Setup first.")
        return

    inputs_dir = os.path.join(ds, "inputs")
    targets_dir = os.path.join(ds, "targets")
    meta_path = os.path.join(ds, "inputs_meta.jsonl")
    log_path = os.path.join(ds, "dataset_build_log.txt")

    hr()
    print("FORMAT DATASET (CLEAN)")
    print(f"dataset_dir: {ds}")
    print("Deletes: inputs/, targets/, inputs_meta.jsonl, dataset_build_log.txt")
    if cfg.wipe_cache_models_for_bank:
        print(f"Also deletes stft_cache + models for bank '{cfg.voicebank}'")
    hr()

    if not prompt_bool("Proceed with deletion?", False):
        print("Cancelled.")
        return

    safe_rmtree(inputs_dir)
    safe_rmtree(targets_dir)
    safe_remove(meta_path)
    safe_remove(log_path)

    if cfg.wipe_cache_models_for_bank:
        safe_rmtree(os.path.join(ds, "stft_cache", cfg.voicebank))
        safe_rmtree(os.path.join(ds, "models", "stft_timbre", cfg.voicebank))

    print("Done.")

# -----------------------------
# Build dataset
# -----------------------------

def dataset_build(cfg: WizardCfg):
    vb_dir = normpath(cfg.voicebank_dir)
    ds_dir = normpath(cfg.dataset_dir)
    ensure_dir(ds_dir)

    oto_path = os.path.join(vb_dir, cfg.oto_relpath)
    if not os.path.exists(oto_path):
        raise SystemExit(f"oto.ini not found: {oto_path}")

    if not cfg.resampler_exe or not os.path.exists(normpath(cfg.resampler_exe)):
        raise SystemExit("Resampler exe path is not set or not found. Set it in Setup.")

    pitches = [p.strip() for p in cfg.pitches_csv.split(",") if p.strip()]
    if not pitches:
        raise SystemExit("No pitches provided. Example: C4 or C4,C#4")

    inputs_dir = os.path.join(ds_dir, "inputs")
    targets_dir = os.path.join(ds_dir, "targets")
    ensure_dir(inputs_dir)
    ensure_dir(targets_dir)

    meta_path = os.path.join(ds_dir, "inputs_meta.jsonl")
    log_path = os.path.join(ds_dir, "dataset_build_log.txt")

    def log_fn(s: str):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(s + "\n")

    if cfg.format_before_build:
        safe_rmtree(inputs_dir)
        safe_rmtree(targets_dir)
        ensure_dir(inputs_dir)
        ensure_dir(targets_dir)
        safe_remove(meta_path)
        safe_remove(log_path)
        if cfg.wipe_cache_models_for_bank:
            safe_rmtree(os.path.join(ds_dir, "stft_cache", cfg.voicebank))
            safe_rmtree(os.path.join(ds_dir, "models", "stft_timbre", cfg.voicebank))

    hr()
    print("DATASET BUILD")
    print(f"voicebank_dir: {vb_dir}")
    print(f"oto_path     : {oto_path}")
    print(f"dataset_dir  : {ds_dir}")
    print(f"resampler_exe: {cfg.resampler_exe}")
    print(f"pitches      : {pitches}")
    print(f"min_target_ms: {cfg.min_target_ms}")
    print(f"format_before_build: {cfg.format_before_build}")
    print(f"wipe_cache_models_for_bank: {cfg.wipe_cache_models_for_bank}")
    print(f"dry_run      : {cfg.dry_run}")
    print(f"limit        : {cfg.limit}")
    hr()

    
    if getattr(cfg, "multipitch_auto", False):
        entries = gather_multipitch_oto_entries(vb_dir, cfg.oto_relpath)
        if not entries:
            # fallback: use base oto if no subfolder oto.ini found
            entries = parse_oto(oto_path)
    else:
        entries = parse_oto(oto_path)
    if not entries:
        raise SystemExit("No oto entries parsed. Check encoding or oto format.")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"Started dataset build at {time.ctime()}\n")

    written = 0
    detected_pitch_count = 0
    fallback_pitch_count = 0
    rendered_ok = 0
    rendered_fail = 0
    skipped_missing = 0
    target_fail = 0

    for idx, e in enumerate(entries, start=1):
        src_wav = os.path.join(vb_dir, e.wav)
        if not os.path.exists(src_wav):
            alt = os.path.join(vb_dir, os.path.basename(e.wav))
            if os.path.exists(alt):
                src_wav = alt
            else:
                skipped_missing += 1
                continue

        try:
            sr, ch, sw, n = wav_info(src_wav)
            total_ms = (n / float(sr)) * 1000.0
        except Exception:
            skipped_missing += 1
            continue

        
        tgt_key = f"{e.alias}|{src_wav}|{e.offset}|{e.cutoff}|simple"
        tgt_hash = sha1_short(tgt_key)
        alias_stem = sanitize_filename(e.alias)
        tgt_name = f"{alias_stem}__{tgt_hash}.wav"
        tgt_path = os.path.join(targets_dir, tgt_name)

        if not cfg.dry_run:
            try:
                slice_target_simple(src_wav, tgt_path, e.offset, e.cutoff, min_ms=float(cfg.min_target_ms))
            except Exception as ex:
                target_fail += 1
                log_fn(f"ERROR target slice failed: {src_wav} alias={e.alias} ex={ex}")
                continue

        
        offset = int(round(max(0.0, e.offset)))
        consonant = int(round(max(0.0, e.consonant)))
        cutoff = float(e.cutoff)
        length = compute_length_ms(total_ms, e.offset, cutoff)

        
        detected = detect_pitch_from_wav_relpath(e.wav) if getattr(cfg, "multipitch_auto", False) else None
        if detected:
            detected_pitch_count += 1
        else:
            fallback_pitch_count += 1
        entry_pitches = [detected] if detected else pitches

        for pitch in entry_pitches:
            key = f"{e.alias}|{pitch}|{src_wav}|{offset}|{length}|{consonant}|{cutoff}|{cfg.flags}|{cfg.tempo}|{cfg.velocity}"
            h = sha1_short(key)
            in_name = f"{alias_stem}__{h}__{pitch}.wav"
            in_path = os.path.join(inputs_dir, in_name)

            rc = render_with_resampler(
                resampler=normpath(cfg.resampler_exe),
                cmd_template=cfg.cmd_template,
                in_wav=src_wav,
                out_wav=in_path,
                pitch=pitch,
                velocity=cfg.velocity,
                flags=cfg.flags,
                offset=offset,
                length=length,
                consonant=consonant,
                cutoff=cutoff,
                volume=cfg.volume,
                modulation=cfg.modulation,
                tempo=cfg.tempo,
                extras="AA",
                dry_run=cfg.dry_run,
                log_fn=log_fn
            )

            ok = (rc == 0) and (cfg.dry_run or (os.path.exists(in_path) and os.path.getsize(in_path) > 44))
            if ok:
                rendered_ok += 1
            else:
                rendered_fail += 1
                continue

            rel_in = os.path.relpath(in_path, ds_dir).replace("\\", "/")
            rel_tg = os.path.relpath(tgt_path, ds_dir).replace("\\", "/")
            meta = {"input_wav": rel_in, "target_wav": rel_tg, "alias": e.alias, "pitch": pitch, "hash": h}
            with open(meta_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            written += 1

            if cfg.limit and written >= cfg.limit:
                break
        if cfg.limit and written >= cfg.limit:
            break

        if idx % 250 == 0:
            print(f"Processed oto {idx}/{len(entries)} | meta={written} ok={rendered_ok} fail={rendered_fail} missing={skipped_missing} target_fail={target_fail}")

    hr()
    print("DATASET BUILD COMPLETE")
    print(f"meta file     : {meta_path}")
    print(f"meta lines    : {written}")
    print(f"render ok/fail: {rendered_ok}/{rendered_fail}")
    print(f"missing wavs  : {skipped_missing}")
    print(f"target slice failures: {target_fail}")
    print(f"pitch detection: detected={detected_pitch_count} fallback={fallback_pitch_count}")
    hr()



def cache_stft(cfg: WizardCfg) -> str:
    meta_path = os.path.join(cfg.dataset_dir, "inputs_meta.jsonl")
    if not os.path.exists(meta_path):
        raise SystemExit(f"Missing: {meta_path} (build dataset first)")

    cache_root = os.path.join(cfg.dataset_dir, "stft_cache", cfg.voicebank)
    ensure_dir(cache_root)

    items: List[Dict[str, Any]] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            it = json.loads(line)
            items.append(it)
            if cfg.limit and len(items) >= cfg.limit:
                break

    skip_missing = skip_short = skip_sr = kept = 0
    shard_idx = 0
    buf_X: List[np.ndarray] = []
    buf_Y: List[np.ndarray] = []
    buf_meta: List[Dict[str, Any]] = []

    manifest = {
        "voicebank": cfg.voicebank,
        "n_fft": cfg.n_fft, "hop": cfg.hop, "win": cfg.win,
        "patch_T": cfg.patch_T,
        "max_seconds": cfg.max_seconds,
        "shards": []
    }

    hr()
    print(f"CACHING STFT patches voicebank='{cfg.voicebank}'")
    print(f"dataset_dir: {cfg.dataset_dir}")
    print(f"cache_root : {cache_root}")
    print(f"meta lines : {len(items)}")
    hr()

    for i, it in enumerate(items, start=1):
        in_rel = it.get("input_wav")
        tg_rel = it.get("target_wav")
        if not in_rel or not tg_rel:
            continue
        in_wav = resolve_path(cfg.dataset_dir, in_rel)
        tg_wav = resolve_path(cfg.dataset_dir, tg_rel)
        if (not os.path.exists(in_wav)) or (not os.path.exists(tg_wav)):
            skip_missing += 1
            continue

        x, sr1 = read_wav_mono_float(in_wav)
        y, sr2 = read_wav_mono_float(tg_wav)
        if sr1 != sr2:
            if cfg.no_resample:
                skip_sr += 1
                continue
            y = resample_linear(y, sr2, sr1)

        if len(x) < int(0.1 * sr1) or len(y) < int(0.1 * sr1):
            skip_short += 1
            continue

        max_len = int(cfg.max_seconds * sr1)
        if len(x) > max_len:
            s = (len(x) - max_len) // 2
            x = x[s:s + max_len]
        if len(y) > max_len:
            s = (len(y) - max_len) // 2
            y = y[s:s + max_len]

        X = stft_logmag(x, cfg.n_fft, cfg.hop, cfg.win)
        Y = stft_logmag(y, cfg.n_fft, cfg.hop, cfg.win)
        Fbins = cfg.n_fft // 2 + 1
        X = center_crop_or_pad(X, Fbins, cfg.patch_T)
        Y = center_crop_or_pad(Y, Fbins, cfg.patch_T)

        buf_X.append(X[None, None, :, :])
        buf_Y.append(Y[None, None, :, :])
        buf_meta.append({"input_wav": in_rel, "target_wav": tg_rel, "alias": it.get("alias"), "pitch": it.get("pitch"), "hash": it.get("hash")})
        kept += 1

        if len(buf_X) >= cfg.shard_size:
            Xs = np.concatenate(buf_X, axis=0).astype(np.float16)
            Ys = np.concatenate(buf_Y, axis=0).astype(np.float16)
            shard_path = os.path.join(cache_root, f"shard_{shard_idx:05d}.npz")
            np.savez_compressed(shard_path, X=Xs, Y=Ys, meta=json.dumps(buf_meta, ensure_ascii=False))
            manifest["shards"].append({"path": os.path.basename(shard_path), "count": int(Xs.shape[0])})
            print(f"Wrote {os.path.basename(shard_path)}  N={Xs.shape[0]}")
            shard_idx += 1
            buf_X, buf_Y, buf_meta = [], [], []

        if i % 500 == 0:
            print(f"Processed {i}/{len(items)} | kept={kept} missing={skip_missing} sr_skip={skip_sr} short={skip_short}")

    if buf_X:
        Xs = np.concatenate(buf_X, axis=0).astype(np.float16)
        Ys = np.concatenate(buf_Y, axis=0).astype(np.float16)
        shard_path = os.path.join(cache_root, f"shard_{shard_idx:05d}.npz")
        np.savez_compressed(shard_path, X=Xs, Y=Ys, meta=json.dumps(buf_meta, ensure_ascii=False))
        manifest["shards"].append({"path": os.path.basename(shard_path), "count": int(Xs.shape[0])})
        print(f"Wrote {os.path.basename(shard_path)}  N={Xs.shape[0]}")

    man_path = os.path.join(cache_root, "manifest.json")
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    total = sum(int(s["count"]) for s in manifest["shards"])
    hr()
    print(f"Wrote manifest: {man_path}")
    print(f"Total cached patches: {total}")
    print(f"Skip stats: missing_files={skip_missing} sr_skipped={skip_sr} too_short={skip_short} kept={kept}")
    hr()
    return cache_root


def train_stft(cfg: WizardCfg) -> str:
    cache_root = os.path.join(cfg.dataset_dir, "stft_cache", cfg.voicebank)
    man_path = os.path.join(cache_root, "manifest.json")
    if not os.path.exists(man_path):
        raise SystemExit(f"Missing cache manifest: {man_path} (run Cache first)")

    with open(man_path, "r", encoding="utf-8") as f:
        man = json.load(f)

    ds = ShardDataset(cache_root)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=False)

    device = torch.device("cpu")
    model = STFTTimbreUNet(base=cfg.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    out_root = os.path.join(cfg.dataset_dir, "models", "stft_timbre", cfg.voicebank)
    best_path = os.path.join(out_root, "best.pt")
    last_path = os.path.join(out_root, "last.pt")
    ensure_dir(out_root)

    train_meta = {
        "voicebank": cfg.voicebank,
        "base": cfg.base,
        "n_fft": int(man["n_fft"]), "hop": int(man["hop"]), "win": int(man["win"]),
        "patch_T": int(man["patch_T"])
    }

    hr()
    print(f"TRAINING voicebank='{cfg.voicebank}'")
    print(f"cache_root: {cache_root}")
    print(f"model_out : {out_root}")
    print(f"epochs={cfg.epochs} batch_size={cfg.batch_size} lr={cfg.lr} base={cfg.base}")
    hr()

    step = 0
    best = float("inf")
    t0 = time.time()

    for ep in range(1, cfg.epochs + 1):
        model.train()
        run = 0.0
        for bi, (x, y) in enumerate(dl, start=1):
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = F.l1_loss(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            run += float(loss.item())
            if bi == 1 or bi % cfg.print_every == 0:
                avg = run / bi
                print(f"epoch {ep:03d} | batch {bi:05d}/{len(dl):05d} | avg_L1 {avg:.5f} | last_L1 {loss.item():.5f} | {(time.time()-t0)/60:.1f} min")

        ep_loss = run / max(1, len(dl))
        save_ckpt(last_path, model, opt, step, best, train_meta)
        if ep_loss < best:
            best = ep_loss
            save_ckpt(best_path, model, opt, step, best, train_meta)
            print(f"==> new best: {best:.5f} (saved best.pt)")

    hr()
    print("Done.")
    print(f"model: {best_path}")
    hr()
    return out_root


def gen_wrapper_cfg(cfg: WizardCfg):
    if not cfg.wrapper_dir or not exists_dir(cfg.wrapper_dir):
        print("Wrapper directory not set or does not exist. Set it first.")
        return
    real = cfg.real_resampler or cfg.resampler_exe
    if not real or not os.path.exists(normpath(real)):
        print("Real resampler path not set or not found. Set it first.")
        return
    model_root = cfg.model_root or os.path.join(cfg.dataset_dir, "models", "stft_timbre")
    log_file = cfg.log_file or os.path.join(cfg.wrapper_dir, "timbre_wrapper_stft_log.txt")

    out_cfg = {
        "real_resampler": fslash(real),
        "log_file": fslash(log_file),
        "device": "cpu",
        "strength": float(cfg.strength),
        "rms_match": True,
        "n_fft": int(cfg.n_fft),
        "hop": int(cfg.hop),
        "win": int(cfg.win),
        "base": int(cfg.base),
        "model_root": fslash(model_root),
        "model_bank": cfg.voicebank,
        "model_file": "best.pt"
    }

    out_path = os.path.join(cfg.wrapper_dir, "utau_timbre_resampler_wrapper_stft_v2_bankselect.config.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_cfg, f, ensure_ascii=False, indent=2)

    print(f"Wrote wrapper config: {out_path}")
    print("Reminder: ensure wrapper .bat points to your venv python.exe.")


def interactive_setup(cfg: WizardCfg) -> WizardCfg:
    hr()
    print("SETUP (Simple slicing)")
    print("Press Enter to accept defaults.")
    hr()

    while True:
        vb = prompt("Voicebank directory (Singers/<Bank>)", cfg.voicebank_dir or r"C:\Users\bali1\Documents\OpenUtau\Singers\Luminoid")
        vb = normpath(vb)
        if exists_dir(vb):
            cfg.voicebank_dir = vb
            break
        print("That folder does not exist. Try again.")

    cfg.oto_relpath = prompt("oto.ini relative path", cfg.oto_relpath or "oto.ini")

    ds = prompt("Dataset output directory", cfg.dataset_dir or r"V:\DATASET_OUT")
    cfg.dataset_dir = normpath(ds)
    ensure_dir(cfg.dataset_dir)

    cfg.voicebank = prompt("Model voicebank name (folder name)", cfg.voicebank or "Luminoid").strip() or "Luminoid"
    cfg.resampler_exe = normpath(prompt("Real resampler exe path", cfg.resampler_exe or r"V:\doppeltler64.exe"))

    if prompt_bool("Customize resampler render params now?", False):
        cfg.cmd_template = prompt("Resampler command template", cfg.cmd_template)
        cfg.flags = prompt("Flags", cfg.flags)
        cfg.tempo = prompt_float("Tempo", cfg.tempo, 10.0, 400.0)
        cfg.velocity = prompt_int("Velocity", cfg.velocity, 10, 200)
        cfg.volume = prompt_int("Volume", cfg.volume, 0, 200)
        cfg.modulation = prompt_int("Modulation", cfg.modulation, 0, 200)
        cfg.pitches_csv = prompt("Fallback pitches CSV (used if pitch cannot be detected)", cfg.pitches_csv)
        cfg.multipitch_auto = prompt_bool("Multipitch mode: detect pitch from folder name AND read oto.ini from pitch subfolders only?", cfg.multipitch_auto)
        cfg.dry_run = prompt_bool("Dry run (do not render/slice wavs)?", cfg.dry_run)
        cfg.limit = prompt_int("Limit meta lines (0 = all)", cfg.limit, 0, 10**9)

    cfg.format_before_build = prompt_bool("Format (clean) inputs/targets/meta before Build?", cfg.format_before_build)
    cfg.wipe_cache_models_for_bank = prompt_bool("Also wipe stft_cache + models for this bank on format?", cfg.wipe_cache_models_for_bank)
    cfg.min_target_ms = prompt_float("min_target_ms for target slices", cfg.min_target_ms, 40.0, 2000.0)

    if prompt_bool("Customize cache/train params now?", False):
        cfg.n_fft = prompt_int("n_fft (1024 recommended)", cfg.n_fft, 256, 8192)
        cfg.win = prompt_int("win_length (usually same as n_fft)", cfg.win, 256, 8192)
        cfg.hop = prompt_int("hop_length (256 recommended)", cfg.hop, 64, 4096)
        cfg.patch_T = prompt_int("patch_T frames (96 recommended)", cfg.patch_T, 16, 512)
        cfg.max_seconds = prompt_float("max_seconds audio crop (1.2 recommended)", cfg.max_seconds, 0.2, 10.0)
        cfg.shard_size = prompt_int("shard_size (512 recommended)", cfg.shard_size, 16, 4096)
        cfg.no_resample = prompt_bool("If SR differs, SKIP instead of resample?", cfg.no_resample)

        cfg.epochs = prompt_int("epochs", cfg.epochs, 1, 999)
        cfg.batch_size = prompt_int("batch_size (CPU: 8-32)", cfg.batch_size, 1, 256)
        cfg.lr = prompt_float("learning rate", cfg.lr, 1e-7, 1e-2)
        cfg.base = prompt_int("UNet base channels (32 recommended)", cfg.base, 8, 256)
        cfg.print_every = prompt_int("print_every batches", cfg.print_every, 1, 10000)

    return cfg

def wrapper_setup(cfg: WizardCfg) -> WizardCfg:
    hr()
    print("WRAPPER CONFIG SETUP (optional)")
    hr()
    cfg.wrapper_dir = normpath(prompt("Wrapper folder (where wrapper .py/.bat live)", cfg.wrapper_dir or r"C:\Users\bali1\Documents\OpenUtau\Resamplers\timbre_resampler"))
    if not exists_dir(cfg.wrapper_dir):
        print("Folder does not exist. Skipping.")
        return cfg
    cfg.real_resampler = normpath(prompt("Real resampler exe path (can reuse)", cfg.real_resampler or cfg.resampler_exe or r"V:\doppeltler64.exe"))
    cfg.model_root = normpath(prompt("Model root folder (blank = dataset_dir default)", cfg.model_root or ""))
    cfg.log_file = normpath(prompt("Log file path (blank = wrapper default)", cfg.log_file or ""))
    cfg.strength = prompt_float("Default strength", cfg.strength, 0.0, 1.0)
    return cfg

def run_menu(cfg: WizardCfg, cfg_path: str):
    while True:
        hr()
        print("STFT-R WIZARD")
        print(f"voicebank_dir: {cfg.voicebank_dir or '(not set)'}")
        print(f"dataset_dir  : {cfg.dataset_dir or '(not set)'}")
        print(f"model bank   : {cfg.voicebank}")
        print(f"pitches      : {cfg.pitches_csv} | multipitch_auto={cfg.multipitch_auto}")
        print(f"format_before_build: {cfg.format_before_build} | wipe_cache_models_for_bank: {cfg.wipe_cache_models_for_bank}")
        print(f"min_target_ms: {cfg.min_target_ms}")
        print("")
        print("1) Setup / change paths + params")
        print("2) Format dataset folder")
        print("3) Build dataset")
        print("4) Cache STFT patches")
        print("5) Train model")
        print("6) Full pipeline: Build + Cache + Train")
        print("7) Configure wrapper (write bank-select config.json)")
        print("8) Exit")
        hr()
        ch = input("Select: ").strip()

        if ch == "1":
            cfg = interactive_setup(cfg)
            save_last_cfg(cfg_path, cfg)
        elif ch == "2":
            format_dataset(cfg)
            save_last_cfg(cfg_path, cfg)
        elif ch == "3":
            dataset_build(cfg)
            save_last_cfg(cfg_path, cfg)
        elif ch == "4":
            cache_stft(cfg)
            save_last_cfg(cfg_path, cfg)
        elif ch == "5":
            train_stft(cfg)
            save_last_cfg(cfg_path, cfg)
        elif ch == "6":
            dataset_build(cfg)
            cache_stft(cfg)
            train_stft(cfg)
            save_last_cfg(cfg_path, cfg)
        elif ch == "7":
            cfg = wrapper_setup(cfg)
            gen_wrapper_cfg(cfg)
            save_last_cfg(cfg_path, cfg)
        elif ch == "8":
            print("Bye.")
            return
        else:
            print("Invalid selection.")

def main():
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DEFAULT_CFG_NAME)
    cfg = load_last_cfg(cfg_path)
    run_menu(cfg, cfg_path)

if __name__ == "__main__":
    main()
