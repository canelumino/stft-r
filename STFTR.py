
from __future__ import annotations
import os, sys, json, subprocess, wave, tempfile, shutil
from datetime import datetime
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def ts(): return datetime.now().isoformat(timespec="seconds")
def append_log(path: str, msg: str):
    d=os.path.dirname(path)
    if d: os.makedirs(d, exist_ok=True)
    with open(path,"a",encoding="utf-8") as f: f.write(msg)

def load_cfg(script_dir: str) -> dict:
    p=os.path.join(script_dir,"STFTR.config.json")
    with open(p,"r",encoding="utf-8") as f: cfg=json.load(f)
    cfg["_cfg_path"]=p
    return cfg

def resolve_model_path(cfg: dict) -> str|None:
    if cfg.get("model_path"):
        return cfg.get("model_path")
    root = cfg.get("model_root")
    bank = cfg.get("model_bank")
    fname = cfg.get("model_file", "best.pt")
    if not root or not bank:
        return None
    root = str(root).replace("\\","/")
    bank = str(bank).strip()
    return os.path.abspath(os.path.join(root, bank, fname))

def read_wav_pcm(path: str) -> Tuple[np.ndarray, int]:
    with wave.open(path,"rb") as wf:
        sr=wf.getframerate(); ch=wf.getnchannels(); sw=wf.getsampwidth()
        n=wf.getnframes(); raw=wf.readframes(n)
    if sw==1:
        x=np.frombuffer(raw,dtype=np.uint8).astype(np.float32); x=(x-128.0)/128.0
    elif sw==2:
        x=np.frombuffer(raw,dtype=np.int16).astype(np.float32)/32768.0
    elif sw==3:
        b=np.frombuffer(raw,dtype=np.uint8).reshape(-1,3)
        x=(b[:,0].astype(np.int32)|(b[:,1].astype(np.int32)<<8)|(b[:,2].astype(np.int32)<<16))
        x=np.where(x>=2**23,x-2**24,x).astype(np.float32)/(2**23)
    elif sw==4:
        x=np.frombuffer(raw,dtype=np.int32).astype(np.float32)/2147483648.0
    else:
        raise ValueError(f"Unsupported sampwidth={sw}")
    if ch>1: x=x.reshape(-1,ch).mean(axis=1)
    return np.nan_to_num(x).astype(np.float32), sr

def write_wav_pcm16(path: str, x: np.ndarray, sr: int):
    x=np.clip(x,-1.0,1.0)
    y=(x*32767.0).astype(np.int16).tobytes()
    with wave.open(path,"wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes(y)

def rms(x: np.ndarray)->float:
    return float(np.sqrt(np.mean(x*x)+1e-12))

class ConvBlock(nn.Module):
    def __init__(self,c_in:int,c_out:int,k:int=3):
        super().__init__(); p=k//2
        self.net=nn.Sequential(
            nn.Conv2d(c_in,c_out,k,padding=p),
            nn.GroupNorm(num_groups=min(8,c_out), num_channels=c_out),
            nn.SiLU(),
            nn.Conv2d(c_out,c_out,k,padding=p),
            nn.GroupNorm(num_groups=min(8,c_out), num_channels=c_out),
            nn.SiLU(),
        )
    def forward(self,x): return self.net(x)

class Down(nn.Module):
    def __init__(self,c_in:int,c_out:int):
        super().__init__()
        self.block=ConvBlock(c_in,c_out); self.pool=nn.AvgPool2d(2,2)
    def forward(self,x):
        h=self.block(x); return self.pool(h),h

class Up(nn.Module):
    def __init__(self,c_in:int,c_skip:int,c_out:int):
        super().__init__()
        self.up=nn.Upsample(scale_factor=2,mode="nearest")
        self.block=ConvBlock(c_in+c_skip,c_out)
    def forward(self,x,skip):
        x=self.up(x)
        dh=skip.shape[-2]-x.shape[-2]; dw=skip.shape[-1]-x.shape[-1]
        if dh or dw:
            x=F.pad(x,(0,max(0,dw),0,max(0,dh)))
            x=x[:,:,:skip.shape[-2],:skip.shape[-1]]
        return self.block(torch.cat([x,skip],dim=1))

class STFTTimbreUNet(nn.Module):
    def __init__(self, base:int=32):
        super().__init__()
        self.inp=ConvBlock(1,base)
        self.d1=Down(base,base*2)
        self.d2=Down(base*2,base*4)
        self.mid=ConvBlock(base*4,base*4)
        self.u2=Up(base*4,base*4,base*2)
        self.u1=Up(base*2,base*2,base)
        self.out=nn.Conv2d(base,1,1)
    def forward(self,x):
        x0=self.inp(x)
        x1,s1=self.d1(x0)
        x2,s2=self.d2(x1)
        xm=self.mid(x2)
        xu2=self.u2(xm,s2)
        xu1=self.u1(xu2,s1)
        return self.out(xu1)

@torch.no_grad()
def apply_model(wav_in: str, wav_out_tmp: str, model_path: str, strength: float, n_fft:int, hop:int, win:int, base:int, device:str, rms_match:bool, log_file: str):
    s=float(max(0.0,min(1.0,strength)))
    x,sr=read_wav_pcm(wav_in)
    rin=rms(x)
    if rin<1e-6 or s<=0.0:
        shutil.copyfile(wav_in, wav_out_tmp); return

    dev=torch.device(device)
    model=STFTTimbreUNet(base=base).to(dev)
    ckpt=torch.load(model_path,map_location=dev)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    xt=torch.from_numpy(x).to(dev,dtype=torch.float32)
    window=torch.hann_window(win, device=dev, dtype=torch.float32)
    spec=torch.stft(xt,n_fft=n_fft,hop_length=hop,win_length=win,window=window,center=True,pad_mode="reflect",return_complex=True)
    mag=spec.abs().clamp_min(1e-7); phase=spec/mag
    logmag=torch.log(mag); mu=logmag.mean(); logmag_n=logmag-mu

    Fbins,T=logmag_n.shape
    padF=(4-(Fbins%4))%4; padT=(4-(T%4))%4
    inp=logmag_n.unsqueeze(0).unsqueeze(0)
    if padF or padT:
        inp=F.pad(inp,(0,padT,0,padF),mode="replicate")
    pred=model(inp).squeeze(0).squeeze(0)[:Fbins,:T]

    out_logmag_n=(1.0-s)*logmag_n+s*pred
    out_mag=torch.exp(out_logmag_n+mu).clamp_min(1e-7)
    out_spec=out_mag*phase
    y=torch.istft(out_spec,n_fft=n_fft,hop_length=hop,win_length=win,window=window,center=True,length=xt.shape[0])
    y=y.clamp(-1.0,1.0).detach().cpu().numpy().astype(np.float32)

    if rms_match:
        rout=rms(y)
        if rout>0: y=y*(rin/rout)

    append_log(log_file, f"[{ts()}] apply bank={os.path.basename(os.path.dirname(model_path))} strength={s:.3f} RMS in={rin:.6f} out={rms(y):.6f}\n")
    write_wav_pcm16(wav_out_tmp, y, sr)

def main():
    script_dir=os.path.dirname(os.path.abspath(__file__))
    cfg=load_cfg(script_dir)
    log_file=cfg.get("log_file", os.path.join(script_dir,"timbre_wrapper_stft_log.txt"))
    real=cfg.get("real_resampler")
    strength=float(cfg.get("strength",0.6))
    device=str(cfg.get("device","cpu"))
    rms_match=bool(cfg.get("rms_match",True))
    n_fft=int(cfg.get("n_fft",1024)); hop=int(cfg.get("hop",256)); win=int(cfg.get("win",1024)); base=int(cfg.get("base",32))

    model_path=resolve_model_path(cfg)
    argv=sys.argv[1:]
    append_log(log_file, f"\\n[{ts()}] WRAPPER STFT v2 bank={cfg.get('model_bank')} model={model_path} argv={argv}\\n")

    if not real:
        append_log(log_file, f"[{ts()}] ERROR: real_resampler not set\\n"); return 2

    proc=subprocess.run([os.path.abspath(real), *argv], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    append_log(log_file, f"[{ts()}] RESAMPLER rc={proc.returncode}\\n")
    if proc.returncode!=0:
        try: append_log(log_file, proc.stderr.decode("utf-8",errors="replace")[:2000]+"\\n")
        except Exception: pass
        return proc.returncode

    if len(argv)<2: return 3
    out_wav=argv[1]
    if not os.path.exists(out_wav) or os.path.getsize(out_wav)<44:
        append_log(log_file, f"[{ts()}] ERROR: out_wav missing/empty: {out_wav}\\n"); return 4

    if (not model_path) or (not os.path.exists(model_path)) or strength<=0.0:
        append_log(log_file, f"[{ts()}] skipping correction. model_path={model_path}\\n"); return 0

    tmp_dir=os.path.dirname(out_wav) or script_dir
    fd,tmp_path=tempfile.mkstemp(prefix="stft_timbre_", suffix=".wav", dir=tmp_dir)
    os.close(fd)
    try:
        apply_model(out_wav, tmp_path, model_path, strength, n_fft, hop, win, base, device, rms_match, log_file)
        if os.path.getsize(tmp_path)>=44:
            shutil.move(tmp_path, out_wav)
            append_log(log_file, f"[{ts()}] replaced out_wav\\n")
    except Exception as ex:
        append_log(log_file, f"[{ts()}] ERROR: correction failed: {type(ex).__name__}: {ex}\\n")
        try: os.remove(tmp_path)
        except Exception: pass
        return 0
    return 0

if __name__=="__main__":
    raise SystemExit(main())
