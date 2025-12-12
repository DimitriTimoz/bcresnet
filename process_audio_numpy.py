import os
import shutil
import random
import glob
import wave
import sys
import struct
import math
import argparse
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import numpy as np
except ImportError:
    print("Error: numpy is required. Please install it with 'pip install numpy'")
    sys.exit(1)

# Configuration
SRC_DIR = os.path.abspath("donut")
DST_DIR = os.path.abspath("donut_processed")
BAR_NOISES_SRC = os.path.abspath("bar-noises.wav")
BAR_NOISES_16K = os.path.abspath("bar-noises-16k.wav")

def ensure_bar_noises_16k():
    """Convert bar noises to 16k mono pcm_s16le if not exists."""
    if not os.path.exists(BAR_NOISES_SRC):
        print(f"Error: {BAR_NOISES_SRC} not found.")
        sys.exit(1)
        
    print(f"Pre-converting {BAR_NOISES_SRC} to 16k mono...")
    cmd = [
        "ffmpeg", "-y",
        "-i", BAR_NOISES_SRC,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        BAR_NOISES_16K
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Conversion complete.")

def read_wav(path):
    """Read a wav file return (rate, data_numpy_int16)."""
    with wave.open(path, 'rb') as wf:
        n_channels = wf.getnchannels()
        rate = wf.getframerate()
        n_frames = wf.getnframes()
        data = wf.readframes(n_frames)
        
        # Convert to numpy
        arr = np.frombuffer(data, dtype=np.int16)
        
        # If stereo, just take first channel for simplicity (since we work in mono 16k context)
        # But target files are mono 16k usually.
        if n_channels > 1:
             arr = arr.reshape(-1, n_channels)
             arr = arr[:, 0] # Take left channel
             
    return rate, arr

def write_wav(path, rate, data):
    """Write numpy int16 array to wav."""
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2) # 16-bit
        wf.setframerate(rate)
        wf.writeframes(data.tobytes())

# Global to hold the noise array in workers? 
# Multiprocessing doesn't share memory easily without copy-on-write. 
# Linux fork should handle it efficiently if we load BEFORE forking? 
# No, ProcessPoolExecutor usage might spawn before loading if not careful.
# But "fork" is default on Linux, so if we load global data before submitting, it might work?
# Actually, 115MB is small. We can just load it in every worker or rely on OS paging.
# Better pattern: Initialize worker with the data.

bar_noise_data = None

def init_worker(noise_path):
    global bar_noise_data
    # Read the noise file once per worker
    try:
        _, bar_noise_data = read_wav(noise_path)
    except Exception as e:
        print(f"Worker failed to load noise: {e}")
        bar_noise_data = np.zeros(0, dtype=np.int16)

def process_file_numpy(file_path, group_type):
    try:
        rate, audio = read_wav(file_path)
        
        # Ensure float processing
        audio_f = audio.astype(np.float32)
        
        # Main Volume: Normal(1.0, 0.15), clamp 0.1-2.0
        main_vol = random.gauss(1.0, 0.15)
        main_vol = max(0.1, min(2.0, main_vol))
        
        audio_f *= main_vol
        
        if group_type == 1: # Mix Bar Noises
            # Noise Volume for Bar: High gain requested
            # Normal(10.0, 2.0)
            noise_vol = random.gauss(3.0, 1.0)
            noise_vol = max(1.0, min(6.0, noise_vol))
            
            if bar_noise_data is not None and len(bar_noise_data) > 0:
                noise_len = len(audio_f)
                bar_len = len(bar_noise_data)
                
                if bar_len > noise_len:
                    start = random.randint(0, bar_len - noise_len)
                    noise_chunk = bar_noise_data[start:start+noise_len].astype(np.float32)
                else:
                    # Pad or tile? file is typically small (<1s), bar noise is 1hr.
                    # Should not happen.
                    noise_chunk = np.zeros_like(audio_f)
                
                audio_f += (noise_chunk * noise_vol)
            
        elif group_type == 2: # White Noise
            # Noise Amp: Normal(0.2, 0.05)
            # Since audio is int16 (scale ~32768), 0.2 means nothing if applied to 1.0 scale.
            # We are working in full scale.
            # Typical speech peak is ~10000-20000.
            # We want noise to be audible.
            # 0.2 * 32768 = ~6500. This is very audible.
            
            amp_factor = random.gauss(0.2, 0.05)
            amp_factor = max(0.01, amp_factor)
            
            noise = np.random.normal(0, amp_factor * 32768, len(audio_f)).astype(np.float32)
            audio_f += noise

        elif group_type == 3: # Unchanged (Vol Mod only)
            pass

        # Clip and convert back
        np.clip(audio_f, -32768, 32767, out=audio_f)
        audio_int16 = audio_f.astype(np.int16)
        
        write_wav(file_path, rate, audio_int16)
        return True
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    if not os.path.exists(SRC_DIR):
        print(f"Error: Source directory '{SRC_DIR}' does not exist.")
        sys.exit(1)
        
    # Pre-step: Ensure bar noise is 16k
    ensure_bar_noises_16k()

    print(f"Step 1: Copying '{SRC_DIR}' to '{DST_DIR}' (using shutil)...")
    if os.path.exists(DST_DIR):
        shutil.rmtree(DST_DIR)
    shutil.copytree(SRC_DIR, DST_DIR)

    print("Step 2: Scanning for .wav files...")
    wav_files = glob.glob(os.path.join(DST_DIR, "**", "*.wav"), recursive=True)
    random.seed(42)
    random.shuffle(wav_files)
    
    total = len(wav_files)
    n_third = total // 3
    
    group1 = wav_files[:n_third]
    group2 = wav_files[n_third:2*n_third]
    group3 = wav_files[2*n_third:]
    
    print(f"Total: {total}. G1: {len(group1)}, G2: {len(group2)}, G3: {len(group3)}")
    
    max_workers = os.cpu_count() or 4
    # Lower workers slightly
    max_workers = max(1, max_workers)

    print(f"Starting processing with {max_workers} workers (numpy)...")
    
    futures = []
    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=(BAR_NOISES_16K,)) as executor:
        for f in group1:
            futures.append(executor.submit(process_file_numpy, f, 1))
        for f in group2:
            futures.append(executor.submit(process_file_numpy, f, 2))
        for f in group3:
            futures.append(executor.submit(process_file_numpy, f, 3))
            
        count = 0
        total_tasks = len(futures)
        for fut in as_completed(futures):
            count += 1
            if count % 200 == 0:
                 print(f"Progress: {count}/{total_tasks} ({(count/total_tasks)*100:.1f}%)", flush=True)

    print("Done.")

if __name__ == "__main__":
    main()
