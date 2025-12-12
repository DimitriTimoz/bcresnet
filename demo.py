#!/usr/bin/env python3
"""
Real-time keyword spotting demo using BCResNet.
Continuously listens to the microphone and displays detected words.

Usage:
    python demo.py --model bcresnet_tau8.0_v2_acc95.50.pth
    python demo.py --model bcresnet_tau8.0_v2_acc95.50.pth --threshold 0.7
"""

import argparse
import sys
import time
import queue
import threading

import numpy as np
import torch
import torchaudio

# Try to import sounddevice for microphone input
try:
    import sounddevice as sd
except ImportError:
    print("Please install sounddevice: pip install sounddevice")
    sys.exit(1)

from bcresnet import BCResNets
from utils import label_dict

# Reverse label dict: index -> name
idx_to_label = {v: k for k, v in label_dict.items()}

# Audio parameters (must match training)
SAMPLE_RATE = 16000
DURATION = 1.0  # 1 second window
CHUNK_SIZE = int(SAMPLE_RATE * 0.1)  # 100ms chunks for responsiveness
WINDOW_SIZE = int(SAMPLE_RATE * DURATION)

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def load_model(model_path, tau, device):
    """Load a trained BCResNet model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Try to infer base_c from checkpoint weights
    # cnn_head.0.weight has shape [base_c * 2, 1, 5, 5]
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Infer base_c from cnn_head.0.weight shape
    cnn_head_shape = state_dict['cnn_head.0.weight'].shape[0]  # This is base_c * 2
    inferred_base_c = cnn_head_shape // 2
    print(f"Inferred base_c={inferred_base_c} from checkpoint (cnn_head channels: {cnn_head_shape})")
    
    model = BCResNets(base_c=inferred_base_c)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {model_path}")
        if 'test_acc' in checkpoint:
            print(f"  Test accuracy: {checkpoint['test_acc']:.2f}%")
        if 'valid_acc' in checkpoint:
            print(f"  Valid accuracy: {checkpoint['valid_acc']:.2f}%")
        if 'tau' in checkpoint:
            print(f"  Tau: {checkpoint['tau']}")
    else:
        model.load_state_dict(checkpoint)
        print(f"Loaded model weights from {model_path}")
    
    model.to(device)
    model.eval()
    return model


def create_mel_transform(device):
    """Create mel spectrogram transform matching training."""
    n_fft = 480
    hop_length = 160
    n_mels = 40
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    ).to(device)
    
    return mel_transform


def preprocess_audio(audio, mel_transform, device):
    """Convert raw audio to mel spectrogram."""
    # Ensure correct length (1 second)
    if len(audio) < WINDOW_SIZE:
        audio = np.pad(audio, (0, WINDOW_SIZE - len(audio)))
    elif len(audio) > WINDOW_SIZE:
        audio = audio[:WINDOW_SIZE]
    
    # Convert to tensor
    waveform = torch.from_numpy(audio).float().unsqueeze(0).to(device)  # [1, samples]
    
    # Compute mel spectrogram
    mel = mel_transform(waveform)
    mel = (mel + 1e-6).log()
    mel = mel.unsqueeze(0)  # [1, 1, n_mels, time]
    
    return mel


def get_prediction(model, mel, threshold=0.5):
    """Get model prediction with confidence. Returns all probabilities."""
    with torch.no_grad():
        output = model(mel)
        probabilities = torch.softmax(output, dim=-1)
        all_probs = probabilities.squeeze().cpu().numpy()
        confidence, predicted_idx = torch.max(probabilities, dim=-1)
        
        confidence = confidence.item()
        predicted_idx = predicted_idx.item()
        predicted_label = idx_to_label[predicted_idx]
        
        # Return label (or None), confidence, and all probabilities
        if confidence >= threshold:
            return predicted_label, confidence, all_probs
        else:
            return None, confidence, all_probs


def print_detection(label, confidence, show_all=False):
    """Print detected word with color based on class."""
    # Skip silence and unknown unless show_all
    if not show_all and label in ['_silence_', '_unknown_']:
        return
    
    # Color based on confidence
    if confidence > 0.9:
        color = Colors.GREEN
    elif confidence > 0.7:
        color = Colors.YELLOW
    else:
        color = Colors.RED
    
    # Special color for donut
    if label == 'donut':
        color = Colors.CYAN + Colors.BOLD
    
    timestamp = time.strftime("%H:%M:%S")
    bar_length = int(confidence * 20)
    bar = '█' * bar_length + '░' * (20 - bar_length)
    
    print(f"{Colors.BLUE}[{timestamp}]{Colors.ENDC} "
          f"{color}{label:12s}{Colors.ENDC} "
          f"[{bar}] {confidence*100:5.1f}%")


def audio_callback(indata, frames, time_info, status, audio_queue):
    """Callback for audio stream."""
    if status:
        print(f"Audio status: {status}")
    audio_queue.put(indata.copy())


def run_demo(model_path, tau, threshold, device_id, show_all):
    """Main demo loop with real-time donut probability graph."""
    import matplotlib.pyplot as plt
    from collections import deque
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(model_path, tau, device)
    mel_transform = create_mel_transform(device)
    
    # Audio buffer (circular buffer of 1 second)
    audio_buffer = np.zeros(WINDOW_SIZE, dtype=np.float32)
    audio_queue = queue.Queue()
    
    # History for donut probability graph (last 100 samples ~ 5 seconds)
    history_len = 100
    donut_history = deque([0.0] * history_len, maxlen=history_len)
    time_history = deque(range(-history_len, 0), maxlen=history_len)
    
    # Get donut class index
    donut_idx = label_dict.get('donut', 12)
    
    # Print available classes
    print(f"\n{Colors.HEADER}Detecting keywords:{Colors.ENDC}")
    keywords = [k for k in label_dict.keys() if k not in ['_silence_', '_unknown_']]
    print(", ".join(keywords))
    print(f"\nConfidence threshold: {threshold*100:.0f}%")
    print(f"\n{Colors.BOLD}Listening... (Close graph window or Ctrl+C to stop){Colors.ENDC}\n")
    print("-" * 50)
    
    # Setup matplotlib for real-time plotting
    plt.ion()  # Interactive mode
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    fig.suptitle('BCResNet Keyword Spotting - Real-time Demo', fontsize=14, fontweight='bold')
    
    # Donut probability line plot
    line_donut, = ax1.plot(list(time_history), list(donut_history), 'b-', linewidth=2, label='Donut probability')
    ax1.axhline(y=threshold, color='r', linestyle='--', alpha=0.7, label=f'Threshold ({threshold:.0%})')
    ax1.set_ylim(0, 1)
    ax1.set_xlim(-history_len, 0)
    ax1.set_ylabel('Probability')
    ax1.set_xlabel('Time (samples)')
    ax1.set_title('🍩 Donut Detection Probability')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.fill_between(list(time_history), 0, list(donut_history), alpha=0.3)
    
    # Bar chart for all classes
    class_names = [name for name, idx in sorted(label_dict.items(), key=lambda x: x[1])]
    bars = ax2.bar(class_names, [0] * len(class_names), color='steelblue')
    bars[donut_idx].set_color('orange')  # Highlight donut
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Probability')
    ax2.set_title('All Classes Probabilities')
    ax2.axhline(y=threshold, color='r', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Text annotation for detected word
    detection_text = ax1.text(0.98, 0.95, '', transform=ax1.transAxes, 
                               fontsize=16, fontweight='bold', 
                               ha='right', va='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Start audio stream
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=CHUNK_SIZE,
            device=device_id,
            callback=lambda *args: audio_callback(*args, audio_queue)
        ):
            last_detection = None
            last_detection_time = 0
            cooldown = 0.5  # Seconds between same detections
            sample_count = 0
            
            while plt.fignum_exists(fig.number):
                # Get new audio chunks
                while not audio_queue.empty():
                    chunk = audio_queue.get().flatten()
                    # Shift buffer and add new chunk
                    audio_buffer = np.roll(audio_buffer, -len(chunk))
                    audio_buffer[-len(chunk):] = chunk
                
                # Preprocess and predict
                mel = preprocess_audio(audio_buffer, mel_transform, device)
                label, confidence, all_probs = get_prediction(model, mel, threshold)
                
                # Update donut history
                donut_prob = all_probs[donut_idx]
                donut_history.append(donut_prob)
                sample_count += 1
                
                # Update plots every few iterations for performance
                if sample_count % 2 == 0:
                    # Update donut line plot
                    line_donut.set_ydata(list(donut_history))
                    
                    # Update fill
                    ax1.collections.clear()
                    ax1.fill_between(list(time_history), 0, list(donut_history), alpha=0.3, color='blue')
                    ax1.axhline(y=threshold, color='r', linestyle='--', alpha=0.7)
                    
                    # Update bar chart
                    for bar, prob in zip(bars, all_probs):
                        bar.set_height(prob)
                    
                    # Update detection text
                    if label is not None:
                        if label == 'donut':
                            detection_text.set_text(f'🍩 DONUT! ({confidence:.0%})')
                            detection_text.set_color('darkorange')
                        else:
                            detection_text.set_text(f'Detected: {label} ({confidence:.0%})')
                            detection_text.set_color('darkgreen')
                    else:
                        detection_text.set_text('')
                    
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()
                
                # Print detection (with cooldown to avoid spam)
                current_time = time.time()
                if label is not None:
                    if label != last_detection or (current_time - last_detection_time) > cooldown:
                        print_detection(label, confidence, show_all)
                        last_detection = label
                        last_detection_time = current_time
                
                # Small sleep to reduce CPU usage
                time.sleep(0.05)
                
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Stopped.{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.RED}Error: {e}{Colors.ENDC}")
        raise
    finally:
        plt.close('all')


def list_audio_devices():
    """List available audio input devices."""
    print("\nAvailable audio input devices:")
    print("-" * 40)
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            default = " (default)" if i == sd.default.device[0] else ""
            print(f"  [{i}] {dev['name']}{default}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Real-time keyword spotting demo")
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (.pth file)')
    parser.add_argument('--tau', type=float, default=8,
                        help='Model tau value (default: 8)')
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='Confidence threshold (0-1, default: 0.6)')
    parser.add_argument('--device', type=int, default=None,
                        help='Audio input device ID (use --list-devices to see available)')
    parser.add_argument('--list-devices', action='store_true',
                        help='List available audio input devices and exit')
    parser.add_argument('--show-all', action='store_true',
                        help='Show silence and unknown detections too')
    
    args = parser.parse_args()
    
    if args.list_devices:
        list_audio_devices()
        return
    
    print(f"\n{Colors.HEADER}{'='*50}{Colors.ENDC}")
    print(f"{Colors.BOLD}  BCResNet Keyword Spotting Demo{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*50}{Colors.ENDC}\n")
    
    run_demo(
        model_path=args.model,
        tau=args.tau,
        threshold=args.threshold,
        device_id=args.device,
        show_all=args.show_all
    )


if __name__ == "__main__":
    main()
