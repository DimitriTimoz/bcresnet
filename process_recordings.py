import argparse
import os
import torch
import torchaudio

def main():
    parser = argparse.ArgumentParser(description="Extract 1s chunks from audio at periodic intervals.")
    parser.add_argument('filename', help='Path to the input audio file (e.g., bad.mp3)')
    parser.add_argument('--offset', type=float, default=1.078, help='Initial time offset in seconds (default: 1.078)')
    parser.add_argument('--interval', type=float, default=1.5, help='Time interval between chunks in seconds (default: 1.5)')
    parser.add_argument('--output-dir', type=str, help='Directory to save chunks (default: <filename>_chunks)')

    args = parser.parse_args()

    if not os.path.exists(args.filename):
        print(f"Error: File '{args.filename}' not found.")
        return

    # Determine output directory
    if args.output_dir is None:
        base_name = os.path.basename(args.filename)
        name_no_ext = os.path.splitext(base_name)[0]
        args.output_dir = f"{name_no_ext}_chunks"

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")
    else:
        print(f"Using output directory: {args.output_dir}")

    print(f"Loading {args.filename}...")
    try:
        # Load the entire audio file
        waveform, sample_rate = torchaudio.load(args.filename)
    except Exception as e:
        print(f"Failed to load audio file: {e}")
        return

    channels, total_samples = waveform.shape
    duration_sec = total_samples / sample_rate
    print(f"Audio Duration: {duration_sec:.2f}s | Sample Rate: {sample_rate}Hz | Channels: {channels}")

    k = 0
    chunks_extracted = 0
    chunk_samples = int(1.0 * sample_rate)  # 1 second of samples

    while True:
        # Calculate start time t = offset + k * interval
        start_time = args.offset + (k * args.interval)
        
        # We need a full 1s chunk
        if start_time + 1.0 > duration_sec:
            break

        start_sample = int(start_time * sample_rate)
        # End sample is start + 1s worth of samples
        end_sample = start_sample + chunk_samples
        
        # Extract chunk
        chunk = waveform[:, start_sample:end_sample]

        # Save chunk
        output_filename = os.path.join(args.output_dir, f"chunk_{k:04d}.wav")
        torchaudio.save(output_filename, chunk, sample_rate)
        
        if k % 10 == 0:
             print(f"Saved {output_filename} (t={start_time:.3f}s)")
        
        k += 1
        chunks_extracted += 1

    print(f"Done. Extracted {chunks_extracted} chunks to '{args.output_dir}'.")

if __name__ == "__main__":
    main()
