#!/usr/bin/env python3
"""
Test script to verify donut.zip download and extraction.
Run: python test_donut_download.py
"""

import os
import tempfile
import shutil

def test_donut_download():
    """Test downloading and extracting the donut class."""
    from utils import DownloadDonutClass, DONUT_URL
    
    print("=" * 50)
    print("Testing Donut Class Download")
    print("=" * 50)
    print(f"URL: {DONUT_URL}")
    
    # Create a temporary directory for testing
    test_dir = tempfile.mkdtemp(prefix="donut_test_")
    print(f"Test directory: {test_dir}")
    
    try:
        # Download and extract using the actual function
        DownloadDonutClass(test_dir)
        
        # Check if donut directory exists
        donut_dir = os.path.join(test_dir, "donut")
        if not os.path.isdir(donut_dir):
            print(f"[ERROR] Donut directory not found at {donut_dir}")
            print("  Check that donut.zip contains a 'donut/' folder")
            return False
        
        # List files in donut directory
        files = os.listdir(donut_dir)
        wav_files = [f for f in files if f.endswith('.wav')]
        
        print(f"\n[SUCCESS] Donut class downloaded and extracted!")
        print(f"  Location: {donut_dir}")
        print(f"  Total files: {len(files)}")
        print(f"  WAV files: {len(wav_files)}")
        
        if wav_files:
            print(f"\n  Sample files:")
            for f in wav_files[:5]:
                filepath = os.path.join(donut_dir, f)
                size = os.path.getsize(filepath)
                print(f"    - {f} ({size} bytes)")
            if len(wav_files) > 5:
                print(f"    ... and {len(wav_files) - 5} more")
        else:
            print("[WARNING] No .wav files found in donut directory!")
            return False
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup test directory
        print(f"\nCleaning up test directory: {test_dir}")
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    success = test_donut_download()
    print("\n" + "=" * 50)
    if success:
        print("TEST PASSED - Donut download is working!")
    else:
        print("TEST FAILED - Check the error messages above")
    print("=" * 50)
    exit(0 if success else 1)
