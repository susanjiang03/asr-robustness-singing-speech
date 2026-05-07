"""
Audio processing utilities for voice segmentation and modification
"""




def shorten_audio_segment(seg: np.ndarray, keep_ratio: float, min_samples: int = 400):
    """Shorten audio segment using time stretching"""
    if len(seg) < min_samples:
        return seg

    # librosa time_stretch: rate > 1 speeds up (shortens)
    rate = 1.0 / keep_ratio
    try:
        result = librosa.effects.time_stretch(seg, rate=rate)
        # Check if result is valid
        if result is None:
            print(f"       ⚠️  time_stretch returned None, using original segment")
            return seg
        if not isinstance(result, np.ndarray):
            print(f"       ⚠️  time_stretch returned invalid type, using original segment")
            return seg
        if len(result) == 0:
            print(f"       ⚠️  time_stretch returned empty array, using original segment")
            return seg
        return result
    except Exception as e:
        print(f"       ⚠️  time_stretch failed: {e}, using original segment")
        return seg


def process_elongated_segments(
    input_path: str, 
    output_path: str,
    target_sr: int = 16000,
    min_elongated_sec: float = 0.50,
    shorten_factor: float = 0.75,
    min_segment_samples: int = 400
):
    """Process audio file to shorten elongated voiced segments"""
    
    # Load audio
    y, sr = sf.read(input_path)
    
    if len(y.shape) > 1:
        y = y.mean(axis=1)
    
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    
    # Find voiced segments
    segments = find_voiced_segments(y, target_sr)
    
    # Process segments
    pieces = []
    last = 0
    modified_count = 0

    for start, end, duration in segments:
        if start > last:
            pieces.append(y[last:start])

        voiced_seg = y[start:end]

        if duration >= min_elongated_sec:
            voiced_seg = shorten_audio_segment(voiced_seg, shorten_factor, min_segment_samples)
            modified_count += 1

        pieces.append(voiced_seg)
        last = end

    if last < len(y):
        pieces.append(y[last:])

    y_out = np.concatenate(pieces) if pieces else y
    
    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, y_out, target_sr)

    return {
        "input_path": input_path,
        "output_path": output_path,
        "original_duration_sec": len(y) / target_sr,
        "new_duration_sec": len(y_out) / target_sr,
        "modified_segments": modified_count,
    }
