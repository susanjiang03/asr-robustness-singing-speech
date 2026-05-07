"""
Advanced audio processing with multiple parameter configurations
"""
import os
from pathlib import Path
from typing import List, Dict, Any

import librosa
import numpy as np
import soundfile as sf

from .voice_segmentation import find_voiced_segments, shorten_audio_segment


class AudioVariantProcessor:
    """Process audio with multiple parameter configurations for comparison"""
    
    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr
    
    def create_audio_variants(
        self,
        input_path: str,
        output_dir: str,
        variant_configs: List[Dict[str, Any]]
    ) -> Dict[str, Dict]:
        """Create multiple audio variants with different processing parameters"""
        
        # Load original audio once
        print(f"    📁 Loading audio: {Path(input_path).name}")
        y, sr = sf.read(input_path)
        
        if len(y.shape) > 1:
            y = y.mean(axis=1)
            print(f"    🔄 Converted to mono: {y.shape}")
        
        if sr != self.target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sr)
            print(f"    🔄 Resampled: {sr}Hz → {self.target_sr}Hz")
        
        original_duration = len(y) / self.target_sr
        print(f"    ⏱️  Original duration: {original_duration:.2f}s")
        
        results = {}
        
        for i, config in enumerate(variant_configs):
            variant_name = config["name"]
            print(f"    🎛️  [{i+1}/{len(variant_configs)}] Processing {variant_name}...")
            print(f"       Parameters: min_elongated={config.get('min_elongated_sec', 0.5)}s, "
                  f"shorten_factor={config.get('shorten_factor', 0.75)}, "
                  f"frame_length={config.get('frame_length', 1024)}")
            
            # Create variant-specific output path
            output_path = os.path.join(output_dir, f"{variant_name}_{Path(input_path).name}")
            
            # Process with variant-specific parameters
            try:
                variant_result = self.process_with_config(
                    y, input_path, output_path, config
                )
                variant_result["config"] = config
                results[variant_name] = variant_result
                
                # Log processing results
                reduction_pct = variant_result.get("duration_reduction_pct", 0)
                modified_segments = variant_result.get("modified_segments", 0)
                total_segments = variant_result.get("total_segments", 0)
                
                print(f"       ✅ Completed: {reduction_pct:.1f}% duration reduction, "
                      f"{modified_segments}/{total_segments} segments modified")
                print(f"       📁 Saved to: {Path(output_path).name}")
                
            except Exception as e:
                print(f"       ❌ Error processing {variant_name}: {e}")
                # Add failed result info
                results[variant_name] = {
                    "input_path": input_path,
                    "output_path": output_path,
                    "error": str(e),
                    "config": config
                }
        
        print(f"    🎉 All variants processed for {Path(input_path).name}")
        return results
    
    def process_with_config(
        self,
        y: np.ndarray,
        input_path: str,
        output_path: str,
        config: Dict[str, Any]
    ) -> Dict:
        """Process audio with specific configuration"""
        
        # Extract parameters
        min_elongated_sec = config.get("min_elongated_sec", 0.5)
        shorten_factor = config.get("shorten_factor", 0.75)
        min_segment_samples = config.get("min_segment_samples", 400)
        frame_length = config.get("frame_length", 1024)
        hop_length = config.get("hop_length", 256)
        
        print(f"       🔍 Finding voiced segments with frame_length={frame_length}, hop_length={hop_length}...")
        
        # Find voiced segments with custom parameters
        segments = find_voiced_segments(y, self.target_sr, frame_length, hop_length)
        
        # Ensure segments is not None or empty
        if segments is None:
            segments = []
        
        # Process segments
        pieces = []
        last = 0
        modified_count = 0
        
        # Handle empty segments list
        if not segments:
            print(f"       ⚠️  No voiced segments found, using original audio")
            return {
                "input_path": input_path,
                "output_path": output_path,
                "original_duration_sec": len(y) / self.target_sr,
                "new_duration_sec": len(y) / self.target_sr,
                "duration_reduction_sec": 0.0,
                "duration_reduction_pct": 0.0,
                "modified_segments": 0,
                "total_segments": 0,
                "modification_rate": 0.0
            }
        
        print(f"       📊 Found {len(segments)} voiced segments")
        
        elongated_count = 0
        for start, end, duration in segments:
            if start > last:
                pieces.append(y[last:start])
            
            voiced_seg = y[start:end]
            
            if duration >= min_elongated_sec:
                elongated_count += 1
                print(f"       ✂️  Shortening segment: {duration:.2f}s → {duration*shorten_factor:.2f}s")
                try:
                    voiced_seg = shorten_audio_segment(voiced_seg, shorten_factor, min_segment_samples)
                    modified_count += 1
                except Exception as e:
                    print(f"       ⚠️  Error shortening segment: {e}")
            
            pieces.append(voiced_seg)
            last = end
        
        if last < len(y):
            pieces.append(y[last:])
        
        y_out = np.concatenate(pieces) if pieces else y
        
        print(f"       📝 Modified {modified_count}/{elongated_count} elongated segments")
        
        # Save output
        print(f"       💾 Saving processed audio...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, y_out, self.target_sr)
        
        return {
            "input_path": input_path,
            "output_path": output_path,
            "original_duration_sec": len(y) / self.target_sr,
            "new_duration_sec": len(y_out) / self.target_sr,
            "duration_reduction_sec": (len(y) - len(y_out)) / self.target_sr,
            "duration_reduction_pct": (1 - len(y_out) / len(y)) * 100,
            "modified_segments": modified_count,
            "total_segments": len(segments),
            "modification_rate": modified_count / len(segments) if segments else 0
        }


def get_default_variant_configs() -> List[Dict[str, Any]]:
    """Get default audio variant configurations for comparison"""
    
    return [
        {
            "name": "conservative",
            "description": "Conservative shortening - minimal changes",
            "min_elongated_sec": 0.8,  # Only very long segments
            "shorten_factor": 0.9,     # Keep 90% of duration
            "min_segment_samples": 800,
            "frame_length": 1024,
            "hop_length": 256
        },
        {
            "name": "moderate",
            "description": "Moderate shortening - balanced approach",
            "min_elongated_sec": 0.5,  # Medium-length segments
            "shorten_factor": 0.75,    # Keep 75% of duration
            "min_segment_samples": 400,
            "frame_length": 1024,
            "hop_length": 256
        },
        {
            "name": "aggressive",
            "description": "Aggressive shortening - maximum reduction",
            "min_elongated_sec": 0.3,  # Even shorter segments
            "shorten_factor": 0.5,     # Keep 50% of duration
            "min_segment_samples": 200,
            "frame_length": 1024,
            "hop_length": 256
        },
        {
            "name": "ultra_aggressive",
            "description": "Ultra aggressive - extreme reduction",
            "min_elongated_sec": 0.2,  # Very short threshold
            "shorten_factor": 0.3,     # Keep only 30%
            "min_segment_samples": 100,
            "frame_length": 512,        # Higher resolution analysis
            "hop_length": 128
        },
        {
            "name": "pitch_sensitive",
            "description": "Pitch-sensitive processing",
            "min_elongated_sec": 0.4,
            "shorten_factor": 0.7,
            "min_segment_samples": 300,
            "frame_length": 2048,       # Better pitch resolution
            "hop_length": 512
        },
        {
            "name": "time_preserving",
            "description": "Time-preserving with quality focus",
            "min_elongated_sec": 1.0,   # Only very long segments
            "shorten_factor": 0.8,     # Keep 80%
            "min_segment_samples": 1000,
            "frame_length": 1024,
            "hop_length": 256
        }
    ]


def create_variant_datasets(
    original_csv: str,
    base_output_dir: str,
    variant_configs: List[Dict[str, Any]] = None
) -> Dict[str, str]:
    """Create multiple datasets with different audio processing variants"""
    
    if variant_configs is None:
        variant_configs = get_default_variant_configs()
    
    import csv
    
    # Load original dataset
    with open(original_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    variant_datasets = {}
    
    for config in variant_configs:
        variant_name = config["name"]
        print(f"\n🎛️  Creating {variant_name} variant dataset...")
        
        # Create variant-specific directories
        variant_audio_dir = os.path.join(base_output_dir, f"audio_{variant_name}")
        os.makedirs(variant_audio_dir, exist_ok=True)
        
        # Process all audio files
        variant_rows = []
        processing_summary = []
        
        processor = AudioVariantProcessor()
        
        for i, row in enumerate(rows):
            if i % 10 == 0:
                print(f"  Progress: {i+1}/{len(rows)}")
            
            input_audio = row["audio"].strip()
            text = row["text"].strip()
            
            # Create audio variants
            try:
                variant_results = processor.create_audio_variants(
                    input_audio, variant_audio_dir, [config]
                )
                
                # Get the variant result
                variant_result = variant_results[variant_name]
                variant_audio_path = variant_result["output_path"]
                
                variant_rows.append({
                    "audio": variant_audio_path,
                    "text": text,
                    "original_audio": input_audio
                })
                
                processing_summary.append(variant_result)
                
            except Exception as e:
                print(f"    ❌ Error processing {input_audio}: {e}")
                # Use original audio as fallback
                variant_rows.append({
                    "audio": input_audio,
                    "text": text,
                    "original_audio": input_audio
                })
        
        # Save variant CSV
        variant_csv = os.path.join(base_output_dir, f"dataset_{variant_name}.csv")
        
        with open(variant_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["audio", "text", "original_audio"])
            writer.writeheader()
            writer.writerows(variant_rows)
        
        # Calculate summary statistics
        if processing_summary:
            total_original = sum(s["original_duration_sec"] for s in processing_summary)
            total_new = sum(s["new_duration_sec"] for s in processing_summary)
            total_modified = sum(s["modified_segments"] for s in processing_summary)
            
            print(f"  ✅ {variant_name} dataset created:")
            print(f"     Samples: {len(variant_rows)}")
            print(f"     Duration reduction: {(1 - total_new/total_original)*100:.1f}%")
            print(f"     Modified segments: {total_modified}")
            print(f"     CSV: {variant_csv}")
        
        variant_datasets[variant_name] = variant_csv
    
    return variant_datasets


def analyze_variant_characteristics(variant_csv: str) -> Dict:
    """Analyze characteristics of a variant dataset"""
    
    import csv
    import pandas as pd
    
    # Load dataset
    df = pd.read_csv(variant_csv)
    
    # Analyze audio files
    durations = []
    modifications = []
    
    for _, row in df.iterrows():
        try:
            audio_path = row["audio"]
            if os.path.exists(audio_path):
                y, sr = sf.read(audio_path)
                if len(y.shape) > 1:
                    y = y.mean(axis=1)
                duration = len(y) / sr
                durations.append(duration)
        except:
            pass
    
    return {
        "variant_name": Path(variant_csv).stem.replace("dataset_", ""),
        "csv_path": variant_csv,
        "sample_count": len(df),
        "avg_duration_sec": np.mean(durations) if durations else 0,
        "total_duration_sec": sum(durations) if durations else 0,
        "duration_std": np.std(durations) if durations else 0
    }
