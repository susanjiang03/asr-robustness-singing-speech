"""
Parallel evaluation utilities for ASR models with configurable batching
"""
import csv
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from typing import List, Dict, Any

import torch
from jiwer import cer

from ..models.asr_models import get_model
from ..utils.text_utils import levenshtein_ops, normalize_text


class ParallelEvaluator:
    """Parallel ASR model evaluator with configurable batching"""
    
    def __init__(self, max_workers: int = 3, batch_size: int = 10):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.progress_queue = Queue()
        self.completed_count = 0
        self.total_count = 0
        self.lock = threading.Lock()
        
    def _process_batch(self, batch_data: List[Dict], model_instance, model_name: str):
        """Process a batch of audio files"""
        batch_results = []
        
        for row in batch_data:
            audio = row["audio"].strip()
            ref = normalize_text(row["text"])
            
            try:
                # Transcribe audio
                hyp_raw = model_instance.transcribe(audio)
                hyp = normalize_text(hyp_raw)
                
                # Calculate error operations
                counts = levenshtein_ops(ref, hyp)
                
                result = {
                    "audio": audio,
                    "reference": ref,
                    "hypothesis": hyp,
                    "substitution": counts["substitution"],
                    "insertion": counts["insertion"],
                    "deletion": counts["deletion"],
                }
                
                batch_results.append(result)
                
                # Update progress
                with self.lock:
                    self.completed_count += 1
                    progress = (self.completed_count / self.total_count) * 100
                    print(f"\r[{model_name}] Progress: {progress:.1f}% ({self.completed_count}/{self.total_count})", end="", flush=True)
                    
            except Exception as e:
                print(f"\n❌ Error processing {audio}: {e}")
                # Add failed result with empty hypothesis
                batch_results.append({
                    "audio": audio,
                    "reference": ref,
                    "hypothesis": "",
                    "substitution": 0,
                    "insertion": 0,
                    "deletion": len(ref),
                })
        
        return batch_results
    
    def _create_batches(self, rows: List[Dict]) -> List[List[Dict]]:
        """Split data into batches"""
        batches = []
        for i in range(0, len(rows), self.batch_size):
            batch = rows[i:i + self.batch_size]
            batches.append(batch)
        return batches
    
    def evaluate_dataset_parallel(
        self,
        model_name: str,
        input_csv: str,
        output_pred_csv: str,
        device: str = "cpu",
        vosk_model_path: str = None,
        verbose: bool = True
    ):
        """Evaluate ASR model on dataset using parallel processing"""
        
        if verbose:
            print(f"\n🚀 Starting parallel evaluation: {model_name}")
            print(f"   Max workers: {self.max_workers}")
            print(f"   Batch size: {self.batch_size}")
        
        # Initialize model
        model = get_model(model_name, device, vosk_model_path)
        
        # Load dataset
        with open(input_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        self.total_count = len(rows)
        self.completed_count = 0
        
        if verbose:
            print(f"   Total samples: {self.total_count}")
            print(f"   Batches: {len(self._create_batches(rows))}")
        
        # Create batches
        batches = self._create_batches(rows)
        
        # Process batches in parallel
        all_results = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(self._process_batch, batch, model, model_name): batch 
                for batch in batches
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                except Exception as e:
                    print(f"\n❌ Batch failed: {e}")
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ Parallel evaluation completed in {elapsed_time:.1f}s")
        
        # Calculate final metrics
        refs = [r["reference"] for r in all_results]
        hyps = [r["hypothesis"] for r in all_results]
        
        total_counts = {"substitution": 0, "insertion": 0, "deletion": 0}
        total_chars = 0
        
        for result in all_results:
            for k in total_counts:
                total_counts[k] += result[k]
            total_chars += len(result["reference"])
        
        result = {
            "model": model_name,
            "input_csv": input_csv,
            "rows": len(rows),
            "total_reference_characters": total_chars,
            "cer": cer(refs, hyps),
            "substitution": total_counts["substitution"],
            "insertion": total_counts["insertion"],
            "deletion": total_counts["deletion"],
            "substitution_rate": total_counts["substitution"] / total_chars if total_chars else 0.0,
            "insertion_rate": total_counts["insertion"] / total_chars if total_chars else 0.0,
            "deletion_rate": total_counts["deletion"] / total_chars if total_chars else 0.0,
            "parallel_config": {
                "max_workers": self.max_workers,
                "batch_size": self.batch_size,
                "total_time": elapsed_time,
                "samples_per_second": len(rows) / elapsed_time
            }
        }
        
        # Save predictions
        output_path = Path(output_pred_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "audio",
                    "reference",
                    "hypothesis",
                    "substitution",
                    "insertion",
                    "deletion",
                ],
            )
            writer.writeheader()
            writer.writerows(all_results)
        
        if verbose:
            print(f"📁 Saved predictions: {output_path}")
            print(f"📊 CER: {result['cer']:.4f}")
            print(f"⚡ Speed: {result['parallel_config']['samples_per_second']:.2f} samples/sec")
            print(f"🔧 Config: {result['parallel_config']['max_workers']} workers, {result['parallel_config']['batch_size']} batch size")
        
        return result


def evaluate_dataset_parallel(
    model_name: str,
    input_csv: str,
    output_pred_csv: str,
    device: str = "cpu",
    vosk_model_path: str = None,
    max_workers: int = 3,
    batch_size: int = 10,
    verbose: bool = True
):
    """Convenience function for parallel evaluation"""
    
    evaluator = ParallelEvaluator(max_workers=max_workers, batch_size=batch_size)
    
    return evaluator.evaluate_dataset_parallel(
        model_name=model_name,
        input_csv=input_csv,
        output_pred_csv=output_pred_csv,
        device=device,
        vosk_model_path=vosk_model_path,
        verbose=verbose
    )


def benchmark_parallel_configurations(
    model_name: str,
    input_csv: str,
    device: str = "cpu",
    vosk_model_path: str = None,
    configurations: List[Dict] = None
):
    """Benchmark different parallel configurations"""
    
    if configurations is None:
        configurations = [
            {"max_workers": 1, "batch_size": 1},      # Sequential
            {"max_workers": 2, "batch_size": 5},      # Small parallel
            {"max_workers": 3, "batch_size": 10},     # Medium parallel
            {"max_workers": 4, "batch_size": 15},     # Large parallel
        ]
    
    print(f"🔬 Benchmarking {len(configurations)} configurations for {model_name}")
    
    results = []
    
    for i, config in enumerate(configurations):
        print(f"\n--- Configuration {i+1}/{len(configurations)} ---")
        print(f"Workers: {config['max_workers']}, Batch Size: {config['batch_size']}")
        
        evaluator = ParallelEvaluator(
            max_workers=config['max_workers'], 
            batch_size=config['batch_size']
        )
        
        output_csv = f"benchmark_{model_name}_config_{i+1}.csv"
        
        result = evaluator.evaluate_dataset_parallel(
            model_name=model_name,
            input_csv=input_csv,
            output_pred_csv=output_csv,
            device=device,
            vosk_model_path=vosk_model_path,
            verbose=True
        )
        
        results.append({
            "config_id": i+1,
            "max_workers": config['max_workers'],
            "batch_size": config['batch_size'],
            "cer": result['cer'],
            "total_time": result['parallel_config']['total_time'],
            "samples_per_second": result['parallel_config']['samples_per_second']
        })
        
        # Clean up benchmark file
        try:
            Path(output_csv).unlink()
        except:
            pass
    
    # Print benchmark summary
    print(f"\n📊 Benchmark Summary for {model_name}:")
    print(f"{'Config':<10} {'Workers':<10} {'Batch':<8} {'Time(s)':<10} {'Samples/s':<12} {'CER':<8}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['config_id']:<10} {result['max_workers']:<10} {result['batch_size']:<8} "
              f"{result['total_time']:<10.1f} {result['samples_per_second']:<12.2f} {result['cer']:<8.4f}")
    
    # Find best configuration
    fastest = min(results, key=lambda x: x['total_time'])
    most_efficient = max(results, key=lambda x: x['samples_per_second'])
    
    print(f"\n🏆 Fastest: Config {fastest['config_id']} ({fastest['total_time']:.1f}s)")
    print(f"⚡ Most Efficient: Config {most_efficient['config_id']} ({most_efficient['samples_per_second']:.2f} samples/s)")
    
    return results


def get_optimal_config(dataset_size: int, device_type: str = "cpu") -> Dict:
    """Get recommended parallel configuration based on dataset size and device"""
    
    if device_type == "cpu":
        # CPU: More workers for larger datasets
        if dataset_size < 50:
            return {"max_workers": 2, "batch_size": 5}
        elif dataset_size < 200:
            return {"max_workers": 3, "batch_size": 10}
        else:
            return {"max_workers": 4, "batch_size": 15}
    
    elif device_type == "cuda":
        # GPU: Fewer workers due to memory constraints
        if dataset_size < 50:
            return {"max_workers": 1, "batch_size": 10}
        elif dataset_size < 200:
            return {"max_workers": 2, "batch_size": 15}
        else:
            return {"max_workers": 2, "batch_size": 20}
    
    else:
        # Default conservative config
        return {"max_workers": 2, "batch_size": 10}
