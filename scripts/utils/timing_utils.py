"""
Timing and logging utilities for ASR evaluation
"""
import time
import json
from datetime import datetime
from functools import wraps
from pathlib import Path


class ExecutionTimer:
    """Timer for tracking execution time of cells and operations"""
    
    def __init__(self, log_file=None):
        self.start_time = None
        self.log_file = log_file
        self.execution_log = []
        
    def start(self, operation_name):
        """Start timing an operation"""
        self.start_time = time.time()
        print(f"⏱️  Starting: {operation_name}")
        
    def end(self, operation_name):
        """End timing an operation and log result"""
        if self.start_time is None:
            return
            
        elapsed = time.time() - self.start_time
        minutes, seconds = divmod(elapsed, 60)
        
        # Format time string
        if minutes > 0:
            time_str = f"{minutes:.0f}m {seconds:.1f}s"
        else:
            time_str = f"{seconds:.1f}s"
            
        print(f"✅ Completed: {operation_name} ({time_str})")
        
        # Log to file if specified
        log_entry = {
            "operation": operation_name,
            "elapsed_seconds": elapsed,
            "elapsed_formatted": time_str,
            "timestamp": datetime.now().isoformat()
        }
        
        self.execution_log.append(log_entry)
        
        if self.log_file:
            self._save_log()
            
        self.start_time = None
        return elapsed
        
    def _save_log(self):
        """Save execution log to file"""
        log_path = Path(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.execution_log, f, ensure_ascii=False, indent=2)
    
    def get_summary(self):
        """Get summary of all logged operations"""
        if not self.execution_log:
            return "No operations logged"
            
        total_time = sum(entry['elapsed_seconds'] for entry in self.execution_log)
        total_minutes, total_seconds = divmod(total_time, 60)
        
        summary = f"""
📊 Execution Summary:
🕐 Total Time: {total_minutes:.0f}m {total_seconds:.1f}s
📋 Operations: {len(self.execution_log)}

Operation Details:
"""
        for entry in self.execution_log:
            summary += f"  • {entry['operation']}: {entry['elapsed_formatted']}\n"
            
        return summary


# Global timer instance
_global_timer = None


def get_timer(log_file=None):
    """Get or create global timer instance"""
    global _global_timer
    if _global_timer is None:
        _global_timer = ExecutionTimer(log_file)
    return _global_timer


def time_operation(operation_name=None, log_file=None):
    """Decorator for timing function execution"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            timer = get_timer(log_file)
            op_name = operation_name or func.__name__
            
            timer.start(op_name)
            try:
                result = func(*args, **kwargs)
                timer.end(op_name)
                return result
            except Exception as e:
                timer.end(f"{op_name} (ERROR: {str(e)})")
                raise
                
        return wrapper
    return decorator


def log_cell(operation_name, log_file=None):
    """Context manager for timing cell execution"""
    timer = get_timer(log_file)
    
    class CellTimer:
        def __init__(self, name):
            self.name = name
            
        def __enter__(self):
            timer.start(self.name)
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type:
                timer.end(f"{self.name} (ERROR)")
            else:
                timer.end(self.name)
                
    return CellTimer(operation_name)


def log_notebook_completion(notebook_name, total_time, results_summary=None, log_file=None):
    """Log completion of entire notebook execution"""
    timer = get_timer(log_file)
    
    completion_entry = {
        "event": "notebook_completion",
        "notebook": notebook_name,
        "total_execution_time": total_time,
        "timestamp": datetime.now().isoformat(),
        "results_summary": results_summary
    }
    
    timer.execution_log.append(completion_entry)
    
    if timer.log_file:
        timer._save_log()
    
    minutes, seconds = divmod(total_time, 60)
    time_str = f"{minutes:.0f}m {seconds:.1f}s"
    
    print(f"\n🎉 Notebook '{notebook_name}' completed successfully!")
    print(f"🕐 Total execution time: {time_str}")
    
    if results_summary:
        print(f"📊 Results: {results_summary}")
    
    print(f"📝 Execution log saved to: {timer.log_file}")


def initialize_timing(log_file_path=None):
    """Initialize timing system for a notebook"""
    if log_file_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = f"results/execution_logs/timing_log_{timestamp}.json"
    
    timer = get_timer(log_file_path)
    print(f"⏱️  Timing system initialized")
    print(f"📝 Log file: {log_file_path}")
    return timer
