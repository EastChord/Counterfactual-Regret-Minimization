"""
Memory monitoring utility for Python applications.

This module provides a comprehensive memory monitoring class that can track
memory usage, detect memory leaks, and provide detailed memory statistics.
"""
import psutil
import os
import time
import gc
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class MemorySnapshot:
    """Memory usage snapshot at a specific point in time."""
    timestamp: float
    memory_mb: float
    memory_percent: float
    iteration: Optional[int] = None
    description: Optional[str] = None


class MemoryMonitor:
    """
    A comprehensive memory monitoring class for tracking memory usage in Python applications.
    
    Features:
    - Real-time memory usage tracking
    - Memory leak detection
    - Automatic garbage collection when memory usage is high
    - Historical memory usage data
    - Memory usage alerts
    """
    
    def __init__(self, 
                 process_id: Optional[int] = None,
                 alert_threshold_mb: float = 1000.0,
                 gc_threshold_multiplier: float = 2.0,
                 enable_auto_gc: bool = True):
        """
        Initialize memory monitor.
        
        Args:
            process_id: Process ID to monitor (default: current process)
            alert_threshold_mb: Memory threshold in MB to trigger alerts
            gc_threshold_multiplier: Multiplier for initial memory to trigger GC
            enable_auto_gc: Whether to automatically run garbage collection
        """
        self.process = psutil.Process(process_id or os.getpid())
        self.alert_threshold_mb = alert_threshold_mb
        self.gc_threshold_multiplier = gc_threshold_multiplier
        self.enable_auto_gc = enable_auto_gc
        
        # Memory tracking data
        self.snapshots: List[MemorySnapshot] = []
        self.initial_memory: float = 0.0
        self.peak_memory: float = 0.0
        self.gc_count: int = 0
        
        # Initialize with current memory usage
        self.initial_memory = self.get_current_memory_mb()
        self.peak_memory = self.initial_memory
        
    def get_current_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            memory_info = self.process.memory_info()
            return memory_info.rss / 1024 / 1024
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0
    
    def get_memory_percent(self) -> float:
        """Get current memory usage as percentage of system memory."""
        try:
            return self.process.memory_percent()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0
    
    def take_snapshot(self, 
                     iteration: Optional[int] = None, 
                     description: Optional[str] = None) -> MemorySnapshot:
        """
        Take a memory usage snapshot.
        
        Args:
            iteration: Current iteration number (for tracking progress)
            description: Description of the snapshot
            
        Returns:
            MemorySnapshot object with current memory data
        """
        memory_mb = self.get_current_memory_mb()
        memory_percent = self.get_memory_percent()
        
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            memory_mb=memory_mb,
            memory_percent=memory_percent,
            iteration=iteration,
            description=description
        )
        
        self.snapshots.append(snapshot)
        
        # Update peak memory
        if memory_mb > self.peak_memory:
            self.peak_memory = memory_mb
        
        return snapshot
    
    def check_memory_alert(self) -> bool:
        """
        Check if memory usage exceeds alert threshold.
        
        Returns:
            True if memory usage is above threshold
        """
        current_memory = self.get_current_memory_mb()
        return current_memory > self.alert_threshold_mb
    
    def check_and_run_gc(self) -> bool:
        """
        Check if garbage collection should be run and run it if necessary.
        
        Returns:
            True if garbage collection was run
        """
        if not self.enable_auto_gc:
            return False
            
        current_memory = self.get_current_memory_mb()
        gc_threshold = self.initial_memory * self.gc_threshold_multiplier
        
        if current_memory > gc_threshold:
            memory_before = current_memory
            gc.collect()
            self.gc_count += 1
            memory_after = self.get_current_memory_mb()
            
            print(f"Garbage collection triggered: {memory_before:.2f}MB -> {memory_after:.2f}MB "
                  f"(freed: {memory_before - memory_after:.2f}MB)")
            return True
        
        return False
    
    def monitor_iteration(self, 
                         iteration: int, 
                         check_interval: int = 1000,
                         description: Optional[str] = None) -> Optional[MemorySnapshot]:
        """
        Monitor memory usage for a specific iteration.
        
        Args:
            iteration: Current iteration number
            check_interval: How often to check memory (every N iterations)
            description: Optional description for the snapshot
            
        Returns:
            MemorySnapshot if monitoring was performed, None otherwise
        """
        if iteration % check_interval == 0:
            snapshot = self.take_snapshot(iteration, description)
            
            # Check for memory alerts
            if self.check_memory_alert():
                print(f"⚠️  Memory alert at iteration {iteration}: {snapshot.memory_mb:.2f}MB")
            
            # Check and run garbage collection if needed
            self.check_and_run_gc()
            
            return snapshot
        
        return None
    
    def get_memory_stats(self) -> Dict:
        """Get comprehensive memory statistics."""
        current_memory = self.get_current_memory_mb()
        
        return {
            'initial_memory_mb': self.initial_memory,
            'current_memory_mb': current_memory,
            'peak_memory_mb': self.peak_memory,
            'memory_increase_mb': current_memory - self.initial_memory,
            'memory_percent': self.get_memory_percent(),
            'snapshots_count': len(self.snapshots),
            'gc_count': self.gc_count,
            'is_above_threshold': self.check_memory_alert()
        }
    
    def print_memory_stats(self, iteration: Optional[int] = None):
        """Print current memory statistics."""
        stats = self.get_memory_stats()
        
        print(f"\n{'='*50}")
        print(f"MEMORY STATISTICS {f'(Iteration {iteration})' if iteration else ''}")
        print(f"{'='*50}")
        print(f"Initial memory:     {stats['initial_memory_mb']:.2f} MB")
        print(f"Current memory:     {stats['current_memory_mb']:.2f} MB")
        print(f"Peak memory:        {stats['peak_memory_mb']:.2f} MB")
        print(f"Memory increase:    {stats['memory_increase_mb']:.2f} MB")
        print(f"Memory percentage:  {stats['memory_percent']:.2f}%")
        print(f"Snapshots taken:    {stats['snapshots_count']}")
        print(f"GC runs:           {stats['gc_count']}")
        print(f"Above threshold:    {'Yes' if stats['is_above_threshold'] else 'No'}")
        print(f"{'='*50}\n")
    
    def get_memory_trend(self) -> str:
        """Analyze memory usage trend from snapshots."""
        if len(self.snapshots) < 2:
            return "Insufficient data for trend analysis"
        
        recent_snapshots = self.snapshots[-10:]  # Last 10 snapshots
        memory_values = [s.memory_mb for s in recent_snapshots]
        
        if len(memory_values) < 2:
            return "Insufficient data for trend analysis"
        
        # Simple trend analysis
        first_half = memory_values[:len(memory_values)//2]
        second_half = memory_values[len(memory_values)//2:]
        
        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)
        
        if avg_second > avg_first * 1.1:
            return "Increasing trend (potential memory leak)"
        elif avg_second < avg_first * 0.9:
            return "Decreasing trend (memory being freed)"
        else:
            return "Stable trend"
    
    def reset(self):
        """Reset monitor to initial state."""
        self.snapshots.clear()
        self.initial_memory = self.get_current_memory_mb()
        self.peak_memory = self.initial_memory
        self.gc_count = 0
        print("Memory monitor reset to initial state")


# Convenience function for quick monitoring
def quick_memory_check() -> float:
    """Quick memory usage check in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024
