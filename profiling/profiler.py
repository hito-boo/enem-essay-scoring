import time
import psutil
import threading
import torch

class ResourceProfiler:
    def __init__(self, interval=1.0, device=None):
        self.interval = interval
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.has_gpu = self.device.type == "cuda"
        self.cpu_samples = []
        self.ram_samples_mb = []
        self.start_time = None
        self.end_time = None
        self.running = False
        self.thread = None
        self.process = psutil.Process()

    def monitor(self):
        psutil.cpu_percent(interval=None) 
        while self.running:
            self.cpu_samples.append(psutil.cpu_percent(interval=None))
            self.ram_samples_mb.append(self.process.memory_info().rss / (1024 ** 2))
            time.sleep(self.interval)

    def start(self):
        self.cpu_samples = []
        self.ram_samples_mb = []
        self.running = True
        self.start_time = time.time()
        
        if self.has_gpu:
            torch.cuda.reset_peak_memory_stats(self.device)

        self.thread = threading.Thread(target=self.monitor)
        self.thread.start()

    def stop(self):
        self.running = False
        self.end_time = time.time()
        if self.thread:
            self.thread.join()

    def get_results(self):
        avg_cpu = sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0
        peak_cpu = max(self.cpu_samples) if self.cpu_samples else 0
        avg_ram = sum(self.ram_samples_mb) / len(self.ram_samples_mb) if self.ram_samples_mb else 0
        peak_ram = max(self.ram_samples_mb) if self.ram_samples_mb else 0
        total_time = self.end_time - self.start_time if self.start_time and self.end_time else 0
        peak_vram_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2) if self.has_gpu else 0

        return {
            "time_seconds": total_time,
            "avg_cpu_percent": avg_cpu,
            "peak_cpu_percent": peak_cpu,
            "avg_ram_mb": avg_ram,
            "peak_ram_mb": peak_ram,
            "peak_vram_mb": peak_vram_mb,
        }
