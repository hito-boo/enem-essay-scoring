import time
import psutil
import pynvml
import threading

class ResourceProfiler:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.cpu_samples = []
        self.gpu_samples = []
        self.start_time = None
        self.end_time = None
        self.running = False
        self.thread = None
        
        try:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0) 
            self.has_gpu = True
        except pynvml.NVMLError:
            self.has_gpu = False

    def monitor(self):
        psutil.cpu_percent(interval=None) 
        while self.running:
            self.cpu_samples.append(psutil.cpu_percent(interval=None))
            if self.has_gpu:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                self.gpu_samples.append(util.gpu)
            time.sleep(self.interval)

    def start(self):
        self.cpu_samples = []
        self.gpu_samples = []
        self.running = True
        self.start_time = time.time()
        
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
        avg_gpu = sum(self.gpu_samples) / len(self.gpu_samples) if self.gpu_samples else 0
        peak_gpu = max(self.gpu_samples) if self.gpu_samples else 0
        total_time = self.end_time - self.start_time if self.start_time and self.end_time else 0
        
        return {
            "time_seconds": total_time,
            "avg_cpu_percent": avg_cpu,
            "peak_cpu_percent": peak_cpu,
            "avg_gpu_percent": avg_gpu,
            "peak_gpu_percent": peak_gpu
        }
