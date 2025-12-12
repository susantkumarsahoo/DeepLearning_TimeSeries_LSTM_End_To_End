"""
Single Launcher Script for FastAPI Backend + Streamlit Frontend
Run both services with a single command: python run_app.py
"""

import subprocess
import sys
import os
import time
import signal
import requests
from pathlib import Path

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_colored(message, color=Colors.OKGREEN):
    """Print colored message"""
    print(f"{color}{message}{Colors.ENDC}")

def print_header(message):
    """Print header with decorations"""
    print("\n" + "=" * 80)
    print_colored(message, Colors.HEADER + Colors.BOLD)
    print("=" * 80 + "\n")

def check_port_available(port):
    """Check if a port is available"""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    return result != 0

def wait_for_api(max_attempts=30, delay=1):
    """Wait for FastAPI to be ready"""
    print_colored("⏳ Waiting for FastAPI to start...", Colors.OKCYAN)
    
    for attempt in range(max_attempts):
        try:
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                print_colored("✅ FastAPI is ready!", Colors.OKGREEN)
                return True
        except requests.exceptions.RequestException:
            pass
        
        print(f"   Attempt {attempt + 1}/{max_attempts}...", end='\r')
        time.sleep(delay)
    
    print_colored("❌ FastAPI failed to start in time", Colors.FAIL)
    return False

def kill_process_on_port(port):
    """Kill any process running on the specified port (Windows)"""
    try:
        if sys.platform == "win32":
            # Windows command
            result = subprocess.run(
                f'netstat -ano | findstr :{port}',
                shell=True,
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                # Extract PIDs and kill them
                lines = result.stdout.strip().split('\n')
                pids = set()
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 5:
                        pids.add(parts[-1])
                
                for pid in pids:
                    try:
                        subprocess.run(f'taskkill /F /PID {pid}', shell=True, capture_output=True)
                        print_colored(f"   Killed process on port {port} (PID: {pid})", Colors.WARNING)
                    except:
                        pass
        else:
            # Linux/Mac command
            subprocess.run(f'lsof -ti:{port} | xargs kill -9', shell=True, capture_output=True)
    except Exception as e:
        print_colored(f"   Could not kill process on port {port}: {e}", Colors.WARNING)

def start_fastapi():
    """Start FastAPI backend"""
    print_header("🚀 STARTING FASTAPI BACKEND")
    
    # Check if port 8000 is available
    if not check_port_available(8000):
        print_colored("⚠️  Port 8000 is already in use. Attempting to free it...", Colors.WARNING)
        kill_process_on_port(8000)
        time.sleep(2)
    
    print_colored("📡 Starting FastAPI server on http://localhost:8000", Colors.OKBLUE)
    
    # Start FastAPI with uvicorn - show output in real-time
    fastapi_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    return fastapi_process

def start_streamlit(streamlit_file):
    """Start Streamlit frontend"""
    print_header("🎨 STARTING STREAMLIT FRONTEND")
    
    # Check if port 8501 is available
    if not check_port_available(8501):
        print_colored("⚠️  Port 8501 is already in use. Attempting to free it...", Colors.WARNING)
        kill_process_on_port(8501)
        time.sleep(2)
    
    print_colored(f"🌐 Starting Streamlit from {streamlit_file} on http://localhost:8501", Colors.OKBLUE)
    
    # Start Streamlit - show output in real-time
    streamlit_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", streamlit_file, 
         "--server.port", "8501", 
         "--server.headless", "true",
         "--browser.gatherUsageStats", "false",
         "--logger.level", "info"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    return streamlit_process

def monitor_processes(fastapi_process, streamlit_process):
    """Monitor both processes and handle output"""
    print_header("✅ BOTH SERVICES ARE RUNNING")
    print_colored("FastAPI Backend:    http://localhost:8000", Colors.OKGREEN)
    print_colored("FastAPI Docs:       http://localhost:8000/docs", Colors.OKGREEN)
    print_colored("Streamlit Frontend: http://localhost:8501", Colors.OKGREEN)
    print_colored("\n💡 Tip: Predictions may take time for large date ranges", Colors.OKCYAN)
    print_colored("    Watch the logs below for progress updates\n", Colors.OKCYAN)
    print_colored("Press CTRL+C to stop both services\n", Colors.WARNING)
    print("=" * 80)
    
    import threading
    import queue
    
    # Queues for process output
    fastapi_queue = queue.Queue()
    streamlit_queue = queue.Queue()
    
    def read_output(process, output_queue, prefix):
        """Read process output in a separate thread"""
        try:
            for line in iter(process.stdout.readline, ''):
                if line:
                    output_queue.put((prefix, line.rstrip()))
        except:
            pass
    
    # Start threads to read output
    fastapi_thread = threading.Thread(
        target=read_output, 
        args=(fastapi_process, fastapi_queue, "[FastAPI]"),
        daemon=True
    )
    streamlit_thread = threading.Thread(
        target=read_output, 
        args=(streamlit_process, streamlit_queue, "[Streamlit]"),
        daemon=True
    )
    
    fastapi_thread.start()
    streamlit_thread.start()
    
    try:
        while True:
            # Check if processes are still running
            fastapi_status = fastapi_process.poll()
            streamlit_status = streamlit_process.poll()
            
            if fastapi_status is not None:
                print_colored(f"\n❌ FastAPI process exited with code {fastapi_status}", Colors.FAIL)
                break
            
            if streamlit_status is not None:
                print_colored(f"\n❌ Streamlit process exited with code {streamlit_status}", Colors.FAIL)
                break
            
            # Print FastAPI output
            while not fastapi_queue.empty():
                try:
                    prefix, line = fastapi_queue.get_nowait()
                    # Color code important messages
                    if "ERROR" in line or "error" in line:
                        print_colored(f"{prefix} {line}", Colors.FAIL)
                    elif "WARNING" in line or "warning" in line:
                        print_colored(f"{prefix} {line}", Colors.WARNING)
                    elif "prediction" in line.lower() or "generating" in line.lower():
                        print_colored(f"{prefix} {line}", Colors.OKCYAN)
                    elif "POST /predict" in line:
                        print_colored(f"{prefix} {line}", Colors.OKGREEN)
                    else:
                        print(f"{prefix} {line}")
                except queue.Empty:
                    break
            
            # Print Streamlit output (less verbose)
            while not streamlit_queue.empty():
                try:
                    prefix, line = streamlit_queue.get_nowait()
                    # Only show important Streamlit messages
                    if any(keyword in line.lower() for keyword in ["error", "warning", "you can now view"]):
                        if "error" in line.lower():
                            print_colored(f"{prefix} {line}", Colors.FAIL)
                        elif "warning" in line.lower():
                            print_colored(f"{prefix} {line}", Colors.WARNING)
                        else:
                            print_colored(f"{prefix} {line}", Colors.OKCYAN)
                except queue.Empty:
                    break
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print_colored("\n\n🛑 Shutting down services...", Colors.WARNING)
    
    return fastapi_process, streamlit_process

def cleanup(fastapi_process, streamlit_process):
    """Clean up and terminate processes"""
    print_header("🧹 CLEANING UP")
    
    # Terminate processes
    if fastapi_process and fastapi_process.poll() is None:
        print_colored("Stopping FastAPI...", Colors.OKCYAN)
        fastapi_process.terminate()
        try:
            fastapi_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            fastapi_process.kill()
    
    if streamlit_process and streamlit_process.poll() is None:
        print_colored("Stopping Streamlit...", Colors.OKCYAN)
        streamlit_process.terminate()
        try:
            streamlit_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            streamlit_process.kill()
    
    # Extra cleanup for ports
    kill_process_on_port(8000)
    kill_process_on_port(8501)
    
    print_colored("\n✅ All services stopped successfully!", Colors.OKGREEN)
    print_colored("=" * 80, Colors.OKGREEN)

def find_streamlit_file():
    """Find the Streamlit application file"""
    possible_names = [
        'streamlit_app.py',
        'frontend.py',
        'dashboard.py',
        'ui.py',
        'main_app.py',
        'app_streamlit.py',
        'web_app.py'
    ]
    
    for filename in possible_names:
        if Path(filename).exists():
            return filename
    
    return None

def verify_files():
    """Verify required files exist"""
    # Check for app.py (FastAPI)
    if not Path('app.py').exists():
        print_colored("❌ Missing required file: app.py", Colors.FAIL)
        print_colored("Please ensure app.py (FastAPI backend) is in the current directory.", Colors.WARNING)
        return False, None
    
    # Check for Streamlit file
    streamlit_file = find_streamlit_file()
    if not streamlit_file:
        print_colored("❌ No Streamlit application file found!", Colors.FAIL)
        print_colored("Please ensure one of these files exists:", Colors.WARNING)
        print_colored("  - streamlit_app.py (recommended)", Colors.OKCYAN)
        print_colored("  - frontend.py", Colors.OKCYAN)
        print_colored("  - dashboard.py", Colors.OKCYAN)
        print_colored("  - ui.py", Colors.OKCYAN)
        
        # List all Python files in directory
        print_colored("\nPython files found in current directory:", Colors.WARNING)
        py_files = list(Path('.').glob('*.py'))
        if py_files:
            for f in py_files:
                print_colored(f"  - {f.name}", Colors.OKCYAN)
        else:
            print_colored("  No Python files found!", Colors.FAIL)
        
        return False, None
    
    print_colored(f"✓ Found Streamlit file: {streamlit_file}", Colors.OKGREEN)
    return True, streamlit_file

def main():
    """Main function to run both services"""
    print_header("🚀 TIME SERIES FORECASTING APPLICATION LAUNCHER")
    print_colored("Starting FastAPI Backend + Streamlit Frontend\n", Colors.OKBLUE)
    
    # Verify files
    files_ok, streamlit_file = verify_files()
    if not files_ok:
        sys.exit(1)
    
    fastapi_process = None
    streamlit_process = None
    
    try:
        # Start FastAPI
        fastapi_process = start_fastapi()
        
        # Wait for FastAPI to be ready
        if not wait_for_api():
            print_colored("❌ Failed to start FastAPI. Check the logs above.", Colors.FAIL)
            cleanup(fastapi_process, None)
            sys.exit(1)
        
        # Start Streamlit
        streamlit_process = start_streamlit(streamlit_file)
        
        # Wait a bit for Streamlit to start
        time.sleep(3)
        
        # Monitor both processes
        fastapi_process, streamlit_process = monitor_processes(fastapi_process, streamlit_process)
        
    except Exception as e:
        print_colored(f"\n❌ Error occurred: {str(e)}", Colors.FAIL)
        import traceback
        traceback.print_exc()
    
    finally:
        # Always cleanup
        cleanup(fastapi_process, streamlit_process)

if __name__ == "__main__":
    main()

# Run with: python run_app.py