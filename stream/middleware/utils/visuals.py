import sys
import threading
import time


class PreImportSpinner:
    """Progress bar with locked cursor position"""

    def __init__(self, estimated_time=60):
        self.running = False
        self.thread = None
        self.start_time = None
        self.estimated_time = estimated_time
        self.line_saved = False

    def _spin(self):
        """Progress bar that stays on same line"""
        while self.running:
            elapsed = time.time() - self.start_time
            progress = min(elapsed / self.estimated_time, 1.0)
            percent = int(progress * 100)

            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            time_str = f"{minutes}:{seconds:02d}"

            bar_length = 30
            filled = int(bar_length * progress)
            bar = "█" * filled + "░" * (bar_length - filled)

            # Save position on first write
            if not self.line_saved:
                sys.stderr.write("\033[s")  # Save cursor position
                self.line_saved = True

            # Restore position, clear line, write
            sys.stderr.write("\033[u")  # Restore cursor position
            sys.stderr.write("\033[K")  # Clear from cursor to end of line
            sys.stderr.write(f"⏳ Starting STREAM [{bar}] {percent:3d}% ({time_str})")
            sys.stderr.flush()
            time.sleep(0.1)

    def start(self):
        sys.stderr.write("\n")
        sys.stderr.write("⏱️  First startup may take ~1 minute...\n")
        sys.stderr.write("\n")

        # Hide cursor
        sys.stderr.write("\033[?25l")
        sys.stderr.flush()

        self.start_time = time.time()
        self.running = True
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

        elapsed = time.time() - self.start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        time_str = f"{minutes}:{seconds:02d}"

        # Final update
        if self.line_saved:
            sys.stderr.write("\033[u")  # Restore position
            sys.stderr.write("\033[K")  # Clear line

        bar = "█" * 30
        sys.stderr.write(f"✓ Starting STREAM [{bar}] 100% ({time_str})\n")

        # Show cursor
        sys.stderr.write("\033[?25h")
        sys.stderr.write("\n")
        sys.stderr.flush()
