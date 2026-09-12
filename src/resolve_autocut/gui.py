"""
GUI for Resolve AutoCut.

This module provides a simple tkinter-based GUI for the application.

Features:
- Media file selection
- Analysis progress display
- Filler detection table with enable/disable checkboxes
- Configuration options
- Export buttons for timeline, SRT, and JSON
"""

import os
import tkinter as tk
from tkinter import (
    ttk, filedialog, messagebox, scrolledtext, StringVar, DoubleVar, BooleanVar
)
from typing import Optional, Dict, Any, List, Callable
import threading
import queue
import logging

from .app import ResolveAutoCut, AnalysisResult, AnalysisError
from .models import FillerDetection, FillerType, Interval
from .intervals import merge_overlapping, invert_intervals, pad_intervals


logger = logging.getLogger(__name__)


class GUIError(Exception):
    """Exception raised for GUI errors."""
    pass


class AnalysisThread(threading.Thread):
    """Thread for running analysis in the background."""
    
    def __init__(
        self,
        app: ResolveAutoCut,
        media_path: str,
        progress_queue: queue.Queue
    ):
        super().__init__()
        self.app = app
        self.media_path = media_path
        self.progress_queue = progress_queue
        self.result = None
        self.error = None
        self.daemon = True
    
    def run(self):
        """Run the analysis."""
        try:
            def progress_callback(progress):
                self.progress_queue.put(progress)
            
            self.result = self.app.analyze(
                self.media_path,
                progress_callback=progress_callback
            )
        except Exception as e:
            self.error = str(e)


class ResolveAutoCutGUI:
    """Main GUI application."""
    
    def __init__(self, root: tk.Tk):
        """Initialize the GUI.
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("Resolve AutoCut")
        self.root.geometry("1200x800")
        
        # Application state
        self.app = ResolveAutoCut()
        self.media_path = None
        self.analysis: Optional[AnalysisResult] = None
        self.analysis_thread: Optional[AnalysisThread] = None
        self.progress_queue = queue.Queue()
        
        # Configuration
        self.cut_padding_var = DoubleVar(value=0.05)
        self.whisper_model_var = StringVar(value="base")
        self.confidence_threshold_var = DoubleVar(value=0.5)
        
        # Create UI
        self._create_widgets()
        self._create_menu()
        
        # Setup periodic progress checking
        self.root.after(100, self._check_progress)
        
        # Setup logging
        self._setup_logging()
    
    def _create_widgets(self):
        """Create all UI widgets."""
        # Main notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Analysis tab
        self.analysis_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_tab, text="Analysis")
        self._create_analysis_tab()
        
        # Review tab
        self.review_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.review_tab, text="Review")
        self._create_review_tab()
        
        # Export tab
        self.export_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.export_tab, text="Export")
        self._create_export_tab()
        
        # Log tab
        self.log_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.log_tab, text="Log")
        self._create_log_tab()
    
    def _create_analysis_tab(self):
        """Create widgets for the Analysis tab."""
        frame = self.analysis_tab
        
        # File selection
        file_frame = ttk.LabelFrame(frame, text="Media File")
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.file_path_var = StringVar()
        
        file_label = ttk.Label(file_frame, text="Selected file:")
        file_label.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, width=80)
        self.file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        
        browse_btn = ttk.Button(
            file_frame, text="Browse...", command=self._browse_file
        )
        browse_btn.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Options frame
        options_frame = ttk.LabelFrame(frame, text="Options")
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Cut padding
        padding_label = ttk.Label(options_frame, text="Cut Padding (seconds):")
        padding_label.grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        
        padding_entry = ttk.Entry(options_frame, textvariable=self.cut_padding_var, width=10)
        padding_entry.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        
        # Whisper model
        model_label = ttk.Label(options_frame, text="Whisper Model:")
        model_label.grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        
        model_combo = ttk.Combobox(
            options_frame,
            textvariable=self.whisper_model_var,
            values=["tiny", "base", "small", "medium"],
            width=10
        )
        model_combo.grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
        model_combo.current(1)  # Default to "base"
        
        # Confidence threshold
        conf_label = ttk.Label(options_frame, text="Min Confidence:")
        conf_label.grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
        
        conf_entry = ttk.Entry(options_frame, textvariable=self.confidence_threshold_var, width=10)
        conf_entry.grid(row=2, column=1, padx=5, pady=2, sticky=tk.W)
        
        # Media info
        info_frame = ttk.LabelFrame(frame, text="Media Information")
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.media_info_text = scrolledtext.ScrolledText(
            info_frame, height=4, wrap=tk.WORD
        )
        self.media_info_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Progress
        progress_frame = ttk.LabelFrame(frame, text="Progress")
        progress_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.progress_var = StringVar(value="Ready")
        self.progress_label = ttk.Label(
            progress_frame, textvariable=self.progress_var
        )
        self.progress_label.pack(fill=tk.X, padx=5, pady=5)
        
        self.progress_bar = ttk.Progressbar(
            progress_frame, orient=tk.HORIZONTAL, length=200, mode='determinate'
        )
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        analyze_btn = ttk.Button(
            button_frame,
            text="Analyze",
            command=self._start_analysis,
            state=tk.NORMAL
        )
        analyze_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        cancel_btn = ttk.Button(
            button_frame,
            text="Cancel",
            command=self._cancel_analysis,
            state=tk.DISABLED
        )
        cancel_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Store button references
        self.analyze_btn = analyze_btn
        self.cancel_btn = cancel_btn
    
    def _create_review_tab(self):
        """Create widgets for the Review tab."""
        frame = self.review_tab
        
        # Summary
        summary_frame = ttk.LabelFrame(frame, text="Summary")
        summary_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.summary_text = scrolledtext.ScrolledText(
            summary_frame, height=4, wrap=tk.WORD
        )
        self.summary_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Filler detections table
        table_frame = ttk.LabelFrame(frame, text="Filler Detections")
        table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Treeview for detections
        self.detections_tree = ttk.Treeview(
            table_frame,
            columns=("Enabled", "Start", "End", "Type", "Confidence", "Duration"),
            show="headings"
        )
        
        # Configure columns
        self.detections_tree.heading("Enabled", text="Remove", command=lambda: self._sort_column("Enabled", False))
        self.detections_tree.heading("Start", text="Start (s)", command=lambda: self._sort_column("Start", False))
        self.detections_tree.heading("End", text="End (s)")
        self.detections_tree.heading("Type", text="Type")
        self.detections_tree.heading("Confidence", text="Confidence")
        self.detections_tree.heading("Duration", text="Duration (s)")
        
        self.detections_tree.column("Enabled", width=60, anchor=tk.CENTER)
        self.detections_tree.column("Start", width=80, anchor=tk.E)
        self.detections_tree.column("End", width=80, anchor=tk.E)
        self.detections_tree.column("Type", width=80, anchor=tk.W)
        self.detections_tree.column("Confidence", width=80, anchor=tk.E)
        self.detections_tree.column("Duration", width=80, anchor=tk.E)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(
            table_frame, orient=tk.VERTICAL, command=self.detections_tree.yview
        )
        self.detections_tree.configure(yscrollcommand=scrollbar.set)
        
        self.detections_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind double-click to toggle enabled
        self.detections_tree.bind("<Double-1>", self._toggle_detection_enabled)
        
        # Enable/disable all buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        enable_all_btn = ttk.Button(
            button_frame,
            text="Enable All",
            command=self._enable_all_detections
        )
        enable_all_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        disable_all_btn = ttk.Button(
            button_frame,
            text="Disable All",
            command=self._disable_all_detections
        )
        disable_all_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        refresh_btn = ttk.Button(
            button_frame,
            text="Refresh",
            command=self._refresh_detections_table
        )
        refresh_btn.pack(side=tk.LEFT, padx=5, pady=5)
    
    def _create_export_tab(self):
        """Create widgets for the Export tab."""
        frame = self.export_tab
        
        # Export info
        info_frame = ttk.LabelFrame(frame, text="Export Status")
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.export_info_text = scrolledtext.ScrolledText(
            info_frame, height=6, wrap=tk.WORD
        )
        self.export_info_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Output directory
        output_frame = ttk.LabelFrame(frame, text="Output Directory")
        output_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.output_dir_var = StringVar()
        
        dir_label = ttk.Label(output_frame, text="Directory:")
        dir_label.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.dir_entry = ttk.Entry(output_frame, textvariable=self.output_dir_var, width=60)
        self.dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        
        output_browse_btn = ttk.Button(
            output_frame, text="Browse...", command=self._browse_output_dir
        )
        output_browse_btn.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Base filename
        filename_frame = ttk.LabelFrame(frame, text="Base Filename")
        filename_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.base_filename_var = StringVar()
        
        filename_label = ttk.Label(filename_frame, text="Base name (without extension):")
        filename_label.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.filename_entry = ttk.Entry(
            filename_frame, textvariable=self.base_filename_var, width=40
        )
        self.filename_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        
        # Export buttons
        button_frame = ttk.LabelFrame(frame, text="Export")
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        export_fcpxml_btn = ttk.Button(
            button_frame,
            text="Export FCPXML Timeline",
            command=self._export_fcpxml
        )
        export_fcpxml_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        export_srt_btn = ttk.Button(
            button_frame,
            text="Export SRT Subtitles",
            command=self._export_srt
        )
        export_srt_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        export_json_btn = ttk.Button(
            button_frame,
            text="Export JSON",
            command=self._export_json
        )
        export_json_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        export_all_btn = ttk.Button(
            button_frame,
            text="Export All",
            command=self._export_all
        )
        export_all_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Store button references
        self.export_buttons = [
            export_fcpxml_btn, export_srt_btn, export_json_btn, export_all_btn
        ]
        
        # Update button states
        self._update_export_button_states()
    
    def _create_log_tab(self):
        """Create widgets for the Log tab."""
        frame = self.log_tab
        
        self.log_text = scrolledtext.ScrolledText(
            frame, wrap=tk.WORD, height=24
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Clear button
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        clear_btn = ttk.Button(
            button_frame,
            text="Clear",
            command=self._clear_log
        )
        clear_btn.pack(side=tk.RIGHT, padx=5, pady=5)
    
    def _create_menu(self):
        """Create the menu bar."""
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open...", command=self._browse_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Options menu
        options_menu = tk.Menu(menubar, tearoff=0)
        options_menu.add_command(
            label="Set Cut Padding...",
            command=self._show_cut_padding_dialog
        )
        menubar.add_cascade(label="Options", menu=options_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)
    
    def _setup_logging(self):
        """Setup logging to the GUI."""
        # Create a custom handler
        class GUIHandler(logging.Handler):
            def __init__(self, text_widget):
                super().__init__()
                self.text_widget = text_widget
            
            def emit(self, record):
                try:
                    msg = self.format(record)
                    self.text_widget.insert(tk.END, msg + "\n")
                    self.text_widget.see(tk.END)
                    self.text_widget.update()
                except Exception:
                    pass
        
        # Add handler
        gui_handler = GUIHandler(self.log_text)
        gui_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(gui_handler)
        logging.getLogger().setLevel(logging.INFO)
    
    def _browse_file(self):
        """Open file dialog to select media file."""
        filetypes = [
            ("Video Files", "*.mp4 *.mov *.avi *.mkv"),
            ("Audio Files", "*.wav *.mp3 *.m4a *.aac"),
            ("All Files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select Media File",
            filetypes=filetypes
        )
        
        if file_path:
            self.media_path = file_path
            self.file_path_var.set(file_path)
            self._update_analysis_state()
    
    def _browse_output_dir(self):
        """Open directory dialog to select output directory."""
        dir_path = filedialog.askdirectory(title="Select Output Directory")
        if dir_path:
            self.output_dir_var.set(dir_path)
    
    def _start_analysis(self):
        """Start the analysis thread."""
        if not self.media_path:
            messagebox.showerror("Error", "Please select a media file first")
            return
        
        if not os.path.exists(self.media_path):
            messagebox.showerror("Error", f"File not found: {self.media_path}")
            return
        
        # Update UI state
        self._set_analyzing_state(True)
        self.progress_var.set("Validating media...")
        self.progress_bar["value"] = 0
        
        # Update app configuration
        self.app.cut_padding = self.cut_padding_var.get()
        self.app.confidence_threshold = self.confidence_threshold_var.get()
        self.app.whisper_model = self.whisper_model_var.get()
        
        # Start analysis thread
        self.progress_queue = queue.Queue()
        self.analysis_thread = AnalysisThread(
            self.app, self.media_path, self.progress_queue
        )
        self.analysis_thread.start()
        
        # Check if thread completed immediately
        self.root.after(100, self._check_analysis_complete)
    
    def _cancel_analysis(self):
        """Cancel the current analysis."""
        if self.analysis_thread and self.analysis_thread.is_alive():
            # Note: Threads can't be forcibly stopped in Python
            # We just update the UI state
            self._set_analyzing_state(False)
            self.progress_var.set("Analysis cancelled")
    
    def _check_analysis_complete(self):
        """Check if analysis thread has completed."""
        if self.analysis_thread and not self.analysis_thread.is_alive():
            if self.analysis_thread.error:
                self._set_analyzing_state(False)
                messagebox.showerror(
                    "Analysis Error",
                    f"Analysis failed: {self.analysis_thread.error}"
                )
                self.progress_var.set(f"Error: {self.analysis_thread.error}")
            else:
                self.analysis = self.analysis_thread.result
                self._set_analyzing_state(False)
                self.progress_var.set("Analysis complete")
                self.progress_bar["value"] = 100
                self._update_after_analysis()
        else:
            # Check again
            self.root.after(100, self._check_analysis_complete)
    
    def _set_analyzing_state(self, analyzing: bool):
        """Set the analyzing state and update UI."""
        self.analyze_btn["state"] = tk.DISABLED if analyzing else tk.NORMAL
        self.cancel_btn["state"] = tk.NORMAL if analyzing else tk.DISABLED
        self._update_export_button_states()
    
    def _check_progress(self):
        """Check for progress updates from the analysis thread."""
        try:
            while True:
                progress = self.progress_queue.get_nowait()
                if "percent" in progress:
                    self.progress_bar["value"] = progress["percent"]
                if "step" in progress:
                    self.progress_var.set(f"{progress['step']}... ({progress.get('percent', 0):.0f}%)")
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self._check_progress)
    
    def _update_after_analysis(self):
        """Update UI after analysis completes."""
        if self.analysis is None:
            return
        
        # Update media info
        media_info = self.analysis.media_info
        info_text = (
            f"File: {os.path.basename(media_info.path)}\n"
            f"Duration: {media_info.duration:.2f} seconds\n"
            f"Video: {media_info.width}x{media_info.height} @ {media_info.frame_rate}\n"
            f"Audio: {media_info.sample_rate}Hz, {media_info.channels} channels"
        )
        self.media_info_text.delete(1.0, tk.END)
        self.media_info_text.insert(tk.END, info_text)
        
        # Update summary
        summary = (
            f"Filler Detections: {len(self.analysis.filler_detections)} "
            f"(Enabled: {len(self.analysis.enabled_detections)}, "
            f"Disabled: {len(self.analysis.disabled_detections)})\n"
            f"Words: {len(self.analysis.words)}\n"
            f"Captions: {len(self.analysis.captions)}\n"
            f"Total Removed: {self.analysis.total_removed_duration:.2f} seconds\n"
            f"Edited Duration: {self.analysis.edited_duration:.2f} seconds"
        )
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, summary)
        
        # Update detections table
        self._update_detections_table()
        
        # Set default output directory
        if not self.output_dir_var.get():
            default_dir = os.path.join(
                os.path.expanduser("~"),
                "Documents",
                "Resolve AutoCut Output"
            )
            self.output_dir_var.set(default_dir)
        
        # Set default base filename
        if not self.base_filename_var.get():
            base_name = os.path.splitext(
                os.path.basename(self.media_path)
            )[0]
            self.base_filename_var.set(base_name)
        
        # Update button states
        self._update_export_button_states()
    
    def _update_detections_table(self):
        """Update the detections table with current analysis data."""
        if self.analysis is None:
            return
        
        # Clear existing items
        for item in self.detections_tree.get_children():
            self.detections_tree.delete(item)
        
        # Add new items
        for i, detection in enumerate(self.analysis.filler_detections):
            tags = ("enabled" if detection.enabled else "disabled",)
            self.detections_tree.insert(
                "",
                tk.END,
                values=(
                    "Yes" if detection.enabled else "No",
                    f"{detection.start:.3f}",
                    f"{detection.end:.3f}",
                    detection.filler_type.value,
                    f"{detection.confidence:.3f}",
                    f"{detection.duration:.3f}"
                ),
                tags=tags
            )
        
        # Apply tag colors
        self.detections_tree.tag_configure("enabled", foreground="green")
        self.detections_tree.tag_configure("disabled", foreground="red")
    
    def _refresh_detections_table(self):
        """Refresh the detections table."""
        self._update_detections_table()
    
    def _toggle_detection_enabled(self, event):
        """Toggle enabled state for a detection."""
        item = self.detections_tree.selection()[0]
        values = self.detections_tree.item(item, "values")
        index = int(item)
        
        # Find the detection
        if 0 <= index < len(self.analysis.filler_detections):
            det = self.analysis.filler_detections[index]
            det.enabled = not det.enabled
            
            # Update the analysis
            self.analysis = self.app.update_detection_enabled(
                self.analysis, index, det.enabled
            )
            
            # Update the table
            self._update_detections_table()
            
            # Update summary
            self._update_after_analysis()
    
    def _enable_all_detections(self):
        """Enable all detections."""
        if self.analysis is None:
            return
        
        for det in self.analysis.filler_detections:
            det.enabled = True
        
        self.analysis = self.app.update_detection_enabled(
            self.analysis, 0, True
        )
        self._update_detections_table()
        self._update_after_analysis()
    
    def _disable_all_detections(self):
        """Disable all detections."""
        if self.analysis is None:
            return
        
        for det in self.analysis.filler_detections:
            det.enabled = False
        
        self.analysis = self.app.update_detection_enabled(
            self.analysis, 0, False
        )
        self._update_detections_table()
        self._update_after_analysis()
    
    def _update_export_button_states(self):
        """Update export button states based on analysis availability."""
        has_analysis = self.analysis is not None
        analyzing = (
            self.analysis_thread is not None and 
            self.analysis_thread.is_alive()
        )
        
        state = tk.NORMAL if has_analysis and not analyzing else tk.DISABLED
        
        for btn in self.export_buttons:
            btn["state"] = state
    
    def _update_analysis_state(self):
        """Update state when media file changes."""
        # Clear previous analysis
        self.analysis = None
        self._last_analysis = None
        
        # Clear UI
        self.media_info_text.delete(1.0, tk.END)
        self.summary_text.delete(1.0, tk.END)
        
        for item in self.detections_tree.get_children():
            self.detections_tree.delete(item)
        
        self.export_info_text.delete(1.0, tk.END)
        
        # Update button states
        self._update_export_button_states()
        
        # If we have a valid file, show its basic info
        if self.media_path and os.path.exists(self.media_path):
            try:
                from .media import inspect_media
                media_info = inspect_media(self.media_path)
                info_text = (
                    f"File: {os.path.basename(self.media_path)}\n"
                    f"Duration: {media_info.duration:.2f} seconds"
                )
                self.media_info_text.insert(tk.END, info_text)
            except Exception as e:
                self.media_info_text.insert(tk.END, f"Error loading media: {e}")
    
    def _export_fcpxml(self):
        """Export FCPXML timeline."""
        self._export("fcpxml")
    
    def _export_srt(self):
        """Export SRT subtitles."""
        self._export("srt")
    
    def _export_json(self):
        """Export JSON analysis."""
        self._export("json")
    
    def _export_all(self):
        """Export all formats."""
        if self.analysis is None:
            return
        
        output_dir = self.output_dir_var.get()
        if not output_dir:
            messagebox.showerror("Error", "Please select an output directory")
            return
        
        base_name = self.base_filename_var.get()
        if not base_name:
            messagebox.showerror("Error", "Please enter a base filename")
            return
        
        try:
            self.app.export_all(
                self.analysis,
                output_dir,
                base_name=base_name
            )
            
            # Update export info
            info = (
                f"Exported FCPXML: {os.path.join(output_dir, base_name + '.fcpxml')}\n"
                f"Exported SRT: {os.path.join(output_dir, base_name + '.srt')}\n"
                f"Exported JSON: {os.path.join(output_dir, base_name + '.json')}"
            )
            self.export_info_text.delete(1.0, tk.END)
            self.export_info_text.insert(tk.END, info)
            
            messagebox.showinfo("Success", "All files exported successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {e}")
    
    def _export(self, format: str):
        """Export a single format."""
        if self.analysis is None:
            return
        
        output_dir = self.output_dir_var.get()
        if not output_dir:
            messagebox.showerror("Error", "Please select an output directory")
            return
        
        base_name = self.base_filename_var.get()
        if not base_name:
            base_name = os.path.splitext(
                os.path.basename(self.media_path)
            )[0]
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{base_name}.{format}")
            
            if format == "fcpxml":
                self.app.export_timeline(self.analysis, output_path)
            elif format == "srt":
                self.app.export_srt(self.analysis, output_path)
            elif format == "json":
                self.app.export_json(self.analysis, output_path)
            
            # Update export info
            info = f"Exported {format.upper()}: {output_path}"
            self.export_info_text.delete(1.0, tk.END)
            self.export_info_text.insert(tk.END, info)
            
            messagebox.showinfo("Success", f"{format.upper()} exported successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {e}")
    
    def _show_cut_padding_dialog(self):
        """Show dialog to set cut padding."""
        # Simple approach: just update the variable
        # Could add a proper dialog if needed
        pass
    
    def _sort_column(self, col, reverse):
        """Sort treeview by column."""
        # This is a placeholder - would need to implement sorting
        pass
    
    def _clear_log(self):
        """Clear the log text."""
        self.log_text.delete(1.0, tk.END)
    
    def _show_about(self):
        """Show about dialog."""
        about_text = """
        Resolve AutoCut
        
        AI-assisted video cleanup for DaVinci Resolve.
        Local, open-source, Apple Silicon optimized.
        
        Version: 0.1.0
        
        Filler detection powered by UHM by Desert Ant Labs.
        Transcription powered by MLX Whisper.
        
        MIT License - Copyright (c) 2026 Resolve AutoCut Team
        """
        messagebox.showinfo("About Resolve AutoCut", about_text)


def run_gui():
    """Run the Resolve AutoCut GUI."""
    root = tk.Tk()
    app = ResolveAutoCutGUI(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()
