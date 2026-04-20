import sys
import os
import asyncio
import json
import re
import shutil
import time
import enum
import queue
import threading
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, TypedDict
from datetime import datetime
import hashlib
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk # Added for ttk widgets

# Data Science / PDF Processing
import pandas as pd
import numpy as np
import pdfplumber
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Gemini SDK
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    
    # NEW SDK FOR SUMMARIZATION (THINKING MODE)
    from google import genai as new_genai
    from google.genai import types as new_types
except ImportError:
    print("[CRITICAL] google-generativeai or google-genai library not installed.")
    sys.exit(1)

# Token Counting
try:
    import tiktoken
except ImportError:
    print("[WARNING] tiktoken not installed. Token counting will be approximate.")
    tiktoken = None

# PDF Rendering & Markdown Conversion
try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None
try:
    import fitz  # PyMuPDF
    import pymupdf4llm # PDF to Markdown
except ImportError:
    fitz = None
    print("[CRITICAL] PyMuPDF (fitz) or pymupdf4llm not installed.")

# Global safety settings for Gemini API calls
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# ==========================================
# MODEL PRICING DATA
# ==========================================
MODEL_PRICING_DATA = [
  {
    "display_name": "Gemini 3 Flash Lite",
    "model_id": "gemini-3.1-flash-lite-preview",
    "pricing": [
      {
        "type": "text",
        "context": "all context lengths",
        "input_usd": 0.25,
        "output_usd": 1.50
      }
    ],
    "knowledge_cutoff": "Jan 2025"
  },
  {
    "display_name": "Gemini 3 Flash",
    "model_id": "gemini-3-flash-preview",
    "pricing": [
      {
        "type": "text",
        "context": "all context lengths",
        "input_usd": 0.50,
        "output_usd": 3.00
      }
    ],
    "knowledge_cutoff": "Jan 2025"
  },
  {
    "display_name": "Gemini 2.0 Flash",
    "model_id": "gemini-2.0-flash",
    "pricing": [
      {
        "type": "text",
        "context": "all context lengths",
        "input_usd": 0.10,
        "output_usd": 0.40
      }
    ],
    "knowledge_cutoff": "Aug 2024"
  },
  {
    "display_name": "Gemini 2.0 Flash-Lite",
    "model_id": "gemini-2.0-flash-lite",
    "pricing": [
      {
        "type": "text",
        "context": "all context lengths",
        "input_usd": 0.075,
        "output_usd": 0.30
      }
    ],
    "knowledge_cutoff": "Aug 2024"
  },
  {
    "display_name": "Gemini 3.1 Pro (Preview)",
    "model_id": "gemini-3.1-pro-preview",
    "pricing": [
      {
        "type": "text",
        "context": "<= 200K tokens",
        "input_usd": 2.00,
        "output_usd": 12.00
      },
      {
        "type": "text",
        "context": "> 200K tokens",
        "input_usd": 4.00,
        "output_usd": 18.00
      }
    ],
    "knowledge_cutoff": "Jan 2025"
  }
]

API_USAGE_STATS = []

# ==========================================
# AVAILABLE MODELS FOR SELECTION
# ==========================================
AVAILABLE_MODELS = {
    "gemini-3-flash-preview": "Gemini 3 Flash",
    "gemini-3.1-flash-lite-preview": "Gemini 3 Flash Lite",
    "gemini-2.0-flash": "Gemini 2.0 Flash",
    "gemini-2.0-flash-lite": "Gemini 2.0 Flash-Lite",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro (Preview)"
}

DEFAULT_MODEL_CONFIG = {
    "toc_model": "gemini-3.1-flash-lite-preview",
    "visual_model": "gemini-3-flash-preview",
    "summary_model": "gemini-3-flash-preview"
}

def get_model_timeout(model_id: str) -> int:
    """Returns timeout in seconds based on model type."""
    if not model_id:
        return 300
    if "pro" in model_id.lower():
        return 600  # 10 minutes for pro models
    return 300      # 5 minutes for flash/lite models

# ==========================================
# TOKEN COUNTING & PRICING FUNCTIONS
# ==========================================
def count_tokens(text: str) -> int:
    if tiktoken is None:
        return len(text) // 4
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return len(text) // 4


# ==========================================
# GLOBAL TOC CACHING
# ==========================================
GLOBAL_CACHE_FILE = Path(__file__).parent / ".toc_cache.json"

def get_pdf_hash(pdf_path: str) -> str:
    """Generates a unique hash for a PDF file based on its size and content chunks."""
    h = hashlib.sha256()
    size = os.path.getsize(pdf_path)
    h.update(str(size).encode())
    with open(pdf_path, "rb") as f:
        # Hash first 512KB and last 512KB for speed + uniqueness
        h.update(f.read(512 * 1024))
        if size > 512 * 1024:
            f.seek(-512 * 1024, os.SEEK_END)
            h.update(f.read(512 * 1024))
    return h.hexdigest()

def load_toc_cache() -> Dict[str, Any]:
    if GLOBAL_CACHE_FILE.exists():
        try:
            with open(GLOBAL_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_toc_to_cache(pdf_hash: str, verified_mapping: List[Dict]):
    cache = load_toc_cache()
    # Save the mapping and the timestamp
    cache[pdf_hash] = {
        "mapping": verified_mapping,
        "cached_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "filename_hint": Path(pdf_hash).name # Just for debugging the json
    }
    try:
        with open(GLOBAL_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[WARNING] TOC Cache save failed: {e}")


def get_model_pricing(model_id: str, token_count: int) -> Optional[Dict[str, float]]:
    model_info = None
    for model in MODEL_PRICING_DATA:
        if model["model_id"] == model_id:
            model_info = model
            break
    if not model_info or not model_info.get("pricing"):
        return None
    for pricing_tier in model_info["pricing"]:
        if pricing_tier["type"] != "text":
            continue
        context = pricing_tier["context"]
        if "<=" in context:
            threshold = int(context.split("<=")[1].replace("K", "000").split()[0])
            if token_count <= threshold:
                return {"input_per_million": pricing_tier["input_usd"], "output_per_million": pricing_tier["output_usd"], "tier": context}
        elif ">" in context:
            threshold = int(context.split(">")[1].replace("K", "000").split()[0])
            if token_count > threshold:
                return {"input_per_million": pricing_tier["input_usd"], "output_per_million": pricing_tier["output_usd"], "tier": context}
        else:
            return {"input_per_million": pricing_tier["input_usd"], "output_per_million": pricing_tier["output_usd"], "tier": context}
    return None

def calculate_api_cost(input_tokens: int, output_tokens: int, model_id: str) -> Dict[str, Any]:
    pricing = get_model_pricing(model_id, input_tokens)
    if not pricing:
        return {"model": model_id, "input_tokens": input_tokens, "output_tokens": output_tokens, "estimated_cost_usd": None, "error": "Pricing not available"}
    input_cost = (input_tokens / 1_000_000) * pricing["input_per_million"]
    output_cost = (output_tokens / 1_000_000) * pricing["output_per_million"]
    total_cost = input_cost + output_cost
    return {
        "model": model_id, "input_tokens": input_tokens, "output_tokens": output_tokens,
        "pricing_tier": pricing["tier"], "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6), "estimated_cost_usd": round(total_cost, 6)
    }

def safe_rmtree(path: Path, retries: int = 3) -> None:
    if not path.exists():
        return
    import gc
    for attempt in range(retries):
        try:
            gc.collect()
            time.sleep(0.5)
            shutil.rmtree(path)
            return
        except PermissionError:
            if attempt < retries - 1:
                print(f"    [CLEANUP] Retry {attempt + 1}/{retries} for {path.name}...")
                time.sleep(1)
            else:
                print(f"    [CLEANUP] Could not clean up {path.name}. Delete manually.")
        except Exception as e:
            print(f"    [CLEANUP] Error for {path.name}: {e}")
            return


# ==========================================
# GUI FILE SELECTION
# ==========================================
def select_pdf_file() -> Optional[str]:
    print("\n[UI] Opening file selection dialog...")
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(
        title="Select PDF File",
        filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
    )
    root.destroy()
    if file_path:
        print(f"[OK] Selected: {Path(file_path).name}")
        return file_path
    else:
        print("[ERROR] No file selected.")
        return None


def create_configuration_dialog(pdf_path: str) -> Optional[Dict[str, str]]:
    print("\n[UI] Opening configuration dialog...")
    pdf_name = Path(pdf_path).name
    file_size_mb = Path(pdf_path).stat().st_size / (1024 * 1024)

    print("[ANALYSIS] Analyzing PDF and pre-calculating costs...")
    cost_matrix = calculate_all_model_costs(pdf_path) # Assuming this function exists elsewhere
    total_chars = cost_matrix["metadata"]["total_chars"]
    total_pages = cost_matrix["metadata"]["total_pages"]

    result = {"cancelled": True}

    # Create an independent window for the dialog
    dialog = tk.Tk()
    dialog.title("Book Chapter Summarizer - Configuration")
    dialog.geometry("800x900")
    dialog.attributes('-topmost', True) # Assure it appears

    style = ttk.Style()
    style.theme_use('clam')
    style.configure("Bold.TLabel", font=('Segoe UI', 9, 'bold'))
    style.configure("Total.TLabel", font=('Segoe UI', 12, 'bold'), foreground="#0066cc")

    main_canvas = tk.Canvas(dialog)
    scrollbar = ttk.Scrollbar(dialog, orient="vertical", command=main_canvas.yview)
    scrollable_frame = ttk.Frame(main_canvas)
    scrollable_frame.bind("<Configure>", lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all")))
    main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    main_canvas.configure(yscrollcommand=scrollbar.set)
    main_canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    content_frame = ttk.Frame(scrollable_frame, padding="20")
    content_frame.pack(fill="both", expand=True)

    info_frame = ttk.LabelFrame(content_frame, text="File Information", padding="10")
    info_frame.pack(fill="x", pady=(0, 15))
    ttk.Label(info_frame, text=f"File: {pdf_name}", font=('Segoe UI', 10, 'bold')).grid(row=0, column=0, sticky=tk.W, columnspan=3)
    ttk.Label(info_frame, text=f"Size: {file_size_mb:.2f} MB").grid(row=1, column=0, sticky=tk.W, padx=(0, 20))
    ttk.Label(info_frame, text=f"Pages: {total_pages}").grid(row=1, column=1, sticky=tk.W, padx=(0, 20))
    ttk.Label(info_frame, text=f"Characters: {total_chars:,}").grid(row=1, column=2, sticky=tk.W)

    toc_var = tk.StringVar(value=DEFAULT_MODEL_CONFIG["toc_model"])
    visual_var = tk.StringVar(value=DEFAULT_MODEL_CONFIG["visual_model"])
    summary_var = tk.StringVar(value=DEFAULT_MODEL_CONFIG["summary_model"])
    total_cost_var = tk.StringVar(value="$0.00")

    def format_cost(cost: float) -> str:
        if cost == 0:
            return "0.00"
        if cost >= 0.005:
            return f"{round(cost, 2):.2f}"
        # For small numbers, show enough precision (up to 8 decimals) to see the value
        formatted = f"{cost:.8f}".rstrip('0').rstrip('.')
        if formatted == "0" or formatted == "":
            return f"{cost:.2e}" # Handle extremely tiny values
        return formatted

    def update_total(*args):
        try:
            t_cost = cost_matrix["toc_models"].get(toc_var.get(), {}).get("cost_usd", 0) or 0
            v_cost = cost_matrix["visual_models"].get(visual_var.get(), {}).get("cost_usd", 0) or 0
            s_cost = cost_matrix["summary_models"].get(summary_var.get(), {}).get("cost_usd", 0) or 0
            total = t_cost + v_cost + s_cost
            total_cost_var.set(f"${format_cost(total)} USD")
        except Exception as e:
            total_cost_var.set("Error calculating")

    toc_var.trace("w", update_total)
    visual_var.trace("w", update_total)
    summary_var.trace("w", update_total)

    def create_model_table(parent, title, variable, step_key):
        frame = ttk.LabelFrame(parent, text=title, padding="10")
        frame.pack(fill="x", pady=(0, 15))
        ttk.Label(frame, text="Select", style="Bold.TLabel").grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(frame, text="Model", style="Bold.TLabel").grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Label(frame, text="Est. Cost", style="Bold.TLabel").grid(row=0, column=2, padx=5, pady=5, sticky=tk.E)
        for row_idx, (model_id, model_name) in enumerate(sorted(AVAILABLE_MODELS.items(), key=lambda x: x[1]), 1):
            step_cost = cost_matrix[step_key].get(model_id, {}).get("cost_usd", 0) or 0
            rb = ttk.Radiobutton(frame, variable=variable, value=model_id)
            rb.grid(row=row_idx, column=0, padx=5, pady=2)
            lbl = ttk.Label(frame, text=model_name)
            lbl.grid(row=row_idx, column=1, padx=5, pady=2, sticky=tk.W)
            lbl.bind("<Button-1>", lambda e, r=rb: r.invoke())
            ttk.Label(frame, text=f"${format_cost(step_cost)}").grid(row=row_idx, column=2, padx=5, pady=2, sticky=tk.E)

    create_model_table(content_frame, "Step 1: Table of Contents Extraction", toc_var, "toc_models")
    create_model_table(content_frame, "Step 2: Chapter Title Verification (Targeted Vision)", visual_var, "visual_models")
    create_model_table(content_frame, "Step 3: Chapter Summarization", summary_var, "summary_models")

    total_frame = ttk.Frame(content_frame, padding="15", relief="raised", borderwidth=1)
    total_frame.pack(fill="x", pady=(10, 0))
    ttk.Label(total_frame, text="TOTAL ESTIMATED COST:", font=('Segoe UI', 11)).pack(side="left")
    ttk.Label(total_frame, textvariable=total_cost_var, style="Total.TLabel").pack(side="right")
    
    # Toggle for skipping summarization
    run_summaries_var = tk.BooleanVar(value=True)
    summarize_check = ttk.Checkbutton(content_frame, text="Generate Summaries (Skip if you just want extracted markdown)", variable=run_summaries_var)
    summarize_check.pack(pady=10)

    button_frame = ttk.Frame(content_frame)
    button_frame.pack(pady=20)

    def on_process():
        result.clear()
        result.update({
            "toc_model": toc_var.get(), 
            "visual_model": visual_var.get(), 
            "summary_model": summary_var.get(), 
            "run_summaries": run_summaries_var.get(),
            "cancelled": False
        })
        dialog.quit() # Stop mainloop without destroying yet

    def on_cancel():
        result["cancelled"] = True
        dialog.quit() # Stop mainloop

    ttk.Button(button_frame, text="Process Book", command=on_process, width=20).grid(row=0, column=0, padx=10)
    ttk.Button(button_frame, text="Cancel", command=on_cancel, width=20).grid(row=0, column=1, padx=10)

    update_total()
    dialog.update_idletasks()
    x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
    y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
    dialog.geometry(f'+{x}+{y}')

    # Start main event loop for this dialog blocking until quit()
    dialog.mainloop()
    try:
        dialog.destroy()
    except tk.TclError:
        pass # Handle case where window was already closed manually

    if result.get("cancelled", True):
        print("[CANCELLED] Configuration cancelled.")
        return None
    print("[OK] Configuration confirmed.")
    return {k: v for k, v in result.items() if k != "cancelled"}


# ==========================================
# OUTPUT FOLDER MANAGEMENT
# ==========================================
def create_output_folder(pdf_path: str) -> Path:
    pdf_name = Path(pdf_path).stem
    if len(pdf_name) > 60:
        pdf_name = pdf_name[:60].strip()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{pdf_name}_{timestamp}"
    # BATCH MODE: if OUTPUT_BASE_DIR is set, use that instead of the PDF's folder
    env_base = os.environ.get("OUTPUT_BASE_DIR")
    if env_base:
        base_dir = Path(env_base)
    else:
        # Default: output next to the PDF
        base_dir = Path(os.path.dirname(os.path.abspath(pdf_path))) / "output" / "books"
    output_folder = base_dir / folder_name
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"[FOLDER] Output folder created: {output_folder}")
    return output_folder


# ==========================================
# UI PROGRESS TRACKER
# ==========================================
class ProgressWindow:
    def __init__(self):
        self.queue = queue.Queue()
        self.closed = False
        
        self.window = tk.Tk()
        self.root = self.window  # Alias for backward compatibility with .after()
        self.window.title("Book Chapter Summarizer - Progress")
        self.window.geometry("700x550")
        
        # UI Setup
        self.status_var = tk.StringVar(value="Initializing pipeline...")
        self.progress_var = tk.DoubleVar(value=0)
        
        header_frame = ttk.Frame(self.window, padding=20)
        header_frame.pack(fill="x")
        
        ttk.Label(header_frame, text="Processing Book Pipeline", font=('Segoe UI', 12, 'bold')).pack(anchor="w")
        self.status_label = ttk.Label(header_frame, textvariable=self.status_var, font=('Segoe UI', 10))
        self.status_label.pack(anchor="w", pady=(5, 10))
        
        self.pb = ttk.Progressbar(header_frame, variable=self.progress_var, maximum=100, length=600)
        self.pb.pack(fill="x")
        
        log_frame = ttk.Frame(self.window, padding=(20, 0, 20, 20))
        log_frame.pack(fill="both", expand=True)
        
        self.log_text = tk.Text(log_frame, height=15, width=80, font=('Consolas', 9), bg="#f5f5f5", relief="flat")
        self.log_text.pack(side="left", fill="both", expand=True)
        
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        # Centering and display
        self.window.update_idletasks()
        x = (self.window.winfo_screenwidth() // 2) - (self.window.winfo_width() // 2)
        y = (self.window.winfo_screenheight() // 2) - (self.window.winfo_height() // 2)
        self.window.geometry(f'+{x}+{y}')
        self.window.attributes('-topmost', True)
        
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self.check_queue)

    def on_close(self):
        self.closed = True
        self.window.destroy()

    def update(self, msg, progress=None):
        self.queue.put(("log", msg, progress))

    def check_queue(self):
        if self.closed: return
        while not self.queue.empty():
            try:
                type, msg, progress = self.queue.get_nowait()
                if type == "log":
                    self.log_text.insert(tk.END, f"{msg}\n")
                    self.log_text.see(tk.END)
                    self.status_var.set(msg if len(msg) < 100 else msg[:97] + "...")
                    if progress is not None:
                        self.progress_var.set(progress)
            except:
                pass
        self.root.after(100, self.check_queue)


# ==========================================
# MODEL CONTEXT WINDOWS
# ==========================================
MODEL_CONTEXT_WINDOWS = {
    "gemini-3-flash-preview": 1000000,
    "gemini-3.1-flash-lite-preview": 1000000,
    "gemini-3.1-pro-preview": 2000000,
    "gemini-2.0-flash": 1000000,
    "gemini-2.0-flash-lite": 1000000
}

# ==========================================
# API LOGGING
# ==========================================
def log_api_call(title: str, model_id: str, prompt: str, response_obj: Any, error: Optional[Exception] = None, process_log: Optional[List[str]] = None) -> Dict[str, Any]:
    log_entry = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "model": model_id,
        "status": "success" if error is None else "error"
    }
    input_tokens = count_tokens(prompt)
    context_limit = MODEL_CONTEXT_WINDOWS.get(model_id, "unknown")
    log_entry["input_tokens"] = input_tokens
    log_entry["context_limit"] = context_limit
    log_entry["context_usage_pct"] = round((input_tokens / context_limit * 100), 2) if isinstance(context_limit, int) else "N/A"

    if error is None and response_obj:
        try:
            output_text = response_obj.text if hasattr(response_obj, 'text') else str(response_obj)
            output_tokens = count_tokens(output_text)
            log_entry["output_tokens"] = output_tokens
            log_entry["output_length_chars"] = len(output_text)
            if hasattr(response_obj, 'prompt_feedback'):
                feedback = response_obj.prompt_feedback
                if feedback and hasattr(feedback, 'block_reason'):
                    log_entry["safety_blocked"] = True
                    log_entry["block_reason"] = str(feedback.block_reason)
            if hasattr(response_obj, 'candidates') and response_obj.candidates:
                candidate = response_obj.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    log_entry["finish_reason"] = str(candidate.finish_reason)
        except Exception as parse_error:
            log_entry["parse_error"] = str(parse_error)

    if error:
        log_entry["error_type"] = type(error).__name__
        log_entry["error_message"] = str(error)

    if process_log is not None:
        log_msg = f"  [API] {title}: {log_entry['status'].upper()}"
        if error:
            log_msg += f" - {log_entry.get('error_type')}: {log_entry.get('error_message')[:100]}"
        else:
            log_msg += f" - In:{input_tokens}tk Out:{log_entry.get('output_tokens', 'N/A')}tk"
        process_log.append(log_msg)

    API_USAGE_STATS.append(log_entry)
    return log_entry


# ==========================================
# OUTPUT FILE GENERATION
# ==========================================
def clean_summary_text(summary: str) -> str:
    if not summary or summary.strip() == "":
        return "No summary available."
    summary = summary.strip()
    
    # Remove markers if present (legacy and new)
    markers = ["SHORT:", "LONG:", "# Short Summary", "# Long Summary"]
    for marker in markers:
        if summary.upper().startswith(marker.upper()):
            summary = summary[len(marker):].strip()
            # If after stripping one marker we find another (e.g. from mixed-up extraction), 
            # we don't recursive but this handles the basic case.
            break
            
    return summary if summary else "No summary available."


def generate_output_files(output_folder: Path, chapters: List[Dict], process_log: List[str], summaries_generated: bool = True):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = output_folder / "chapters_full_data.json"
    clean_chapters = []
    
    # Generate individual markdown files per chapter
    chapters_dir = output_folder / "chapters_markdown"
    chapters_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, ch in enumerate(chapters, start=1):
        title = ch.get('chapter_title', f"Chapter_{idx}")
        # Sanitize filename
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        if len(safe_title) > 50:
            safe_title = safe_title[:50].strip()
        filename = f"{idx:02d} - {safe_title}.md"
        
        md_content = f"# {title}\n\n"
        if 'pdf_display_page_number' in ch:
            md_content += f"*Starts at PDF Page {ch['pdf_display_page_number']}*\n\n"
            
        md_content += ch.get("full_text_markdown", "No text extracted.")
        
        with open(chapters_dir / filename, "w", encoding="utf-8") as f:
            f.write(md_content)
            
        # Cleaned dict for JSON
        clean_ch = {
            "order": idx,
            "title": title,
            "page": ch.get("pdf_display_page_number"),
            "full_text_markdown": ch.get("full_text_markdown", "")
        }
        if summaries_generated:
            clean_ch["short_summary"] = ch.get("short_summary", "")
            clean_ch["long_summary"] = ch.get("long_summary", "")
            
        clean_chapters.append(clean_ch)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(clean_chapters, f, indent=2, ensure_ascii=False)
    print(f"[OK] Generated: {json_path.name}")
    print(f"[OK] Generated individual markdown files in {chapters_dir.name}/")

    if summaries_generated:
        short_md_path = output_folder / "short_summaries.md"
        with open(short_md_path, "w", encoding="utf-8") as f:
            f.write("# Short Summaries\n\n")
            f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n---\n\n")
            for ch in chapters:
                f.write(f"## {ch.get('chapter_title', 'Unknown')}\n\n{clean_summary_text(ch.get('short_summary', ''))}\n\n---\n\n")
        print(f"[OK] Generated: {short_md_path.name}")

        long_md_path = output_folder / "long_summaries.md"
        with open(long_md_path, "w", encoding="utf-8") as f:
            f.write("# Long Summaries\n\n")
            f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n---\n\n")
            for ch in chapters:
                f.write(f"## {ch.get('chapter_title', 'Unknown')}\n\n{clean_summary_text(ch.get('long_summary', ''))}\n\n---\n\n")
        print(f"[OK] Generated: {long_md_path.name}")


    # Calculate final cost stats
    total_cost = 0.0
    total_in = 0
    total_out = 0
    for entry in API_USAGE_STATS:
        if entry.get("status") == "success":
            in_tk = entry.get("input_tokens", 0)
            out_tk = entry.get("output_tokens", 0)
            model_id = entry.get("model")
            cost_info = calculate_api_cost(in_tk, out_tk, model_id)
            total_cost += cost_info.get("estimated_cost_usd", 0) or 0
            total_in += in_tk
            total_out += out_tk

    cost_summary = [
        "",
        "=" * 40,
        "FINAL ACTUAL COST SUMMARY",
        "=" * 40,
        f"Total Input Tokens:  {total_in:,}",
        f"Total Output Tokens: {total_out:,}",
        f"Actual Total Cost:   ${total_cost:.6f} USD",
        "=" * 40
    ]

    log_path = output_folder / f"process_log_{timestamp}.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("BOOK CHAPTER SUMMARIZER - PROCESS LOG\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for log_entry in process_log:
            f.write(log_entry + "\n")
        for line in cost_summary:
            f.write(line + "\n")
            
    print(f"[OK] Generated: {log_path.name}")
    for line in cost_summary:
        print(line)


@lru_cache(maxsize=16)
def get_pdf_character_count(pdf_path: str) -> Tuple[int, int]:
    total_chars = 0
    total_pages = 0
    try:
        if fitz is not None:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            for page in doc:
                text = page.get_text() or ""
                total_chars += len(text)
            doc.close()
        else:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    total_chars += len(text)
    except Exception as e:
        print(f"[WARNING] Could not extract text for char count: {e}")
        total_chars = total_pages * 2000
    return total_chars, total_pages


def calculate_upfront_pricing(pdf_path: str, model_config: Dict[str, str], total_chars: Optional[int] = None, total_pages: Optional[int] = None) -> Dict[str, Any]:
    if total_chars is None or total_pages is None:
        total_chars, total_pages = get_pdf_character_count(pdf_path)
    estimated_chapters = max(1, total_chars // 25000)
    steps = []

    # 1. TOC Extraction: ~20 images + prompt
    toc_input_tokens = count_tokens("Extract the main Table of Contents from these images.") + (20 * 258)
    toc_output_tokens = 200
    toc_cost = calculate_api_cost(toc_input_tokens, toc_output_tokens, model_config["toc_model"])
    steps.append({
        "name": "TOC Extraction", "model": model_config["toc_model"],
        "input_tokens": toc_input_tokens, "output_tokens": toc_output_tokens,
        "cost_usd": toc_cost.get("estimated_cost_usd", 0) or 0
    })

    # 2. Targeted Vision: ~7 pages per chapter (predicted +/- 3)
    pages_per_chapter = 7
    visual_calls = estimated_chapters  # one batch call per chapter
    visual_input_tokens = (count_tokens("Does this chapter title appear as a heading?") + 258 * pages_per_chapter) * visual_calls
    visual_output_tokens = 30 * visual_calls
    visual_cost = calculate_api_cost(visual_input_tokens, visual_output_tokens, model_config["visual_model"])
    steps.append({
        "name": "Chapter Title Verification", "model": model_config["visual_model"],
        "input_tokens": visual_input_tokens, "output_tokens": visual_output_tokens,
        "cost_usd": visual_cost.get("estimated_cost_usd", 0) or 0,
        "chapters_verified": estimated_chapters
    })

    # 3. Summarization
    markdown_overhead_multiplier = 1.15
    output_ratio = 0.11
    
    # Estimate base input tokens as the entire book with markdown overhead
    summary_total_input = int((total_chars * markdown_overhead_multiplier) / 3.5)
    
    # Estimate dynamic output tokens based on 11% ratio
    summary_total_output = int(summary_total_input * output_ratio)
    
    summary_cost = calculate_api_cost(summary_total_input, summary_total_output, model_config["summary_model"])
    steps.append({
        "name": "Summarization", "model": model_config["summary_model"],
        "input_tokens": summary_total_input, "output_tokens": summary_total_output,
        "cost_usd": summary_cost.get("estimated_cost_usd", 0) or 0,
        "estimated_chapters": estimated_chapters
    })

    total_cost = sum(s["cost_usd"] for s in steps)
    return {
        "steps": steps, "total_cost_usd": total_cost,
        "total_input_tokens": sum(s["input_tokens"] for s in steps),
        "total_output_tokens": sum(s["output_tokens"] for s in steps),
        "total_chars": total_chars, "total_pages": total_pages,
        "estimated_chapters": estimated_chapters
    }


def calculate_all_model_costs(pdf_path: str) -> Dict[str, Any]:
    total_chars, total_pages = get_pdf_character_count(pdf_path)
    cost_breakdown = {
        "metadata": {"total_chars": total_chars, "total_pages": total_pages, "estimated_chapters": max(1, total_chars // 25000)},
        "toc_models": {}, "visual_models": {}, "summary_models": {}
    }
    for model_id in AVAILABLE_MODELS:
        print(f"      - Pre-calculating {AVAILABLE_MODELS[model_id]}...")
        toc_cfg = {"toc_model": model_id, "visual_model": "gemini-2.0-flash", "summary_model": "gemini-2.0-flash"}
        cost_breakdown["toc_models"][model_id] = calculate_upfront_pricing(pdf_path, toc_cfg, total_chars, total_pages)["steps"][0]

        vis_cfg = {"toc_model": "gemini-2.0-flash", "visual_model": model_id, "summary_model": "gemini-2.0-flash"}
        cost_breakdown["visual_models"][model_id] = calculate_upfront_pricing(pdf_path, vis_cfg, total_chars, total_pages)["steps"][1]

        sum_cfg = {"toc_model": "gemini-2.0-flash", "visual_model": "gemini-2.0-flash", "summary_model": model_id}
        cost_breakdown["summary_models"][model_id] = calculate_upfront_pricing(pdf_path, sum_cfg, total_chars, total_pages)["steps"][2]
    return cost_breakdown


# ==========================================
# CONFIGURATION & AUTH
# ==========================================
print("--- KEY AUTHENTICATION CHECK ---")
API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
if not API_KEY:
    # Optional: pick up GEMINI_API_KEY from a .env file next to this script
    # if python-dotenv is installed.
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent / ".env")
        API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
    except ImportError:
        pass

if API_KEY:
    genai.configure(api_key=API_KEY)
    print("API Key loaded from GEMINI_API_KEY environment variable.")
else:
    print("[CRITICAL] GEMINI_API_KEY environment variable is not set.")
    print("           Get a key at https://aistudio.google.com/apikey")
    print("           Then set it, for example:")
    print("             Windows CMD:  set GEMINI_API_KEY=your_key_here")
    print("             PowerShell:   $env:GEMINI_API_KEY='your_key_here'")
    print("             bash/zsh:     export GEMINI_API_KEY=your_key_here")
    print("           Or drop it in a .env file next to this script.")
    sys.exit(1)

# ==========================================
# STRUCTURED OUTPUT SCHEMAS
# ==========================================
class TocItem(TypedDict):
    chapter_title: str
    printed_page_number: int

class ChapterSummary(TypedDict):
    short_summary: str
    long_summary: str

class ModelConfiguration(TypedDict):
    toc_model: str
    visual_model: str
    summary_model: str

class VerifyResult(TypedDict):
    page_index: int


# ==========================================
# STEP 2: DENSE PAGE NUMBER EXTRACTION
# ==========================================
def extract_dense_page_numbers(pdf_path: str) -> Dict[str, Any]:
    """Extract candidate page numbers from ALL pages using pdfplumber text.

    Applies conservative noise filters to remove obviously impossible candidates
    before passing data to RANSAC. The filters are intentionally loose since
    RANSAC handles remaining noise fine.

    Returns dict with:
      - total_pages: int
      - page_candidates: dict mapping pdf_index -> list of candidate ints
      - extractable_ratio: fraction of pages that yielded at least one candidate
    """
    print(f"\n[STEP 2] Extracting page numbers from all pages via text...")
    page_candidates: Dict[int, List[int]] = {}

    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)

            # Conservative maximum printed page number: no book has more printed
            # pages than ~3x its PDF page count (double-page scans + front matter).
            max_plausible_printed = total_pages * 3 + 50

            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                candidates = set()
                for m in re.findall(r'\b\d{1,4}\b', text):
                    val = int(m)
                    if val <= 0:
                        continue

                    # Filter 1: reject numbers larger than any plausible page number
                    if val > max_plausible_printed:
                        continue

                    # Filter 2: reject numbers that are way too far from the pdf index.
                    # A printed page on pdf page 10 can't realistically be 200, and
                    # a printed page on pdf page 300 can't be 5.
                    # Use a generous window: allow up to 60% of total_pages as offset,
                    # with a floor of 30 pages so short books aren't over-filtered.
                    max_offset = max(30, int(total_pages * 0.6))
                    if abs(val - i) > max_offset:
                        continue

                    # Filter 3: reject year-like numbers (1900-2099) on early pages
                    # where they're almost certainly publication dates, not page numbers.
                    # Only apply this in the first 5% of the PDF where front matter lives.
                    if 1900 <= val <= 2099 and i < total_pages * 0.05:
                        continue

                    candidates.add(val)
                page_candidates[i] = sorted(candidates)
    except Exception as e:
        print(f"    [ERROR] pdfplumber could not open '{Path(pdf_path).name}': {e}")
        return {
            "total_pages": 0,
            "page_candidates": {},
            "extractable_ratio": 0
        }

    extractable = sum(1 for v in page_candidates.values() if v)
    ratio = extractable / total_pages if total_pages > 0 else 0
    print(f"    Pages: {total_pages}, Pages with candidates: {extractable} ({ratio:.0%})")

    return {
        "total_pages": total_pages,
        "page_candidates": page_candidates,
        "extractable_ratio": ratio
    }


# ==========================================
# STEP 3: ROBUST LINEAR FIT (RANSAC-style)
# ==========================================
def build_robust_page_mapping(dense_data: Dict[str, Any]) -> Dict[str, Any]:
    """Fit a robust linear model: printed_page = slope * pdf_index + intercept.

    Uses RANSAC approach: for many random pairs, fit a line, count inliers,
    keep the best. The true mapping has slope ~1 (single) or ~2 (double)
    and is strictly monotonic, so even with noisy candidates the correct
    line dominates.

    Returns dict with:
      - slope, intercept: fitted line parameters
      - mode: 'single' or 'double'
      - inlier_count: how many (pdf_idx, number) pairs agree with the fit
      - mapping_func: callable(printed_page) -> pdf_index (float)
      - reverse_func: callable(pdf_index) -> printed_page (float)
    """
    print("\n[STEP 3] Building robust page mapping (RANSAC fit)...")

    total_pages = dense_data["total_pages"]
    page_candidates = dense_data["page_candidates"]

    # Build all (pdf_idx, printed_number) pairs.
    # The heavy noise filtering was already done in extract_dense_page_numbers,
    # so we just take everything that survived.
    points = []
    for pdf_idx, candidates in page_candidates.items():
        for num in candidates:
            points.append((pdf_idx, num))

    if len(points) < 5:
        # Fallback: not enough data, assume 1:1 mapping with offset 0
        print("    [WARNING] Too few data points for robust fit. Using identity mapping.")
        return {
            "slope": 1.0, "intercept": 0.0, "mode": "single",
            "inlier_count": 0,
            "mapping_func": lambda printed: float(printed),
            "reverse_func": lambda pdf_idx: float(pdf_idx)
        }

    points_arr = np.array(points, dtype=np.float64)
    pdf_indices = points_arr[:, 0]
    printed_nums = points_arr[:, 1]

    best_inliers = 0
    best_slope = 1.0
    best_intercept = 0.0

    rng = np.random.default_rng(42)
    n_iterations = min(2000, len(points) * 10)
    inlier_threshold = 2.0  # allow +/- 2 pages of error

    for _ in range(n_iterations):
        # Pick 2 random points
        idxs = rng.choice(len(points), size=2, replace=False)
        x1, y1 = points[idxs[0]]
        x2, y2 = points[idxs[1]]

        if abs(x2 - x1) < 3:
            continue  # too close, skip

        slope = (y2 - y1) / (x2 - x1)

        # Only consider plausible slopes: single page (0.7..1.4) or double page (1.5..2.5)
        if not (0.5 <= slope <= 3.0):
            continue

        intercept = y1 - slope * x1

        # Count inliers
        predicted = slope * pdf_indices + intercept
        errors = np.abs(printed_nums - predicted)
        inlier_mask = errors <= inlier_threshold
        inlier_count = int(inlier_mask.sum())

        if inlier_count > best_inliers:
            best_inliers = inlier_count
            best_slope = slope
            best_intercept = intercept

    # Refine with least-squares on inliers
    predicted = best_slope * pdf_indices + best_intercept
    inlier_mask = np.abs(printed_nums - predicted) <= inlier_threshold
    if inlier_mask.sum() >= 2:
        inlier_x = pdf_indices[inlier_mask]
        inlier_y = printed_nums[inlier_mask]
        # Least squares: y = slope * x + intercept
        A = np.vstack([inlier_x, np.ones(len(inlier_x))]).T
        result = np.linalg.lstsq(A, inlier_y, rcond=None)
        best_slope, best_intercept = result[0]

    # Determine layout mode from slope
    if best_slope >= 1.5:
        mode = "double"
    else:
        mode = "single"

    print(f"    Fit: printed = {best_slope:.4f} * pdf_idx + {best_intercept:.2f}")
    print(f"    Layout mode: {mode.upper()} (slope={best_slope:.3f})")
    print(f"    Inliers: {best_inliers}/{len(points)} candidate pairs")

    # mapping_func: given a printed page number, return the predicted pdf index
    def mapping_func(printed_page: float) -> float:
        return (printed_page - best_intercept) / best_slope

    # reverse_func: given a pdf index, return the predicted printed page
    def reverse_func(pdf_idx: float) -> float:
        return best_slope * pdf_idx + best_intercept

    return {
        "slope": best_slope,
        "intercept": best_intercept,
        "mode": mode,
        "inlier_count": best_inliers,
        "mapping_func": mapping_func,
        "reverse_func": reverse_func
    }


# ==========================================
# STEP 4: MAP TOC TO PDF PAGES
# ==========================================
def map_toc_to_pdf(toc_entries: List[Dict], fit: Dict[str, Any], total_pages: int) -> List[Dict]:
    """Use the linear fit to map each TOC printed page number to a pdf index.

    Returns list of dicts with chapter_title, printed_page, predicted_pdf_index.
    """
    print("\n[STEP 4] Mapping TOC entries to PDF pages via linear fit...")
    mapping_func = fit["mapping_func"]
    mapped = []

    for i, entry in enumerate(toc_entries):
        title = entry.get("chapter_title", "Unknown")
        raw_printed = entry.get("printed_page_number")

        printed = None
        if raw_printed is not None:
            try:
                val = int(raw_printed)
                if val > 0:
                    printed = val
            except (ValueError, TypeError):
                pass

        if printed is None:
            # No page number: estimate from position in TOC
            estimated = int((i / max(1, len(toc_entries))) * total_pages)
            estimated = max(0, min(estimated, total_pages - 1))
            mapped.append({
                "chapter_title": title, "printed_page": 0,
                "predicted_pdf_index": estimated, "status": "Estimated (No Page Number)"
            })
            continue

        pdf_idx_float = mapping_func(printed)
        pdf_idx = int(round(pdf_idx_float))
        pdf_idx = max(0, min(pdf_idx, total_pages - 1))

        mapped.append({
            "chapter_title": title, "printed_page": printed,
            "predicted_pdf_index": pdf_idx, "status": "Mapped (Linear Fit)"
        })
        print(f"    '{title}' p.{printed} -> PDF index {pdf_idx}")

    return mapped


# ==========================================
# HELPERS: RENDERING
# ==========================================
def render_specific_pages(pdf_path: str, page_indices: List[int], output_dir: Path) -> Dict[int, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered_map = {}
    valid_indices = sorted(set(p for p in page_indices if p >= 0))
    if not valid_indices:
        return {}

    def stamp_page_number(img_path: str, page_num: int):
        try:
            img = Image.open(img_path)
            draw = ImageDraw.Draw(img)
            # Try to load a nice font, fallback to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", 40)
            except:
                font = ImageFont.load_default()
            
            text = f"PDF PAGE: {page_num}"
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            text_width = right - left
            text_height = bottom - top
            
            box_y = int(img.height * 0.65)
            x = (img.width - text_width) // 2
            
            padding = 10
            draw.rectangle(
                [x - padding, box_y - padding, x + text_width + padding, box_y + text_height + padding],
                fill="red"
            )
            draw.text((x, box_y), text, fill="white", font=font)
            img.save(img_path, "JPEG", quality=85)
        except Exception as e:
            print(f"Warning: Could not stamp page number on {img_path}: {e}")

    if fitz:
        try:
            doc = fitz.open(pdf_path)
            for idx in valid_indices:
                if idx >= doc.page_count:
                    continue
                try:
                    page = doc.load_page(idx)
                    pix = page.get_pixmap(dpi=150)
                    out_path = output_dir / f"page_check_{idx}.jpg"
                    pix.save(str(out_path))
                    stamp_page_number(str(out_path), idx)
                    rendered_map[idx] = str(out_path)
                except Exception:
                    pass
            doc.close()
        except Exception as e:
            print(f"    [ERROR] PyMuPDF could not open '{Path(pdf_path).name}': {e}")

    elif convert_from_path:
        try:
            images = convert_from_path(pdf_path, dpi=150)
            for idx in valid_indices:
                if idx < len(images):
                    out_path = output_dir / f"page_check_{idx}.jpg"
                    images[idx].save(out_path, "JPEG")
                    stamp_page_number(str(out_path), idx)
                    rendered_map[idx] = str(out_path)
        except Exception as e:
            print(f"    [ERROR] pdf2image could not process '{Path(pdf_path).name}': {e}")
    return rendered_map


# ==========================================
# STEP 5: TARGETED VISION VERIFICATION
# ==========================================
async def verify_chapter_starts(
    pdf_path: str,
    chapter_mapping: List[Dict],
    model_name: str,
    total_pages: int,
    process_log: List[str],
    output_folder: Optional[Path] = None
) -> List[Dict]:
    print("\n[STEP 5] Targeted vision verification of chapter start pages...")

    verify_dir = output_folder / "_temp_verify" if output_folder else Path("verify_titles")
    if verify_dir.exists():
        safe_rmtree(verify_dir)
    verify_dir.mkdir(parents=True, exist_ok=True)

    model = genai.GenerativeModel(model_name)
    fallback_model = genai.GenerativeModel("gemini-2.5-pro")
    print(f"    Using model: {model_name} (fallback: gemini-2.5-pro)")
    print(f"    Verifying {len(chapter_mapping)} chapters...")

    verified = []
    chunk_size = 5

    for batch_start in range(0, len(chapter_mapping), chunk_size):
        batch_end = min(batch_start + chunk_size, len(chapter_mapping))
        tasks = []

        for ch_idx in range(batch_start, batch_end):
            ch = chapter_mapping[ch_idx]
            tasks.append(_verify_single_chapter_with_retry(model, fallback_model, ch, pdf_path, verify_dir, total_pages))

        results = await asyncio.gather(*tasks)
        verified.extend(results)

        if batch_end < len(chapter_mapping):
            await asyncio.sleep(1)

    # Log results
    for ch in verified:
        status = ch.get("status", "")
        title = ch.get("chapter_title", "Unknown")
        pdf_idx = ch.get("pdf_page_index_0_based", "?")
        process_log.append(f"  - '{title}' -> PDF {pdf_idx} ({status})")

    if verify_dir.exists():
        safe_rmtree(verify_dir)

    return verified

async def _verify_single_chapter_with_retry(
    model, fallback_model, ch: Dict, pdf_path: str, verify_dir: Path, total_pages: int
) -> Dict:
    title = ch["chapter_title"]
    predicted = ch["predicted_pdf_index"]

    # Attempt 1: radius 6
    radius_1 = 6
    start = max(0, predicted - radius_1)
    end = min(total_pages - 1, predicted + radius_1)
    page_range_1 = list(range(start, end + 1))
    
    render_map = await asyncio.to_thread(render_specific_pages, pdf_path, page_range_1, verify_dir)
    
    available_pages = []
    images = []
    for idx in page_range_1:
        if idx in render_map:
            available_pages.append(idx)
            images.append(Image.open(render_map[idx]))

    if not images:
        ch["pdf_page_index_0_based"] = predicted
        ch["pdf_display_page_number"] = predicted + 1
        ch["status"] = ch.get("status", "") + " (No images)"
        return ch

    prompt = (
        f"I have {len(images)} page images from a PDF.\n\n"
        f"I need you to find the exact starting page of the chapter titled \"{title}\".\n\n"
        f"INSTRUCTIONS:\n"
        f"1. Look for the page where \"{title}\" appears prominently as the START of a new chapter (in large or bold font).\n"
        f"2. WHEN YOU FIND IT, look for the big RED BOX we stamped onto that same image (located around the middle/bottom of the page).\n"
        f"3. Read the number inside that red box (it says 'PDF PAGE: [number]').\n"
        f"4. Return ONLY that number as an integer in the 'page_index' JSON field.\n\n"
        f"If you absolutely cannot find this chapter heading on ANY of these pages, return 0."
    )

    try:
        response = await asyncio.to_thread(
            model.generate_content,
            [prompt] + images,
            generation_config={"response_mime_type": "application/json", "response_schema": VerifyResult},
            request_options={"timeout": get_model_timeout(model.model_name)}
        )
        data = json.loads(response.text)
        found_idx = data.get("page_index", 0)
        
        if found_idx == -1: found_idx = 0

        # Attempt 2: radius 9 with gemini-2.5-pro
        if found_idx == 0 or found_idx not in available_pages:
            print(f"    [RETRY] '{title}' not found with radius {radius_1}. Retrying with radius 9 and 2.5 model...")
            radius_2 = 9
            start = max(0, predicted - radius_2)
            end = min(total_pages - 1, predicted + radius_2)
            page_range_2 = list(range(start, end + 1))
            
            render_map_2 = await asyncio.to_thread(render_specific_pages, pdf_path, page_range_2, verify_dir)
            available_pages_2 = []
            images_2 = []
            for idx in page_range_2:
                if idx in render_map_2:
                    available_pages_2.append(idx)
                    images_2.append(Image.open(render_map_2[idx]))
                    
            prompt_2 = (
                f"I have {len(images_2)} page images from a PDF.\n\n"
                f"I need you to find the exact starting page of the chapter titled \"{title}\".\n\n"
                f"INSTRUCTIONS:\n"
                f"1. Look for the page where \"{title}\" appears prominently as the START of a new chapter (in large or bold font).\n"
                f"2. WHEN YOU FIND IT, look for the big RED BOX we stamped onto that same image (located around the middle/bottom of the page).\n"
                f"3. Read the number inside that red box (it says 'PDF PAGE: [number]').\n"
                f"4. Return ONLY that number as an integer in the 'page_index' JSON field.\n\n"
                f"If you absolutely cannot find this chapter heading on ANY of these pages, return 0."
            )
            
            response_2 = await asyncio.to_thread(
                fallback_model.generate_content,
                [prompt_2] + images_2,
                generation_config={"response_mime_type": "application/json", "response_schema": VerifyResult},
                request_options={"timeout": get_model_timeout("gemini-1.5-pro")} # Fallback is usually pro
            )
            data_2 = json.loads(response_2.text)
            found_idx = data_2.get("page_index", 0)
            if found_idx == -1: found_idx = 0
            available_pages = available_pages_2

        if found_idx == 0 or found_idx not in available_pages:
            ch["pdf_page_index_0_based"] = predicted
            ch["pdf_display_page_number"] = predicted + 1
            ch["status"] = ch.get("status", "") + " + Vision(not found, keeping fit)"
            print(f"    [MISS] '{title}' not found near PDF {predicted}, keeping fit prediction")
        else:
            shift = found_idx - predicted
            ch["pdf_page_index_0_based"] = found_idx
            ch["pdf_display_page_number"] = found_idx + 1
            if shift == 0:
                ch["status"] = "Verified (Fit confirmed by vision)"
                print(f"    [OK] '{title}' confirmed at PDF {found_idx}")
            else:
                ch["status"] = f"Corrected (Fit {predicted} -> Vision {found_idx}, shift {shift:+d})"
                print(f"    [CORRECTED] '{title}' PDF {predicted} -> {found_idx} (shift {shift:+d})")

    except Exception as e:
        ch["pdf_page_index_0_based"] = predicted
        ch["pdf_display_page_number"] = predicted + 1
        ch["status"] = ch.get("status", "") + f" + Vision error: {type(e).__name__}"
        print(f"    [ERROR] '{title}' vision check failed: {e}")

    return ch


# ==========================================
# STEP 1: TOC EXTRACTION
# ==========================================

def validate_toc_quality(toc_data: List[Dict]) -> Tuple[bool, str]:
    """Validate TOC extraction results with conservative heuristics.
    Returns (is_valid, reason) tuple."""
    if not toc_data:
        return False, "No chapters extracted"

    # Check 1: Do we have page numbers at all?
    pages = []
    for entry in toc_data:
        raw = entry.get("printed_page_number")
        if raw is not None:
            try:
                pages.append(int(raw))
            except (ValueError, TypeError):
                pass

    if len(pages) == 0:
        return False, "No page numbers found in any chapter"

    if len(pages) < len(toc_data) * 0.5:
        return False, f"Only {len(pages)}/{len(toc_data)} chapters have page numbers (below 50% threshold)"

    # Check 2: Monotonic (strictly increasing) - pages must go up
    non_increasing = 0
    for i in range(1, len(pages)):
        if pages[i] <= pages[i - 1]:
            non_increasing += 1

    if non_increasing > len(pages) * 0.3:
        return False, f"Page numbers are NOT monotonically increasing ({non_increasing}/{len(pages)-1} violations). Likely hallucinated."

    # Check 3: Duplicate cluster detection (e.g., all chapters mapped to page 8)
    from collections import Counter
    page_counts = Counter(pages)
    most_common_page, most_common_count = page_counts.most_common(1)[0]
    if most_common_count >= 3 and most_common_count > len(pages) * 0.3:
        return False, f"Page {most_common_page} appears {most_common_count} times out of {len(pages)} chapters. Likely hallucinated."

    # Check 4: Unreasonably small spread (all pages within a tiny range)
    page_spread = max(pages) - min(pages)
    if len(pages) >= 5 and page_spread < 20:
        return False, f"Page spread is only {page_spread} across {len(pages)} chapters. Chapters should span a wider range."

    # Check 5: First chapter page number suspiciously high
    if pages[0] > 100:
        return False, f"First chapter starts at page {pages[0]}, which is suspiciously high for a first chapter."

    return True, "All checks passed"


async def get_toc_initial(pdf_path: str, model_name="gemini-3-flash-preview", output_folder: Optional[Path] = None) -> List[Dict]:
    print("\n[STEP 1] Extracting TOC from first 20 pages...")
    temp_dir = output_folder / "_temp_toc_scan" if output_folder else Path("temp_toc_scan")
    render_map = render_specific_pages(pdf_path, list(range(20)), temp_dir)
    image_paths = sorted(render_map.values())
    if not image_paths:
        return []

    prompt = """
    Extract the main Table of Contents from these images.
    Ignore sub-chapters or front matter (Preface, Acknowledgments, etc).
    Only extract main chapters (e.g. "Chapter 1", "Chapter 2", etc.).
    
    CRITICAL: For each chapter, you MUST extract the printed_page_number.
    The page number is usually found at the END of each TOC line, after dots or spaces.
    Do NOT confuse the page number of the TOC page itself with the chapter page numbers.
    For example, if the TOC is on page 8, do NOT put "8" as the page number for every chapter.
    Each chapter has its OWN unique page number listed in the TOC.
    """
    contents = [prompt] + [Image.open(p) for p in image_paths]


    # Attempt 1: Primary model (gemini-3-flash-preview)
    print(f"    [TOC] Attempt 1 with {model_name}...")
    try:
        model = genai.GenerativeModel(model_name)
        response = await asyncio.to_thread(
            model.generate_content, contents,
            generation_config={"response_mime_type": "application/json", "response_schema": list[TocItem]},
            request_options={"timeout": get_model_timeout(model_name)}
        )
        data = json.loads(response.text)
        
        # Save raw JSON for diagnostics
        if output_folder:
            with open(output_folder / "raw_toc_attempt_1.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        is_valid, reason = validate_toc_quality(data)
        if is_valid:
            print(f"    [TOC] Attempt 1 PASSED validation: {reason}")
            return data
        else:
            print(f"    [TOC] Attempt 1 FAILED validation: {reason}")
    except Exception as e:
        print(f"    [TOC] Attempt 1 ERROR: {e}")
        data = []

    # Attempt 2: Fallback to gemini-2.5-pro (heavier, more accurate)
    fallback_name = "gemini-2.5-pro"
    print(f"    [TOC] Attempt 2 with {fallback_name} (heavy fallback)...")
    try:
        fallback_model = genai.GenerativeModel(fallback_name)
        response = await asyncio.to_thread(
            fallback_model.generate_content, contents,
            generation_config={"response_mime_type": "application/json", "response_schema": list[TocItem]},
            request_options={"timeout": get_model_timeout(fallback_name)}
        )
        data_2 = json.loads(response.text)

        # Save raw JSON for diagnostics
        if output_folder:
            with open(output_folder / "raw_toc_attempt_2.json", "w", encoding="utf-8") as f:
                json.dump(data_2, f, indent=2)

        is_valid_2, reason_2 = validate_toc_quality(data_2)
        if is_valid_2:
            print(f"    [TOC] Attempt 2 PASSED validation: {reason_2}")
            return data_2
        else:
            print(f"    [TOC] Attempt 2 also FAILED validation: {reason_2}")
            # If both fail validation, do not return corrupted data that will break subsequent math.
            return []
    except Exception as e:
        print(f"    [TOC] Attempt 2 ERROR: {e}")
        return []


# ==========================================
# CONTENT EXTRACTION (Markdown)
# ==========================================
def extract_chapter_content(pdf_path: str, chapter_map: List[Dict], total_pages: int) -> List[Dict]:
    print("\n[EXTRACT] Extracting chapter content as Markdown...")
    valid_chapters = [c for c in chapter_map if "pdf_page_index_0_based" in c]
    valid_chapters.sort(key=lambda x: x["pdf_page_index_0_based"])
    updated = []

    for i, chapter in enumerate(valid_chapters):
        start_page = chapter["pdf_page_index_0_based"]
        if i < len(valid_chapters) - 1:
            end_page = max(start_page, valid_chapters[i + 1]["pdf_page_index_0_based"] - 1)
        else:
            end_page = total_pages - 1

        print(f"    '{chapter.get('chapter_title', '?')}' (pages {start_page}..{end_page})...")
        try:
            pages = list(range(start_page, end_page + 1))
            if pages:
                chapter["full_text_markdown"] = pymupdf4llm.to_markdown(pdf_path, pages=pages)
            else:
                chapter["full_text_markdown"] = ""
            chapter["page_range_extracted"] = [start_page, end_page]
        except Exception as e:
            print(f"    [ERROR] {e}")
            chapter["full_text_markdown"] = ""
        updated.append(chapter)

    return updated


# ==========================================
# SUMMARIZATION
# ==========================================
def generate_summaries_robust(json_data_path: str, output_folder: Path, process_log: List[str], model_name: str, ui=None):
    with open(json_data_path, "r", encoding="utf-8") as f:
        chapters = json.load(f)

    if ui: ui.update(f"[SUMMARIZE] Starting with {model_name}...", 80)
    process_log.append(f"[SUMMARIZE] Starting with {model_name}")
    print(f"\n[SUMMARIZE] Generating summaries for {len(chapters)} chapters...")
    
    # Using new SDK for summarization to enable LOW thinking mode
    client = new_genai.Client(api_key=API_KEY)
    
    new_safety_settings = [
        new_types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
        new_types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
        new_types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
        new_types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
    ]
    summarized_count = 0
    total = len(chapters)

    for i, ch in enumerate(chapters):
        title = ch.get("chapter_title", "Unknown")
        text = ch.get("full_text_markdown", "")
        
        if not text:
            print(f"  [SKIP] '{title}' has no text.")
            continue

        if "short_summary" in ch and len(ch["short_summary"]) > 5:
            print(f"  [SKIP] '{title}' already summarized.")
            continue

        print(f"  [API] Summarizing '{title[:50]}'... ", end="", flush=True)
        if ui: ui.update(f"Summarizing {i+1}/{total}: {title[:40]}...", 80 + (i/total * 20))
        
        # Word count math
        # simple split by whitespace for rough word count
        word_count = len(text.split())
        long_target = word_count * 0.10
        long_min = int(long_target * 0.80)
        long_max = int(long_target * 1.20)
        
        short_target = long_target * 0.10
        short_min = int(short_target * 0.80)
        short_max = int(short_target * 1.20)
        
        # Ensure minimums don't drop to 0
        long_min = max(50, long_min)
        long_max = max(100, long_max)
        short_min = max(10, short_min)
        short_max = max(20, short_max)
        
        base_prompt_text = f"""Read the provided chapter and produce 2 summaries based strictly on the requirements below.

CRITICAL INSTRUCTIONS
Be hyper concise and straightforward. Avoid vagueness and fluff. Use concrete details and exact terms. Retain the most important citations in the Long Summary. Do not include citations in the Short Summary.

LENGTH REQUIREMENTS
1. The original chapter text provided below is approximately {word_count} words.
2. The Long Summary section must be between {long_min} and {long_max} words.
3. The Short Summary section must be between {short_min} and {short_max} words.

FORMATTING REQUIREMENTS
You must use Markdown headings exactly as specified below for algorithmic extraction. Do not output any text outside of these headings. 

# Long Summary
Divide this section into as many themes as necessary for the context. Always include important citations here.

## Introduction
Provide a hyper concise introduction to the core premise.

[STRUCTURE GUIDELINE: Use as many ## Theme and ### Argument headers as the content requires. Do NOT stick to just two themes; expand as necessary to cover all logical points.]
## Theme Name
State the theme clearly.
### Argument Name
Provide the specific logic and evidence.
[CONTINUE WITH MORE THEMES AND ARGUMENTS AS NEEDED]

## Extras
Include important concise information not contained in the themes above.

# Short Summary
Write a highly condensed summary. Do not divide this section by themes. Do not include citations.

CHAPTER: {title}
TEXT: {text[:40000]}"""

        max_wordcount_retries = 3
        current_prompt = base_prompt_text
        
        for w_attempt in range(max_wordcount_retries):
            try:
                timeout_val = get_model_timeout(model_name)
                max_api_retries = 3
                response = None
                
                for attempt in range(max_api_retries):
                    try:
                        # Increment timeout slightly on each retry
                        current_timeout = timeout_val + (attempt * 60)
                        
                        # Only apply thinking_config to "pro" models
                        config_kwargs = {
                            "safety_settings": new_safety_settings
                        }
                        if "pro" in model_name.lower():
                            config_kwargs["thinking_config"] = new_types.ThinkingConfig(thinking_level="LOW")
                        
                        config = new_types.GenerateContentConfig(**config_kwargs)
                        
                        # We use the new SDK for the thinking mode
                        response = client.models.generate_content(
                            model=model_name,
                            contents=current_prompt,
                            config=config
                        )
                        break # Success! Break the retry loop
                    except Exception as e:
                        error_str = str(e)
                        error_name = type(e).__name__
                        # RemoteProtocolError is a common transient network hiccup in the new SDK
                        is_transient = "Deadline" in error_name or "Timeout" in error_name or "RemoteProtocolError" in error_name or "503" in error_str or "500" in error_str
                        
                        if attempt < max_api_retries - 1 and is_transient:
                            print(f" [API RETRY {attempt+1} - {error_name}]", end="", flush=True)
                            if ui: ui.update(f"  [API RETRY] {title[:40]} ({error_name})...")
                            time.sleep(10 + (attempt * 10)) # Increased delay for network errors
                        else:
                            raise e # Out of retries or non-transient error

                log_api_call(f"Summarize '{title[:50]}' (Attempt {w_attempt+1})", model_name, current_prompt, response, None, process_log)
                raw_text = response.text

                short_sum = ""
                long_sum = ""
                
                # Robust parsing for # Long Summary and # Short Summary
                long_match = re.search(r"#\s*Long Summary\s*(.*?)(?=#\s*Short Summary|$)", raw_text, re.DOTALL | re.IGNORECASE)
                if long_match:
                    long_sum = long_match.group(1).strip()
                
                short_match = re.search(r"#\s*Short Summary\s*(.*)", raw_text, re.DOTALL | re.IGNORECASE)
                if short_match:
                    short_sum = short_match.group(1).strip()
                
                # Fallback to legacy labels if new ones not found
                if not long_sum and not short_sum:
                    short_match_legacy = re.search(r"SHORT:\s*(.*?)LONG:", raw_text, re.DOTALL | re.IGNORECASE)
                    if short_match_legacy:
                        short_sum = short_match_legacy.group(1).strip()
                    long_match_legacy = re.search(r"LONG:\s*(.*)", raw_text, re.DOTALL | re.IGNORECASE)
                    if long_match_legacy:
                        long_sum = long_match_legacy.group(1).strip()
                        
                # Check word counts
                long_words = len(long_sum.split()) if long_sum else 0
                short_words = len(short_sum.split()) if short_sum else 0

                # Thresholds: Retry if < 20% below target or > 30% above target
                long_off = (long_words < long_target * 0.8) or (long_words > long_target * 1.3)
                short_off = (short_words < short_target * 0.8) or (short_words > short_target * 1.3)
                
                if (long_off or short_off) and w_attempt < max_wordcount_retries - 1:
                    print(f" [LENGTH RETRY {w_attempt+1}] (Long: {long_words}/{long_min}-{long_max}, Short: {short_words}/{short_min}-{short_max})", end="", flush=True)
                    if ui: ui.update(f"  [LENGTH RETRY] {title[:40]} (L:{long_words}, S:{short_words})")
                    
                    # Append feedback to prompt
                    feedback = f"\n\n--- FEEDBACK ON YOUR PREVIOUS ATTEMPT ---\n"
                    feedback += f"We asked you previously for a Long Summary of {long_min}-{long_max} words and a Short Summary of {short_min}-{short_max} words.\n"
                    feedback += f"You returned a Long Summary of {long_words} words and a Short Summary of {short_words} words.\n"
                    if long_off:
                        feedback += f"The Long Summary was off. It must be strictly between {long_min} and {long_max} words.\n"
                    if short_off:
                        feedback += f"The Short Summary was off. It must be strictly between {short_min} and {short_max} words.\n"
                    feedback += f"\nPlease fix it. Do not include this feedback in your output. Here is your previous output that needs fixing:\n\n{raw_text}"
                    
                    current_prompt = base_prompt_text + feedback
                    continue # Retry
                else:
                    # Accepted
                    chapters[i]["short_summary"] = short_sum if short_sum else "Summary unavailable."
                    chapters[i]["long_summary"] = long_sum if long_sum else ""
                    print(" [OK]")
                    if w_attempt > 0:
                        process_log.append(f"  - Length accepted after {w_attempt} retries: Long {long_words}w, Short {short_words}w")
                    summarized_count += 1
                    break

            except Exception as error:
                log_api_call(f"Summarize '{title[:50]}'", model_name, current_prompt, None, error, process_log)
                print(f" [FAILED]: {type(error).__name__}")
                if ui: ui.update(f"  [FAILED] {title[:40]}: {type(error).__name__}")
                chapters[i]["short_summary"] = f"Error: {type(error).__name__}"
                chapters[i]["long_summary"] = ""
                break # break wordcount retry loop on hard error

        checkpoint_path = output_folder / "complete_data.json"
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(chapters, f, indent=2, ensure_ascii=False)
        time.sleep(1)

    process_log.append(f"  - Summarized: {summarized_count} chapters")
    process_log.append(f"[COMPLETE] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"\n[DONE] Generating final output files...")
    if ui: ui.update("Generating final output files...", 100)
    generate_output_files(output_folder, chapters, process_log, summaries_generated=True)
    print(f"[DONE] All files saved to: {output_folder.name}")
    if ui: ui.update(f"[FINISH] All files saved to: {output_folder.name}", 100)


# ==========================================
# MAIN PIPELINE
# ==========================================
async def process_book_final_flow(pdf_path: str, model_config: Dict[str, str], ui=None):
    global API_USAGE_STATS
    API_USAGE_STATS = []
    process_log = []
    
    start_msg = f"Processing PDF: {Path(pdf_path).name}"
    if ui: ui.update(start_msg, 5)
    
    process_log.append("=" * 80)
    process_log.append(start_msg)
    process_log.append(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    process_log.append(f"TOC Model: {model_config['toc_model']}")
    process_log.append(f"Visual Model: {model_config['visual_model']}")
    process_log.append(f"Summary Model: {model_config['summary_model']}")
    process_log.append("=" * 80)
    process_log.append("")

    try:
        output_folder = create_output_folder(pdf_path)
        process_log.append(f"Output folder: {output_folder.name}")
        process_log.append("")

        # Get total pages early (needed for Step 6 if cache is hit)
        total_pages = 0
        if fitz:
            with fitz.open(pdf_path) as doc:
                total_pages = len(doc)
        elif pdfplumber:
            with pdfplumber.open(pdf_path) as doc:
                total_pages = len(doc.pages)
        
        if total_pages == 0:
            raise ValueError("Could not determine PDF page count.")

        # Resume capability
        temp_json_path = output_folder / "book_data_with_text.json"
        run_summaries = model_config.get("run_summaries", True)
        
        if temp_json_path.exists() and run_summaries:
            print(f"\n[RESUME] Found existing data, resuming from summarization...")
            process_log.append("[RESUME] Resuming from summarization...")
            if ui: ui.update("[RESUME] Found existing data, resuming from summarization...", 10)
            generate_summaries_robust(str(temp_json_path), output_folder, process_log, model_config["summary_model"], ui=ui)
            return
        elif temp_json_path.exists() and not run_summaries:
            print(f"\n[RESUME] Found existing data, but summaries are disabled. Generating extracted outputs directly.")
            with open(temp_json_path, "r", encoding="utf-8") as f:
                chapters_with_text = json.load(f)
            generate_output_files(output_folder, chapters_with_text, process_log, summaries_generated=False)
            return

        # ---- GLOBAL TOC CACHE CHECK ----
        pdf_hash = get_pdf_hash(pdf_path)
        global_cache = load_toc_cache()
        cached_data = global_cache.get(pdf_hash)
        
        verified_mapping = None
        if cached_data:
            print(f"\n[CACHE HIT] Found previous analysis from {cached_data['cached_at']}")
            process_log.append(f"[CACHE] Using cached TOC analysis from {cached_data['cached_at']}")
            if ui: ui.update("[CACHE] Using previously verified Table of Contents...", 15)
            verified_mapping = cached_data["mapping"]
            # We skip steps 1-5 and go straight to step 6
        
        if not verified_mapping:
            # ---- STEP 1: TOC Extraction (vision) ----
            if ui: ui.update("[STEP 1] Extracting Table of Contents...", 10)
            toc_entries = await get_toc_initial(pdf_path, model_config["toc_model"], output_folder=output_folder)
            if not toc_entries:
                error_msg = "[CRITICAL ERROR] TOC Extraction yielded zero chapters. Skipping."
                print(error_msg)
                if ui: ui.update(error_msg)
                process_log.append(error_msg)
                generate_output_files(output_folder, [], process_log)
                return

            # Check for missing page numbers
            pages_found = [e for e in toc_entries if e.get("printed_page_number")]
            if len(pages_found) == 0:
                error_msg = "[CRITICAL ERROR] TOC Extraction found titles but NO page numbers. Skipping."
                print(error_msg)
                if ui: ui.update(error_msg)
                process_log.append(error_msg)
                generate_output_files(output_folder, [], process_log)
                return
            
            process_log.append(f"[STEP 1] TOC Extraction Complete")
            process_log.append(f"  - Chapters found: {len(toc_entries)}")
            for entry in toc_entries[:5]:
                process_log.append(f"    - {entry.get('chapter_title', '?')} p.{entry.get('printed_page_number', '?')}")
            if len(toc_entries) > 5:
                process_log.append(f"    ... and {len(toc_entries)-5} more")
            process_log.append("")

            # ---- STEP 2: Dense page number extraction from PDF metadata/text ----
            if ui: ui.update("[STEP 2] Performing dense page analysis...", 20)
            dense_data = extract_dense_page_numbers(pdf_path)
            # Use the early calculated total_pages if possible, otherwise from dense_data
            total_pages = total_pages or dense_data["total_pages"]
            ratio = dense_data["extractable_ratio"]
            process_log.append(f"[STEP 2] Dense Page Number Extraction")
            process_log.append(f"  - Total pages: {total_pages}")
            process_log.append(f"  - Extractable ratio: {ratio:.0%}")
            process_log.append("")

            if ratio < 0.01:
                error_msg = f"[CRITICAL ERROR] PDF text is not extractable (ratio {ratio:.1%}). Scanned PDF without OCR? Skipping."
                print(error_msg)
                if ui: ui.update(error_msg)
                process_log.append(error_msg)
                generate_output_files(output_folder, [], process_log)
                return

            # ---- STEP 3: Robust Linear Fit ----
            if ui: ui.update("[STEP 3] Calculating page offset fit...", 30)
            fit = build_robust_page_mapping(dense_data)
            process_log.append(f"[STEP 3] Robust Linear Fit")
            process_log.append(f"  - Slope: {fit['slope']:.4f}")
            process_log.append(f"  - Intercept: {fit['intercept']:.2f}")
            process_log.append(f"  - Layout mode: {fit['mode'].upper()}")
            process_log.append(f"  - Inliers: {fit['inlier_count']}")
            process_log.append("")

            # ---- STEP 4: Map TOC to PDF pages via the fit ----
            if ui: ui.update("[STEP 4] Mapping chapters to PDF pages...", 40)
            chapter_mapping = map_toc_to_pdf(toc_entries, fit, total_pages)
            process_log.append(f"[STEP 4] TOC -> PDF Mapping via Linear Fit")
            process_log.append(f"  - Chapters mapped: {len(chapter_mapping)}")
            process_log.append("")

            # ---- STEP 5: Targeted vision verification (+/- 3 pages per chapter) ----
            if ui: ui.update("[STEP 5] Verifying chapter starts with vision...", 50)
            verified_mapping = await verify_chapter_starts(
                pdf_path, chapter_mapping, model_config["visual_model"],
                total_pages, process_log, output_folder=output_folder
            )
            process_log.append(f"[STEP 5] Targeted Vision Verification Complete")
            process_log.append("")
            
            # SAVE TO GLOBAL CACHE
            save_toc_to_cache(pdf_hash, verified_mapping)
            print(f"[CACHE] Saved verified TOC mapping to global cache.")

        # ---- STEP 6: Content extraction ----
        if ui: ui.update("[STEP 6] Extracting chapter body text...", 70)
        chapters_with_text = extract_chapter_content(pdf_path, verified_mapping, total_pages)
        process_log.append(f"[STEP 6] Content Extraction Complete")
        process_log.append(f"  - Chapters with text: {len(chapters_with_text)}")
        process_log.append("")

        # Save intermediate data
        with open(temp_json_path, "w", encoding="utf-8") as f:
            json.dump(chapters_with_text, f, indent=2, ensure_ascii=False)
        print(f"[SAVE] Intermediate data saved to '{temp_json_path}'")

        # Cleanup temp dirs
        for d in [output_folder / "_temp_toc_scan"]:
            if d.exists():
                safe_rmtree(d)

        # ---- STEP 7: Summarize or Finish ----
        if run_summaries:
            if ui: ui.update("[STEP 7] Generating hyper-concise summaries...", 85)
            generate_summaries_robust(str(temp_json_path), output_folder, process_log, model_config["summary_model"], ui=ui)
        else:
            if ui: ui.update("[FINISH] Summarization skipped, saving raw extractions...", 90)
            print("\n[SKIP] Summarization disabled by user. Generating outputs...")
            process_log.append("[SKIP] Summarization disabled by user.")
            generate_output_files(output_folder, chapters_with_text, process_log, summaries_generated=False)
            if ui: ui.update(f"[FINISH] Output saved to: {output_folder.name}", 100)
    except Exception as e:
        import traceback
        error_msg = f"[CRITICAL EXCEPTION] Processing halted due to an unexpected error: {e}"
        print(f"\n{error_msg}")
        if ui: ui.update(error_msg)
        traceback.print_exc()
        process_log.append(error_msg)
        process_log.append(traceback.format_exc())
        if 'output_folder' in locals():
            generate_output_files(output_folder, [], process_log)


# ==========================================
# ENTRY POINT
# ==========================================
async def main():
    print("\n" + "=" * 60)
    print("  BOOK CHAPTER SUMMARIZER v2")
    print("  (Dense fit + targeted vision)")
    print("=" * 60)

    # BATCH MODE: accept PDF path as command-line argument
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
        if not os.path.exists(pdf_file):
            print(f"[ERROR] File not found: {pdf_file}")
            return
        print(f"\n[BATCH] Using PDF from command line: {Path(pdf_file).name}")

        # Use default models (can be overridden via optional 2nd arg: toc,vision,summary)
        toc_model = "gemini-3-flash-preview"
        visual_model = "gemini-3-flash-preview"
        summary_model = "gemini-3-flash-preview"

        if len(sys.argv) > 2:
            parts = sys.argv[2].split(",")
            if len(parts) >= 1 and parts[0]: toc_model = parts[0]
            if len(parts) >= 2 and parts[1]: visual_model = parts[1]
            if len(parts) >= 3 and parts[2]: summary_model = parts[2]

        model_config = {
            "toc_model": toc_model,
            "visual_model": visual_model,
            "summary_model": summary_model,
            "run_summaries": True,
        }
    else:
        # INTERACTIVE MODE: original GUI behavior
        pdf_file = select_pdf_file()
        if not pdf_file or not os.path.exists(pdf_file):
            print("[ERROR] No valid file selected. Exiting.")
            return

        model_config = create_configuration_dialog(pdf_file)
        if not model_config:
            print("[ERROR] Configuration cancelled. Exiting.")
            return

    print(f"\n[START] Processing with:")
    print(f"   TOC: {AVAILABLE_MODELS.get(model_config['toc_model'], model_config['toc_model'])}")
    print(f"   Vision: {AVAILABLE_MODELS.get(model_config['visual_model'], model_config['visual_model'])}")
    print(f"   Summary: {AVAILABLE_MODELS.get(model_config['summary_model'], model_config['summary_model']) if model_config.get('run_summaries') else 'SKIPPED'}")

    # Run the pipeline synchronously relative to the main block, no GUI window
    await process_book_final_flow(pdf_file, model_config, ui=None)

if __name__ == "__main__":
    asyncio.run(main())