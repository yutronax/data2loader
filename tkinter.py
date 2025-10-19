"""
AutoDataPrep GUI Module
Tkinter tabanlı grafik kullanıcı arayüzü
"""

import os
import threading
import logging
from typing import Optional
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

from PIL import Image, ImageTk

from core import AutoDataPrep, PrepResult, DEFAULT_BATCH_SIZE, DEFAULT_NUM_WORKERS, DEFAULT_TRAIN_RATIO

logger = logging.getLogger(__name__)


class AutoDataPrepGUI:
    """Tkinter tabanlı grafik arayüz"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("AutoDataPrep - Otomatik Veri Hazırlama Sistemi")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        self.prep_result: Optional[PrepResult] = None
        self.current_images = []
        
        self._setup_ui()
        
        # Başlangıçta ZIP seçim penceresini aç
        self.root.after(100, self._initial_file_select)
    
    def _setup_ui(self):
        """UI bileşenlerini oluştur"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        # Başlık
        title_label = ttk.Label(
            main_frame, 
            text="AutoDataPrep - Otomatik Veri Hazırlama",
            font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, pady=10, sticky=tk.W)
        
        # Input seçimi
        input_frame = ttk.LabelFrame(main_frame, text="Veri Kaynağı", padding="10")
        input_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        input_frame.columnconfigure(1, weight=1)
        
        ttk.Label(input_frame, text="Dataset:").grid(row=0, column=0, sticky=tk.W)
        self.input_path_var = tk.StringVar()
        self.input_entry = ttk.Entry(
            input_frame, textvariable=self.input_path_var, width=50
        )
        self.input_entry.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))
        
        ttk.Button(
            input_frame, text="ZIP Seç", command=self._browse_zip
        ).grid(row=0, column=2, padx=2)
        ttk.Button(
            input_frame, text="Klasör Seç", command=self._browse_folder
        ).grid(row=0, column=3, padx=2)
        
        # Parametreler
        params_frame = ttk.LabelFrame(main_frame, text="Parametreler", padding="10")
        params_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(params_frame, text="Batch Size:").grid(row=0, column=0, sticky=tk.W)
        self.batch_size_var = tk.IntVar(value=DEFAULT_BATCH_SIZE)
        ttk.Spinbox(
            params_frame, from_=1, to=128, textvariable=self.batch_size_var, width=10
        ).grid(row=0, column=1, padx=5, sticky=tk.W)
        
        ttk.Label(params_frame, text="Train Ratio:").grid(row=0, column=2, padx=20, sticky=tk.W)
        self.train_ratio_var = tk.DoubleVar(value=DEFAULT_TRAIN_RATIO)
        ttk.Spinbox(
            params_frame, from_=0.5, to=0.95, increment=0.05, 
            textvariable=self.train_ratio_var, width=10
        ).grid(row=0, column=3, padx=5, sticky=tk.W)
        
        ttk.Label(params_frame, text="Num Workers:").grid(row=0, column=4, padx=20, sticky=tk.W)
        self.num_workers_var = tk.IntVar(value=DEFAULT_NUM_WORKERS)
        ttk.Spinbox(
            params_frame, from_=0, to=8, textvariable=self.num_workers_var, width=10
        ).grid(row=0, column=5, padx=5, sticky=tk.W)
        
        ttk.Label(params_frame, text="Seed:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.seed_var = tk.IntVar(value=42)
        ttk.Spinbox(
            params_frame, from_=0, to=9999, textvariable=self.seed_var, width=10
        ).grid(row=1, column=1, padx=5, sticky=tk.W)
        
        # İşlem butonları
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, pady=10)
        
        self.process_btn = ttk.Button(
            button_frame, text="Veriyi Hazırla", command=self._process_data
        )
        self.process_btn.grid(row=0, column=0, padx=5)
        
        self.visualize_btn = ttk.Button(
            button_frame, text="Örnekleri Görüntüle", 
            command=self._visualize_samples, state=tk.DISABLED
        )
        self.visualize_btn.grid(row=0, column=1, padx=5)
        
        ttk.Button(
            button_frame, text="Çıkış", command=self.root.quit
        ).grid(row=0, column=2, padx=5)
        
        # Log alanı
        log_frame = ttk.LabelFrame(main_frame, text="İşlem Çıktısı", padding="5")
        log_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame, width=80, height=20, wrap=tk.WORD
        )
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=5)
    
    def _browse_zip(self):
        """ZIP dosyası seç"""
        filename = filedialog.askopenfilename(
            title="ZIP Dosyası Seçin",
            filetypes=[("ZIP files", "*.zip"), ("All files", "*.*")]
        )
        if filename:
            self.input_path_var.set(filename)
    
    def _browse_folder(self):
        """Klasör seç"""
        dirname = filedialog.askdirectory(title="Dataset Klasörü Seçin")
        if dirname:
            self.input_path_var.set(dirname)
    
    def _initial_file_select(self):
        """Başlangıçta dosya seçim penceresini göster"""
        result = messagebox.askyesnocancel(
            "Dataset Seçimi",
            "Dataset yüklemek ister misiniz?\n\n"
            "Evet: ZIP dosyası seç\n"
            "Hayır: Klasör seç\n"
            "İptal: Daha sonra seç"
        )
        
        if result is True:
            self._browse_zip()
        elif result is False:
            self._browse_folder()
    
    def _log(self, message: str):
        """Log mesajı ekle"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def _process_data(self):
        """Veriyi işle"""
        input_path = self.input_path_var.get()
        if not input_path:
            messagebox.showerror("Hata", "Lütfen bir dataset seçin!")
            return
        
        if not os.path.exists(input_path):
            messagebox.showerror("Hata", f"Dosya bulunamadı: {input_path}")
            return
        
        self.process_btn.config(state=tk.DISABLED)
        self.visualize_btn.config(state=tk.DISABLED)
        self.log_text.delete(1.0, tk.END)
        self.progress.start()
        
        def process_thread():
            try:
                self._log("İşlem başlatılıyor...")
                self._log(f"Girdi: {input_path}")
                self._log(f"Parametreler: batch_size={self.batch_size_var.get()}, "
                         f"train_ratio={self.train_ratio_var.get()}, "
                         f"seed={self.seed_var.get()}")
                self._log("-" * 80)
                
                prep = AutoDataPrep(
                    seed=self.seed_var.get(),
                    train_ratio=self.train_ratio_var.get()
                )
                
                result = prep.run(
                    input_path=input_path,
                    batch_size=self.batch_size_var.get(),
                    num_workers=self.num_workers_var.get()
                )
                
                self.prep_result = result
                
                self._log("\n" + "=" * 80)
                self._log("✅ İŞLEM BAŞARIYLA TAMAMLANDI!")
                self._log("=" * 80)
                self._log(result.summary())
                
                if "classification_counts" in result.dataset_info:
                    self._log("\n📊 Sınıf Dağılımı:")
                    for cls, counts in result.dataset_info["classification_counts"].items():
                        self._log(f"  • {cls}: {counts['train']} train, "
                                 f"{counts['test']} test (toplam: {counts['total']})")
                
                if "segmentation_counts" in result.dataset_info:
                    counts = result.dataset_info["segmentation_counts"]
                    self._log(f"\n📊 Segmentation İstatistikleri:")
                    self._log(f"  • Toplam çift: {counts['total']}")
                    self._log(f"  • Train: {counts['train']}")
                    self._log(f"  • Test: {counts['test']}")
                
                if "detection_counts" in result.dataset_info:
                    counts = result.dataset_info["detection_counts"]
                    self._log(f"\n📊 Detection İstatistikleri:")
                    self._log(f"  • Toplam çift: {counts['total']}")
                    self._log(f"  • Train: {counts['train']}")
                    self._log(f"  • Test: {counts['test']}")
                
                self._log("\n💾 Veriler şu dizine kaydedildi:")
                self._log(f"  {result.split_path}")
                
                self.root.after(0, lambda: self.visualize_btn.config(state=tk.NORMAL))
                
                messagebox.showinfo(
                    "Başarılı", 
                    f"Veri hazırlama tamamlandı!\n\n"
                    f"Görev: {result.task_type}\n"
                    f"Sınıf sayısı: {result.num_classes}\n"
                    f"Train batches: {len(result.train_loader)}\n"
                    f"Test batches: {len(result.test_loader)}"
                )
                
            except Exception as e:
                logger.exception("İşlem hatası")
                self._log(f"\n❌ HATA: {str(e)}")
                messagebox.showerror("Hata", f"İşlem başarısız:\n{str(e)}")
            
            finally:
                self.root.after(0, self.progress.stop)
                self.root.after(0, lambda: self.process_btn.config(state=tk.NORMAL))
        
        thread = threading.Thread(target=process_thread, daemon=True)
        thread.start()
    
    def _visualize_samples(self):
        """Örnek görüntüleri göster"""
        if not self.prep_result:
            messagebox.showwarning("Uyarı", "Önce veriyi hazırlayın!")
            return
        
        vis_window = tk.Toplevel(self.root)
        vis_window.title("Örnek Görüntüler")
        vis_window.geometry("800x600")
        
        frame = ttk.Frame(vis_window, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(
            frame, 
            text=f"Örnek Görüntüler ({self.prep_result.task_type})",
            font=("Arial", 14, "bold")
        ).pack(pady=10)
        
        canvas_frame = ttk.Frame(frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(canvas_frame, bg="white")
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        try:
            self.current_images = []
            images, labels = next(iter(self.prep_result.train_loader))
            
            cols = 4
            rows = min(2, (len(images) + cols - 1) // cols)
            
            for idx in range(min(8, len(images))):
                row = idx // cols
                col = idx % cols
                
                img_tensor = images[idx]
                img_np = img_tensor.permute(1, 2, 0).numpy()
                img_np = (img_np * 255).astype('uint8')
                pil_img = Image.fromarray(img_np)
                pil_img.thumbnail((180, 180))
                
                photo = ImageTk.PhotoImage(pil_img)
                self.current_images.append(photo)
                
                img_frame = ttk.Frame(scrollable_frame, relief="ridge", borderwidth=2)
                img_frame.grid(row=row, column=col, padx=5, pady=5)
                
                img_label = ttk.Label(img_frame, image=photo)
                img_label.pack()
                
                if self.prep_result.task_type == "classification":
                    label_idx = labels[idx].item()
                    if label_idx < len(self.prep_result.class_names):
                        text = self.prep_result.class_names[label_idx]
                    else:
                        text = f"Label: {label_idx}"
                else:
                    text = f"Sample {idx + 1}"
                
                ttk.Label(
                    img_frame, text=text, 
                    font=("Arial", 9)
                ).pack(pady=2)
        
        except Exception as e:
            messagebox.showerror("Hata", f"Görüntüler yüklenemedi:\n{str(e)}")
            vis_window.destroy()
            return
        
        ttk.Button(
            frame, text="Kapat", command=vis_window.destroy
        ).pack(pady=10)


def launch_gui():
    """GUI'yi başlat"""
    root = tk.Tk()
    
    # Stil ayarları
    style = ttk.Style()
    try:
        style.theme_use('clam')
    except:
        pass
    
    app = AutoDataPrepGUI(root)
    root.mainloop()
