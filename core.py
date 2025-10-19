"""
AutoDataPrep Core Module
Otomatik veri hazırlama ve yükleme sistemi
"""

import os
import zipfile
import uuid
import shutil
import random
import logging
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.data import DataLoader, Dataset
    from torchvision import datasets, transforms

# ============================================================================
# LOGGING AYARLARI
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# SABITLER
# ============================================================================
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 2
DEFAULT_TRAIN_RATIO = 0.8

@dataclass
class PrepResult:
    """Veri hazırlama işleminin sonuç bilgileri"""
    task_type: str
    split_path: str
    num_classes: int
    class_names: List[str]
    dataset_info: Dict[str, Any]
    train_loader: Any
    test_loader: Any
    
    def summary(self) -> str:
        """Sonuçların özet raporu"""
        lines = [
            f"Görev Tipi: {self.task_type}",
            f"Sınıf Sayısı: {self.num_classes}",
            f"Sınıf İsimleri: {', '.join(self.class_names) if self.class_names else 'N/A'}",
            f"Train Batch Sayısı: {len(self.train_loader)}",
            f"Test Batch Sayısı: {len(self.test_loader)}",
            f"Split Dizini: {self.split_path}"
        ]
        return "\n".join(lines)


# ============================================================================
# ANA SINIF: AutoDataPrep
# ============================================================================
class AutoDataPrep:
    """
    Otomatik veri hazırlama ve yükleme sistemi.
    """
    
    def __init__(self, seed: int = 42, train_ratio: float = 0.8):
        self.seed = seed
        self.train_ratio = train_ratio
        self.workdir: Optional[str] = None
        self.dataset_info: Dict[str, Any] = {}
        
        random.seed(self.seed)
        
        logger.info(f"AutoDataPrep başlatıldı (seed={seed}, train_ratio={train_ratio})")
    
    def run(self, input_path: str, batch_size: int = DEFAULT_BATCH_SIZE, 
            num_workers: int = DEFAULT_NUM_WORKERS) -> PrepResult:
        """Ana işlem: Veriyi hazırla ve DataLoader'ları oluştur."""
        logger.info(f"İşlem başlatılıyor: {input_path}")
        
        root = self._materialize_input(input_path)
        root = self._enter_single_folder(root)
        task = self._detect_task(root)
        logger.info(f"Tespit edilen görev: {task}")
        
        split_root = self._standardize_and_split(root, task)
        num_classes, class_names = self._infer_num_classes(split_root, task)
        train_loader, test_loader = self._create_dataloaders(
            split_root, task, batch_size, num_workers
        )
        
        result = PrepResult(
            task_type=task,
            split_path=split_root,
            num_classes=num_classes,
            class_names=class_names,
            dataset_info=self.dataset_info,
            train_loader=train_loader,
            test_loader=test_loader,
        )
        
        logger.info("İşlem başarıyla tamamlandı")
        return result
    
    def _materialize_input(self, input_path: str) -> str:
        """ZIP dosyasını çıkar veya klasör yolunu döndür"""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Girdi bulunamadı: {input_path}")
        
        if zipfile.is_zipfile(input_path):
            session = str(uuid.uuid4())
            out_dir = os.path.join("extracted_data", session)
            os.makedirs(out_dir, exist_ok=True)
            
            logger.info(f"ZIP çıkarılıyor: {input_path} -> {out_dir}")
            with zipfile.ZipFile(input_path, "r") as z:
                z.extractall(out_dir)
            
            self.workdir = out_dir
            self.dataset_info["source"] = {"type": "zip", "path": input_path}
            self.dataset_info["workdir"] = out_dir
            return out_dir
        
        self.dataset_info["source"] = {"type": "folder", "path": input_path}
        self.workdir = input_path
        return input_path
    
    def _enter_single_folder(self, root: str) -> str:
        """Eğer tek bir alt klasör varsa içine gir"""
        items = [os.path.join(root, p) for p in os.listdir(root)]
        only_dirs = [p for p in items if os.path.isdir(p)]
        if len(only_dirs) == 1:
            logger.info(f"Tek klasör tespit edildi, içine giriliyor: {only_dirs[0]}")
            return only_dirs[0]
        return root

    def _has_images(self, path: str) -> bool:
        """Klasörde görüntü dosyası var mı kontrol et"""
        if not os.path.isdir(path):
            return False
        try:
            return any(f.lower().endswith(IMG_EXTS) for f in os.listdir(path))
        except Exception:
            return False
    
    def _detect_task(self, root: str) -> str:
        """Görev tipini otomatik olarak tespit et"""
        subdirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        files = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]
        lower_dirs = [d.lower() for d in subdirs]
        
        self.dataset_info["root_subdirs"] = subdirs
        self.dataset_info["root_files"] = files[:10]
        
        if "boxes" in lower_dirs and "images" in lower_dirs:
            boxes_path = os.path.join(root, "boxes")
            boxes_subdirs = [d for d in os.listdir(boxes_path) if os.path.isdir(os.path.join(boxes_path, d))]
            if len(boxes_subdirs) >= 2:
                self.dataset_info["task"] = "classification"
                self.dataset_info["classification_source"] = "boxes"
                self.dataset_info["class_folders"] = boxes_subdirs
                return "classification"
            else:
                self.dataset_info["task"] = "classification"
                self.dataset_info["classification_source"] = "boxes_flat"
                return "classification"
        
        if "images" in lower_dirs and "labels" in lower_dirs:
            self.dataset_info["task"] = "detection"
            self.dataset_info["detection_format"] = "yolo"
            return "detection"
        
        has_images_in_root = any(f.lower().endswith(IMG_EXTS) for f in files)
        has_xml_in_root = any(f.lower().endswith('.xml') for f in files)
        
        if has_images_in_root and has_xml_in_root:
            self.dataset_info["task"] = "detection"
            self.dataset_info["detection_format"] = "pascal_voc"
            return "detection"
        
        for subdir in subdirs:
            subdir_path = os.path.join(root, subdir)
            try:
                subdir_files = os.listdir(subdir_path)
                has_images = any(f.lower().endswith(IMG_EXTS) for f in subdir_files)
                has_xml = any(f.lower().endswith('.xml') for f in subdir_files)
                
                if has_images and has_xml:
                    self.dataset_info["task"] = "detection"
                    self.dataset_info["detection_format"] = "pascal_voc_subdir"
                    self.dataset_info["detection_subdir"] = subdir
                    return "detection"
            except:
                continue
        
        if "images" in lower_dirs and "masks" in lower_dirs:
            self.dataset_info["task"] = "segmentation"
            return "segmentation"
        
        class_like = [d for d in subdirs if self._has_images(os.path.join(root, d))]
        if len(class_like) >= 2:
            self.dataset_info["task"] = "classification"
            self.dataset_info["class_folders"] = class_like
            return "classification"
        
        self.dataset_info["task"] = "unknown"
        raise RuntimeError(
            f"Görev tipi tespit edilemedi.\n"
            f"Alt klasörler: {subdirs}\n"
            f"Dosya örnekleri: {files[:5]}\n\n"
            "Desteklenen yapılar:\n"
            "- Classification: class1/, class2/, ...\n"
            "- Classification (boxes): boxes/class1/, boxes/class2/, ...\n"
            "- Segmentation: images/, masks/\n"
            "- Detection (YOLO): images/, labels/ (.txt)\n"
            "- Detection (Pascal VOC): images/ + .xml dosyaları"
        )
    
    def _standardize_and_split(self, root: str, task: str) -> str:
        """Veriyi standart yapıya dönüştür ve train/test'e böl"""
        random.seed(self.seed)
        split_root = os.path.join(self.workdir if self.workdir else ".", "standardized_dataset")
        
        if os.path.exists(split_root):
            shutil.rmtree(split_root)
        os.makedirs(split_root, exist_ok=True)
        
        logger.info(f"Veri standartlaştırılıyor ve bölünüyor: {task}")
        
        if task == "classification":
            self._split_classification(root, split_root)
        elif task == "segmentation":
            self._split_segmentation(root, split_root)
        elif task == "detection":
            self._split_detection(root, split_root)
        else:
            raise RuntimeError("Bilinmeyen görev tipi")
        
        self.dataset_info["split_root"] = split_root
        return split_root
    
    def _split_classification(self, root: str, out_root: str):
        """Classification verisi için train/test bölme"""
        if self.dataset_info.get("classification_source") == "boxes":
            root = os.path.join(root, "boxes")
        elif self.dataset_info.get("classification_source") == "boxes_flat":
            boxes_path = os.path.join(root, "boxes")
            files = [f for f in os.listdir(boxes_path) if f.lower().endswith(IMG_EXTS)]
            random.shuffle(files)
            
            n_train = int(len(files) * self.train_ratio)
            train_files = files[:n_train]
            test_files = files[n_train:]
            
            for split, subset in (("train", train_files), ("test", test_files)):
                dst = os.path.join(out_root, split, "object")
                os.makedirs(dst, exist_ok=True)
                for f in subset:
                    shutil.copy2(os.path.join(boxes_path, f), os.path.join(dst, f))
            
            summary = {"object": {"total": len(files), "train": len(train_files), "test": len(test_files)}}
            self.dataset_info["classification_counts"] = summary
            logger.info(f"Sınıf 'object': {len(train_files)} train, {len(test_files)} test")
            return
        
        class_dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        summary = {}
        
        for cls in class_dirs:
            src = os.path.join(root, cls)
            files = [f for f in os.listdir(src) if f.lower().endswith(IMG_EXTS)]
            random.shuffle(files)
            
            n_train = int(len(files) * self.train_ratio)
            train_files = files[:n_train]
            test_files = files[n_train:]
            
            for split, subset in (("train", train_files), ("test", test_files)):
                dst = os.path.join(out_root, split, cls)
                os.makedirs(dst, exist_ok=True)
                for f in subset:
                    shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
            
            summary[cls] = {
                "total": len(files),
                "train": len(train_files),
                "test": len(test_files)
            }
            logger.info(f"Sınıf '{cls}': {len(train_files)} train, {len(test_files)} test")
        
        self.dataset_info["classification_counts"] = summary
    
    def _split_segmentation(self, root: str, out_root: str):
        """Segmentation verisi için train/test bölme"""
        from PIL import Image
        
        img_dir = os.path.join(root, "images")
        msk_dir = os.path.join(root, "masks")
        images = [f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXTS)]
        
        pairs: List[Tuple[str, str]] = []
        for f in images:
            stem = os.path.splitext(f)[0]
            cand = [stem + ".png", stem + ".jpg", stem + ".jpeg"]
            match = next((m for m in cand if os.path.exists(os.path.join(msk_dir, m))), None)
            if match:
                pairs.append((f, match))
        
        random.shuffle(pairs)
        n_train = int(len(pairs) * self.train_ratio)
        train_pairs = pairs[:n_train]
        test_pairs = pairs[n_train:]
        
        for split, subset in (("train", train_pairs), ("test", test_pairs)):
            dimg = os.path.join(out_root, split, "images")
            dmsk = os.path.join(out_root, split, "masks")
            os.makedirs(dimg, exist_ok=True)
            os.makedirs(dmsk, exist_ok=True)
            for imgf, mskf in subset:
                shutil.copy2(os.path.join(img_dir, imgf), os.path.join(dimg, imgf))
                shutil.copy2(os.path.join(msk_dir, mskf), os.path.join(dmsk, mskf))
        
        self.dataset_info["segmentation_counts"] = {
            "total": len(pairs),
            "train": len(train_pairs),
            "test": len(test_pairs)
        }
        logger.info(f"Segmentation: {len(train_pairs)} train, {len(test_pairs)} test")
    
    def _split_detection(self, root: str, out_root: str):
        """Detection verisi için train/test bölme"""
        img_dir = os.path.join(root, "images")
        lbl_dir = os.path.join(root, "labels")
        images = [f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXTS)]
        
        pairs = []
        for f in images:
            stem = os.path.splitext(f)[0]
            lf = stem + ".txt"
            if os.path.exists(os.path.join(lbl_dir, lf)):
                pairs.append((f, lf))
        
        random.shuffle(pairs)
        n_train = int(len(pairs) * self.train_ratio)
        train_pairs = pairs[:n_train]
        test_pairs = pairs[n_train:]
        
        for split, subset in (("train", train_pairs), ("test", test_pairs)):
            dimg = os.path.join(out_root, split, "images")
            dlbl = os.path.join(out_root, split, "labels")
            os.makedirs(dimg, exist_ok=True)
            os.makedirs(dlbl, exist_ok=True)
            for imgf, lblf in subset:
                shutil.copy2(os.path.join(img_dir, imgf), os.path.join(dimg, imgf))
                shutil.copy2(os.path.join(lbl_dir, lblf), os.path.join(dlbl, lblf))
        
        self.dataset_info["detection_counts"] = {
            "total": len(pairs),
            "train": len(train_pairs),
            "test": len(test_pairs)
        }
        logger.info(f"Detection: {len(train_pairs)} train, {len(test_pairs)} test")
    
    def _infer_num_classes(self, split_root: str, task: str) -> Tuple[int, List[str]]:
        """Sınıf sayısını ve isimlerini çıkar"""
        if task == "classification":
            train_root = os.path.join(split_root, "train")
            classes = [d for d in os.listdir(train_root) 
                      if os.path.isdir(os.path.join(train_root, d))]
            return len(classes), sorted(classes)
        
        if task == "segmentation":
            return 2, ["background", "foreground"]
        
        if task == "detection":
            lbl_dir = os.path.join(split_root, "train", "labels")
            max_id = -1
            if os.path.isdir(lbl_dir):
                for f in os.listdir(lbl_dir):
                    if f.endswith(".txt"):
                        with open(os.path.join(lbl_dir, f), "r") as rf:
                            for line in rf:
                                parts = line.strip().split()
                                if len(parts) >= 1:
                                    try:
                                        cid = int(parts[0])
                                    except Exception:
                                        continue
                                    if cid > max_id:
                                        max_id = cid
            if max_id >= 0:
                num_classes = max_id + 1
                class_names = [f"class_{i}" for i in range(num_classes)]
                return num_classes, class_names
            return 0, []
        
        raise RuntimeError(f"Cannot infer classes for unknown task: {task}")
    
    def _create_dataloaders(self, split_root: str, task: str, 
                           batch_size: int, num_workers: int) -> Tuple[Any, Any]:
        """PyTorch DataLoader'ları oluştur"""
        logger.info(f"DataLoader'lar oluşturuluyor (batch_size={batch_size})")
        
        import torch
        from torch.utils.data import DataLoader, Dataset
        from torchvision import datasets, transforms
        from PIL import Image
        
        torch.manual_seed(self.seed)
        
        if task == "classification":
            tfm = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            full_train = datasets.ImageFolder(
                os.path.join(split_root, "train"), transform=tfm
            )
            train_loader = DataLoader(
                full_train, batch_size=batch_size, shuffle=True, 
                num_workers=num_workers
            )
            test_ds = datasets.ImageFolder(
                os.path.join(split_root, "test"), transform=tfm
            )
            test_loader = DataLoader(
                test_ds, batch_size=batch_size, shuffle=False, 
                num_workers=num_workers
            )
            return train_loader, test_loader
        
        if task == "segmentation":
            tfm = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])
            
            class SegDS(Dataset):
                def __init__(self, root: str, split: str):
                    self.img_dir = os.path.join(root, split, "images")
                    self.msk_dir = os.path.join(root, split, "masks")
                    self.files = [f for f in os.listdir(self.img_dir) 
                                 if f.lower().endswith(IMG_EXTS)]
                
                def __len__(self):
                    return len(self.files)
                
                def __getitem__(self, idx):
                    f = self.files[idx]
                    stem = os.path.splitext(f)[0]
                    img = Image.open(os.path.join(self.img_dir, f)).convert("RGB")
                    
                    for me in (".png", ".jpg", ".jpeg"):
                        pm = os.path.join(self.msk_dir, stem + me)
                        if os.path.exists(pm):
                            mpath = pm
                            break
                    else:
                        raise FileNotFoundError(f"Maske bulunamadı: {stem}")
                    
                    msk = Image.open(mpath).convert("L")
                    return tfm(img), (tfm(msk) > 0).float()
            
            train_loader = DataLoader(
                SegDS(split_root, "train"), batch_size=batch_size, 
                shuffle=True, num_workers=num_workers
            )
            test_loader = DataLoader(
                SegDS(split_root, "test"), batch_size=batch_size, 
                shuffle=False, num_workers=num_workers
            )
            return train_loader, test_loader
        
        if task == "detection":
            tfm = transforms.ToTensor()
            
            class DetDS(Dataset):
                def __init__(self, root: str, split: str):
                    self.img_dir = os.path.join(root, split, "images")
                    self.lbl_dir = os.path.join(root, split, "labels")
                    self.files = [f for f in os.listdir(self.img_dir) 
                                 if f.lower().endswith(IMG_EXTS)]
                
                def __len__(self):
                    return len(self.files)
                
                def __getitem__(self, idx):
                    f = self.files[idx]
                    stem = os.path.splitext(f)[0]
                    img = Image.open(os.path.join(self.img_dir, f)).convert("RGB")
                    
                    lbl_path = os.path.join(self.lbl_dir, stem + ".txt")
                    boxes = []
                    if os.path.exists(lbl_path):
                        with open(lbl_path, "r") as rf:
                            for line in rf:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    boxes.append([float(x) for x in parts])
                    
                    return tfm(img), torch.tensor(boxes) if boxes else torch.zeros((0, 5))
            
            train_loader = DataLoader(
                DetDS(split_root, "train"), batch_size=batch_size, 
                shuffle=True, num_workers=num_workers, collate_fn=lambda x: x
            )
            test_loader = DataLoader(
                DetDS(split_root, "test"), batch_size=batch_size, 
                shuffle=False, num_workers=num_workers, collate_fn=lambda x: x
            )
            return train_loader, test_loader
        
        raise RuntimeError(f"Bilinmeyen görev tipi: {task}")
