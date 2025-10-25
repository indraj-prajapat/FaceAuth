import os
import cv2
import numpy as np

class MultiEmbedder:
    """
    Extract embeddings via:
      - InsightFace ArcFace (FaceAnalysis + ONNX recognition fallback)
      - DeepFace ArcFace (detection skipped)
      - facenet-pytorch (direct forward on aligned crop)

    Each returns a 512-D float32 L2-normalized vector or None.
    """

    def __init__(self, device='cuda'):
        self.device = device
        self._insight_app = None
        self._insight_rec = None
        self._deepface = None
        self._facenet = None
        self._torch_device = None
        print(f"[INIT] MultiEmbedder device={device}")

    # ---------------------------
    # Utils
    # ---------------------------
    def _l2(self, v):
        if v is None:
            print("[L2] skip: input is None")
            return None
        n = np.linalg.norm(v)
        if not np.isfinite(n) or n <= 0:
            print(f"[L2] invalid norm={n}")
            return None
        out = (v / n).astype(np.float32)
        print(f"[L2] ok norm={n:.4f}")
        return out

    def _check_image(self, img, tag):
        ok = True
        if img is None:
            print(f"[IMG:{tag}] is None")
            ok = False
        elif not isinstance(img, np.ndarray):
            print(f"[IMG:{tag}] not ndarray: {type(img)}")
            ok = False
        elif img.ndim != 3 or img.shape[2] != 3:
            print(f"[IMG:{tag}] expected HxWx3, got {img.shape}")
            ok = False
        else:
            print(f"[IMG:{tag}] shape={img.shape} dtype={img.dtype}")
        return ok

    # ---------------------------
    # InsightFace helpers
    # ---------------------------
    def _ensure_insight_app(self):
        if self._insight_app is None:
            print("[IFACE] init FaceAnalysis ...")
            from insightface.app import FaceAnalysis
            ctx_id = 0 if self.device.startswith('cuda') else -1
            self._insight_app = FaceAnalysis(name='buffalo_l')
            self._insight_app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            # Log ORT providers used internally
            try:
                from onnxruntime import get_available_providers
                print(f"[IFACE] ORT providers available={get_available_providers()}")
            except Exception:
                pass
            print("[IFACE] FaceAnalysis ready")

    def _ensure_insight_recognition(self):
        if self._insight_rec is not None:
            return
        import onnxruntime as ort
        root = os.path.expanduser("~/.insightface/models/buffalo_l")
        rec_path = os.path.join(root, "w600k_r50.onnx")
        print(f"[IFACE-REC] loading ONNX: {rec_path}")
        if not os.path.exists(rec_path):
            raise RuntimeError(f"[IFACE-REC] missing recognition model at {rec_path}")
        avail = ort.get_available_providers()
        print(f"[IFACE-REC] ORT available providers={avail}")
        use_gpu = self.device.startswith("cuda") and "CUDAExecutionProvider" in avail
        providers = ["CUDAExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        self._insight_rec = ort.InferenceSession(rec_path, providers=providers)
        print(f"[IFACE-REC] session providers={self._insight_rec.get_providers()}")

    def _arcface_preprocess(self, image_bgr: np.ndarray):
        # BGR->RGB, resize 112, CHW, (x-127.5)/128
        rgb = image_bgr[:, :, ::-1]
        if rgb.shape[:2] != (112, 112):
            rgb = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_AREA)
        x = rgb.astype(np.float32)
        x = (x - 127.5) / 128.0
        x = np.transpose(x, (2, 0, 1))[None, ...]
        print(f"[IFACE-REC] preproc blob shape={x.shape} dtype={x.dtype} range=[{x.min():.3f},{x.max():.3f}]")
        return x

    # ---------------------------
    # 1) InsightFace ArcFace
    # ---------------------------
    def extract_arcface_insightface(self, image_bgr: np.ndarray):
        print("\n[STEP] InsightFace ArcFace")
        if not self._check_image(image_bgr, "insightface"):
            return None
        try:
            if image_bgr.dtype != np.uint8:
                image_bgr = image_bgr.astype(np.uint8)
                print("[IFACE] cast to uint8")

            self._ensure_insight_app()

            # light padding helps detectors on tight crops
            padded = cv2.copyMakeBorder(image_bgr, 10, 10, 10, 10, borderType=cv2.BORDER_REFLECT)
            faces = self._insight_app.get(padded)
            print(f"[IFACE] faces detected={len(faces)}")
            if len(faces) > 0:
                emb = faces[0].embedding.astype(np.float32)
                print(f"[IFACE] emb shape={emb.shape} dtype={emb.dtype}")
                return self._l2(emb)

            print("[IFACE] detection failed -> ONNX recognition forward")
            self._ensure_insight_recognition()
            x = self._arcface_preprocess(image_bgr)
            inp_name = self._insight_rec.get_inputs()[0].name
            out = self._insight_rec.run(None, {inp_name: x})[0]
            emb = out[0].astype(np.float32)
            print(f"[IFACE-REC] emb shape={emb.shape} dtype={emb.dtype}")
            return self._l2(emb)

        except Exception as e:
            print(f"[IFACE][ERROR] {type(e).__name__}: {e}")
            return None

    # ---------------------------
    # 2) DeepFace ArcFace (skip detection)
    # ---------------------------
    def extract_arcface_deepface(self, image_bgr: np.ndarray):
        print("\n[STEP] DeepFace ArcFace")
        if not self._check_image(image_bgr, "deepface"):
            return None
        try:
            if self._deepface is None:
                from deepface import DeepFace
                self._deepface = DeepFace
                print("[DF] DeepFace loaded (if TF>=2.20, ensure `pip install tf-keras`)")

            rep = self._deepface.represent(
                img_path=image_bgr,
                model_name="ArcFace",
                enforce_detection=False,
                detector_backend="skip"
            )
            ok = rep and isinstance(rep, list) and "embedding" in rep[0]
            print(f"[DF] represent ok={bool(ok)} keys={list(rep[0].keys()) if ok else 'n/a'}")
            if not ok:
                return None

            emb = np.array(rep[0]["embedding"], dtype=np.float32)
            print(f"[DF] emb shape={emb.shape} dtype={emb.dtype}")
            return self._l2(emb)

        except Exception as e:
            print(f"[DF][ERROR] {type(e).__name__}: {e}")
            return None

    # ---------------------------
    # 3) facenet-pytorch (direct forward)
    # ---------------------------
    def extract_facenet_pytorch(self, image_bgr: np.ndarray):
        print("\n[STEP] facenet-pytorch")
        if not self._check_image(image_bgr, "facenet"):
            return None
        try:
            import torch
            from facenet_pytorch import InceptionResnetV1
            if self._facenet is None:
                dev = 'cuda' if (self.device.startswith('cuda') and torch.cuda.is_available()) else 'cpu'
                self._facenet = InceptionResnetV1(pretrained='vggface2').eval().to(dev)
                self._torch_device = dev
                print(f"[FN] InceptionResnetV1 loaded on {dev}")

            from PIL import Image
            import torchvision.transforms as T
            rgb = image_bgr[:, :, ::-1].astype(np.uint8)
            pil = Image.fromarray(rgb)
            tr = T.Compose([T.Resize((160, 160)), T.ToTensor()])
            t = tr(pil).unsqueeze(0)
            print(f"[FN] tensor shape={tuple(t.shape)} dtype={t.dtype}")
            with torch.no_grad():
                emb = self._facenet(t.to(self._torch_device)).cpu().numpy().reshape(-1).astype(np.float32)
            print(f"[FN] emb shape={emb.shape} dtype={emb.dtype}")
            return self._l2(emb)

        except Exception as e:
            print(f"[FN][ERROR] {type(e).__name__}: {e}")
            print("[FN][HINT] pip install facenet-pytorch")
            return None

    # ---------------------------
    # Run all and select outputs
    # ---------------------------
    def extract_embeddings(self, image_bgr: np.ndarray):
        print("\n[RUN] all extractors")
        out = {
            'insightface_arcface': self.extract_arcface_insightface(image_bgr),
            'deepface_arcface': self.extract_arcface_deepface(image_bgr),
           
            'arcface':self.extract_arcface_insightface(image_bgr)
        }
        print("\n[RUN] insightface arcface outputs:",self.extract_arcface_insightface(image_bgr))
        print("[RUN] deepface arcface outputs:",self.extract_arcface_deepface(image_bgr))
  
        for k, v in out.items():
            status = "OK" if v is not None else "FAIL"
            print(f"[RUN] {k}: {status}")

        # Priority: InsightFace > DeepFace > Facenet
        out1 = {
            'arcface': out['insightface_arcface']  ,
            'adaface': out['deepface_arcface'],  # placeholder
            'elastic': out['insightface_arcface']          # placeholder
        }
        
        if out1['arcface'] is None:
            print("[RUN][WARN] no embedding produced; check earlier logs")
            print("[RUN][HINT] If ORT GPU expected, ensure onnxruntime-gpu installed and CUDA provider available")
        print("[RUN] complete")
        return out1
