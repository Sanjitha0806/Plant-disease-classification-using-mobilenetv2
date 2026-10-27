import os
import sys
import urllib.request

MOBILENET_URL = "https://huggingface.co/rarfileexe/Plant-Disease-Detector/resolve/main/model_4_mobilenet_finetuned.keras"
DEST_NAME = "mobilenet_v2_plant_disease.keras"

def download_model(dest_dir=None):
    if dest_dir is None:
        dest_dir = os.path.dirname(os.path.abspath(__file__))
    dest_path = os.path.join(dest_dir, DEST_NAME)

    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 20_000_000:
        print(f"[OK] MobileNetV2 model already exists and ready: {dest_path} ({os.path.getsize(dest_path)} bytes)")
        return dest_path

    print(f"Downloading pre-trained MobileNetV2 weights (~24 MB) to {DEST_NAME}...")
    
    def reporthook(block_num, block_size, total_size):
        if total_size > 0:
            downloaded = block_num * block_size
            percent = min(100, int(downloaded * 100 / total_size))
            mb_down = downloaded / (1024 * 1024)
            mb_tot = total_size / (1024 * 1024)
            sys.stdout.write(f"\rProgress: [{percent:3d}%] {mb_down:.1f} MB / {mb_tot:.1f} MB")
            sys.stdout.flush()

    try:
        # Use User-Agent for Hugging Face
        req = urllib.request.Request(MOBILENET_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            block_size = 64 * 1024
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)
                if total_size > 0:
                    percent = min(100, int(downloaded * 100 / total_size))
                    mb_down = downloaded / (1024 * 1024)
                    mb_tot = total_size / (1024 * 1024)
                    sys.stdout.write(f"\rProgress: [{percent:3d}%] {mb_down:.1f} MB / {mb_tot:.1f} MB")
                    sys.stdout.flush()
                    
        print("\n[SUCCESS] MobileNetV2 model downloaded successfully!")
        return dest_path
    except Exception as e:
        print(f"\n[ERROR] Failed to download model: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise

if __name__ == "__main__":
    download_model()
