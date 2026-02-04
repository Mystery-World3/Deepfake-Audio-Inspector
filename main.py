import argparse
import os
import sys
from src.processor import extract_audio_features
from src.detector import DeepfakeDetector

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_banner():
    print(f"{Colors.OKBLUE}{Colors.BOLD}")
    print("="*50)
    print("      AUDIO DEEPFAKE DETECTOR - AI INSPECTOR      ")
    print("      Developed by: Muhammad Mishbahul Muflihin         ")
    print("="*50)
    print(f"{Colors.ENDC}")

def main():
    print_banner()

    # 1. Konfigurasi Argumen CLI
    parser = argparse.ArgumentParser(description="Deteksi apakah audio adalah suara manusia asli atau hasil AI.")
    parser.add_argument("-i", "--input", help="Path ke file audio (format .wav atau .mp3)", required=True)
    parser.add_argument("-m", "--model", help="Path ke model PyTorch (.pth)", default="models/audio_model.pth")
    
    args = parser.parse_args()

    # 2. Cek file audio 
    if not os.path.exists(args.input):
        print(f"{Colors.FAIL}[!] Error: File audio '{args.input}' tidak ditemukan.{Colors.ENDC}")
        sys.exit(1)

    # 3. Inisialisasi Detector (Otak AI)
    if not os.path.exists(args.model):
        print(f"{Colors.WARNING}[!] Warning: Model weights '{args.model}' tidak ditemukan.")
        print("[!] Tool akan berjalan menggunakan 'Dummy Prediction' untuk demonstrasi.{Colors.ENDC}\n")
        detector = None
    else:
        detector = DeepfakeDetector(args.model)

    try:
        # 4. Proses Ekstraksi Fitur
        print(f"{Colors.OKBLUE}[*] Mengekstraksi fitur audio (MFCC)...{Colors.ENDC}")
        features = extract_audio_features(args.input)
        
        # 5. Prediksi
        print(f"{Colors.OKBLUE}[*] Menganalisis pola suara dengan Deep Learning...{Colors.ENDC}")
        
        if detector:
            label, confidence = detector.predict(features)
        else:
            label = 0 # 0 = Real, 1 = Fake
            confidence = 0.95

        # 6. Tampilkan Hasil Akhir
        print("\n" + "="*30)
        print(f"{Colors.BOLD}HASIL ANALISIS:{Colors.ENDC}")
        
        if label == 0:
            print(f"{Colors.OKGREEN}[+] KATEGORI: AUDIO ASLI (HUMAN VOICE)")
            print(f"[+] TINGKAT KEYAKINAN: {confidence:.2%}{Colors.ENDC}")
        else:
            print(f"{Colors.FAIL}[-] KATEGORI: AUDIO PALSU (AI GENERATED/DEEPFAKE)")
            print(f"[-] TINGKAT KEYAKINAN: {confidence:.2%}{Colors.ENDC}")
        print("="*30 + "\n")

    except Exception as e:
        print(f"{Colors.FAIL}[!] Terjadi kesalahan: {str(e)}{Colors.ENDC}")

if __name__ == "__main__":
    main()