# IDXDataLoader.py
import gzip
import numpy as np
import struct
from typing import Optional, Tuple
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')  # Interaktif olmayan arka uç

def load_idx(file_path: str) -> np.ndarray:
    with gzip.open(file_path, 'rb') as f:
        magic_number = struct.unpack('>I', f.read(4))[0]
        if magic_number not in (0x00000803, 0x00000801):
            raise ValueError(f"Geçersiz Magic Number: {magic_number}")
        
        num_items = struct.unpack('>I', f.read(4))[0]
        shape: Optional[Tuple[int, ...]] = None
        if magic_number == 0x00000803:  # 3D veri (ör. görüntüler)
            rows, cols = struct.unpack('>II', f.read(8))
            shape = (num_items, rows, cols)
        elif magic_number == 0x00000801:  # 1D veri (ör. etiketler)
            shape = (num_items,)
        
        if shape is None:
            raise ValueError(f"Bilinmeyen Magic Number: {magic_number}")

        buffer = f.read()
        data = np.frombuffer(buffer, dtype=np.uint8).reshape(shape)
    return data

train_images_path = 'train-images-idx3-ubyte.gz'
train_labels_path = 'train-labels-idx1-ubyte.gz'

if __name__ == "__main__":
    try:
        train_images = load_idx(train_images_path)
        train_labels = load_idx(train_labels_path)
        print("Eğitim görüntü boyutu:", train_images.shape)
        print("Eğitim etiket boyutu:", train_labels.shape)

        plt.imshow(train_images[0], cmap='gray')
        plt.title(f"Etiket: {train_labels[0]}")
        plt.axis('off')
        plt.savefig('output_image.png')
        print("Görüntü 'output_image.png' dosyasına kaydedildi.")

    except FileNotFoundError as e:
        print(f"Dosya bulunamadı: {e}")
    except ValueError as e:
        print(f"Geçersiz dosya formatı: {e}")

