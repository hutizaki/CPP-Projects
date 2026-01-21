"""
MNIST Data Loader
Loads MNIST data in the same format as the C++ implementation
"""
import struct
import numpy as np

def read_bytes_be(file, byte_size):
    """Read bytes in big-endian format"""
    buffer = file.read(byte_size)
    if len(buffer) != byte_size:
        raise IOError(f"Expected {byte_size} bytes, got {len(buffer)}")
    
    value = 0
    for i in range(byte_size):
        value = (value << 8) | buffer[i]
    return value

def load_mnist_labels(filepath):
    """Load MNIST labels from file"""
    label_magic_num = 2049
    
    with open(filepath, 'rb') as f:
        magic_num = read_bytes_be(f, 4)
        if label_magic_num != magic_num:
            raise ValueError(f"Magic number mismatch! Expected {label_magic_num}, got {magic_num}")
        
        num_labels = read_bytes_be(f, 4)
        print(f"Loading {num_labels} labels...")
        
        labels = []
        tenth = num_labels // 10
        percent = 0
        
        for i in range(num_labels):
            if i % tenth == 0 and i > 0:
                percent += 10
                print(f"{percent}% complete")
            
            byte = f.read(1)
            if not byte:
                raise IOError(f"Unexpected end of file at label {i}")
            labels.append(byte[0])
        
        print("Labels have been fully loaded")
        return np.array(labels, dtype=np.int32)

def load_mnist_images(filepath):
    """Load MNIST images from file"""
    image_magic_num = 2051
    
    with open(filepath, 'rb') as f:
        magic_num = read_bytes_be(f, 4)
        if image_magic_num != magic_num:
            raise ValueError(f"Magic number mismatch! Expected {image_magic_num}, got {magic_num}")
        
        num_images = read_bytes_be(f, 4)
        num_rows = read_bytes_be(f, 4)
        num_cols = read_bytes_be(f, 4)
        
        print(f"Loading {num_images} images ({num_rows}x{num_cols})...")
        
        images = []
        tenth = num_images // 10
        percent = 0
        
        for i in range(num_images):
            if i % tenth == 0 and i > 0:
                percent += 10
                print(f"{percent}% complete")
            
            image = []
            for j in range(num_rows * num_cols):
                byte = f.read(1)
                if not byte:
                    raise IOError(f"Unexpected end of file at image {i}, pixel {j}")
                # Normalize to [0, 1] like C++ implementation
                image.append(byte[0] / 255.0)
            
            images.append(image)
        
        print("Images have been fully loaded")
        return np.array(images, dtype=np.float32)
