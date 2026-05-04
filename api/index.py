import sys
import os

# Tambahkan root folder ke path
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

# Set working directory ke root agar path CSV & JSON ketemu
os.chdir(root)

from app import app
