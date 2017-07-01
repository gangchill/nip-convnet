from pathlib import Path
import sys
for path in Path(__file__).resolve().parents:
    if path.name == 'nip-convnet':
        sys_path = str(path)
        break
sys.path.append(sys_path)
print(sys_path)