import h5py
import numpy as np
import os

root_path = '/workspace/data/GOPRO/train'
h5_folder = os.listdir(root_path)
h5_folder.sort()

save_folder = '/workspace/data/FFTformer/results/gen_train/'

for h5 in h5_folder:
    h5_path = os.path.join(root_path, h5)
    with h5py.File(h5_path, 'a') as f:
        print(f"\n### Inspecting file: {h5} ###")
        print("Existing keys:", list(f.keys()))
        
        # 'images' 그룹에서 blur 이미지 불러오기
        imgs = f['images']
        gen_event = f['gen_event']
        scene = h5_path.split('/')[-1][:-3]

        folder_path = os.path.join(save_folder,scene)


        for event in gen_event:
            event_data = gen_event[f'{event}'][:]
            save_folder_path = os.path.join(folder_path,event)
            os.makedirs(save_folder_path, exist_ok=True)
            np.save(os.path.join(save_folder_path,'out.npy'), event_data )
    