import h5py
import numpy as np
import os

root_path = '/workspace/data/GOPRO_original_voxel/test'
h5_folder = os.listdir(root_path)
h5_folder.sort()

save_folder = '/workspace/data/V2S/test'


for h5 in h5_folder:
    h5_path = os.path.join(root_path, h5)
    with h5py.File(h5_path, 'a') as f:
        print(f"\n### Inspecting file: {h5} ###")
        print("Existing keys:", list(f.keys()))
        
        # 'images' 그룹에서 blur 이미지 불러오기
        imgs = f['images']
        gt_events = f['voxels']
        gen_event = f['B2V']
        scene = h5_path.split('/')[-1][:-3]

        folder_path = os.path.join(save_folder,scene)


        for event in gen_event:
            gt_event_key = event.replace('image','voxel')
            gt_event = gt_events[f'{gt_event_key}'][:]

            voxel = gen_event[f'{event}'][:]
            save_folder_path = os.path.join(folder_path,event)
            os.makedirs(save_folder_path, exist_ok=True)

            scer = np.zeros_like(voxel)
            scer[0] = -1800 * (voxel[0] + voxel[1] + voxel[2])
            scer[1] = -1800 *  (voxel[1] + voxel[2])
            scer[2] = -1800 * voxel[2]
            scer[3] = voxel[3] * 1800
            scer[4] = (voxel[3] + voxel[4]) * 1800
            scer[5] = (voxel[3] + voxel[4] + voxel[5]) * 1800

            np.save(os.path.join(save_folder_path,'out.npy'), scer)

            # max_val = np.max(np.abs(gt_event))
            # gt_event = gt_event / max_val

            # max_val = np.max(np.abs(scer))
            # scer = scer / max_val

            # gt_event = (gt_event + 1) / 2
            # scer = (scer + 1) / 2

            # rmse = np.sqrt(np.mean((gt_event - scer) ** 2))


    