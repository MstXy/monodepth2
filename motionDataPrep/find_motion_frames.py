import os
import numpy as np

filepath = "/mnt/km-nfs/ns100002-share/zcy-exp/kitti_movements"

SEQS = [
        "02", "03", "04", "05", "06", "07", "08", "09", "10", 
        "11", "12", "13", "15", "16", "17", "18", "19", "20", "21"
    ]

output_file_path = "/mnt/km-nfs/ns100002-share/zcy-exp/monodepth2/splits/motion/all.txt" 

with open(output_file_path, "a") as output_file:

    for seq in SEQS:

        print("Working on:", seq)

        folder_path = os.path.join(filepath, seq, "SegmentationClassNpy")
        all_files = os.listdir(folder_path)
        file_count = len(all_files)

        for d in all_files:
            mask = np.load(os.path.join(folder_path, d))
            if len(np.unique(mask)) > 1:
                frame_number = int(d.strip(".npy"))
                # omit first frame and last frame
                if frame_number > 0 and frame_number < file_count - 1:
                    output_file.write("{} {} l".format(seq, frame_number) +"\n")
