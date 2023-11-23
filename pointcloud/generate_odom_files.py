SEQ = 0
SIDE = "l"

imgtotal = 4541
output_fn = "/mnt/km-nfs/ns100002-share/zcy-exp/monodepth2/splits/odom/test_files_{:02d}.txt".format(SEQ)

line = ""
for i in range(imgtotal - 1):
    line += "{} {} {}\n".format(SEQ, i, SIDE)

with open(output_fn, 'w') as f:
    f.write(line)