import os, sys, multiprocessing

if len(sys.argv) < 3:
    print "Usage: python gen_test.py input_folder output_folder"
    exit(1)

fi = sys.argv[1]
fo = sys.argv[2]

cmd = "convert -resize 96x96\! "
imgs = os.listdir(fi)

for img in imgs:
    md = ""
    md += cmd
    md += os.path.join(fi,img)
    md += " " + os.path.join(fo,img)
    os.system(md)
