import os, sys, multiprocessing

if len(sys.argv) < 3:
    print "Usage: python gen_test.py input_folder output_folder"
    exit(1)

fi = sys.argv[1]
fo = sys.argv[2]

cmd = "convert -resize 48x48 -gravity center -background white -extent 48x48 "
imgs = os.listdir(fi)

for img in imgs:
    md = ""
    md += cmd
    md += os.path.join(fi,img)
    md += " " + os.path.join(fo,img)
    os.system(md)
