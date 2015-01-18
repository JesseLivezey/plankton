import os, sys

if len(sys.argv) < 3:
    print "Usage: python gen_train.py input_folder output_folder"
    exit(1)
fi = sys.argv[1]
fo = sys.argv[2]

if not os.path.exists(fo):
    os.makedirs(fo)

cmd = "convert -resize 48x48\! "
classes = os.listdir(fi)

os.chdir(fo)
for cls in classes:
    try:
        os.mkdir(cls)
    except:
        pass
    imgs = os.listdir(os.path.join(fi,cls))
    for img in imgs:
        md = ""
        md += cmd
        md += os.path.join(fi,cls,img)
        md += " " + os.path.join(fo,cls,img)
        os.system(md)
