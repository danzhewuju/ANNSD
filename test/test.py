from util.seeg_utils import read_edf_raw, read_annotations

print("Hello")
path = "/home/yh/yh/dataset/positiveDataAvailable/BDP/BDP{}.edf"
for i in range(1, 6):
    new_path = path.format(i)
    data = read_edf_raw(new_path)
    print(read_annotations(data))
