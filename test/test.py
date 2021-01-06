from util.seeg_utils import read_edf_raw, read_annotations

print("Hello")
path = "/home/yh/yh/dataset/positiveDataAvailable/BDP/BDP{}.edf"
# for i in range(1, 6):
i = 1
new_path = path.format(i)
data = read_edf_raw(new_path)
ano = read_annotations(data)
print(ano)
print(ano['onset'])

i = 2
new_path = path.format(i)
data = read_edf_raw(new_path)
ano = read_annotations(data)
print(ano)
print(ano['onset'])

i = 3
new_path = path.format(i)
data = read_edf_raw(new_path)
ano = read_annotations(data)
print(ano)
print(ano['onset'])

i = 4
new_path = path.format(i)
data = read_edf_raw(new_path)
ano = read_annotations(data)
print(ano)
print(ano['onset'])

i = 5
new_path = path.format(i)
data = read_edf_raw(new_path)
ano = read_annotations(data)
print(ano)
print(ano['onset'])

i = 6
new_path = path.format(i)
data = read_edf_raw(new_path)
ano = read_annotations(data)
print(ano)
print(ano['onset'])


