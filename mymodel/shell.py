# from run import run
from util.run_util import get_gpu_free
from time import sleep
from baselines.Run import run
import argparse


def monitoring_gpu(level=4, time = 10):
    # print(get_gpu_free(0)/1e9)
    count = 1
    free = get_gpu_free(0) / (1024**3)
    while free < level:
        print("检测第{}次， 当前GPU显存不足, 当前空余显存{:.2f}GB!".format(count, free))
        count += 1
        sleep(time)
        free = get_gpu_free(0) / (1024**3)
    print("GPU显存充足，开始载入代码！")
    parser = argparse.ArgumentParser()
    arg = parser.parse_args()
    arg.patient = 'WSH'
    run()


if __name__ == '__main__':
    monitoring_gpu(level=0, time=30)
