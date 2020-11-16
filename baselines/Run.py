import argparse

from baseline import Baselines


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_ratio', type=float, default=0.001, help='learning ratio of model')
    parser.add_argument('-dim', '--output_dim', type=int, default=32, help='number of hidden units in encoder')
    parser.add_argument('-bs', '--batch_size', type=int, default=8, help='number of bath size')
    parser.add_argument('-gpu', '--GPU', type=int, default=0, help='GPU ID')
    parser.add_argument('-ep', '--epoch', type=int, default=5, help='number of epoch')

    parser.add_argument('-trp', '--train_path', type=str, default="../preprocess/train_{}.csv",
                        help='training data path')
    parser.add_argument('-tep', '--test_path', type=str, default="../preprocess/test_{}.csv",
                        help='test data path')
    parser.add_argument('-vap', '--val_path', type=str, default="../preprocess/val_{}.csv",
                        help='val data path')
    parser.add_argument('-atp', '--attention_path', type=str, default="../preprocess/attention_{}.csv",
                        help='attention data path')
    parser.add_argument('-p', '--patient', type=str, default="BDP", help='patient name')
    parser.add_argument('-ban', '--basename', type=str, default='dpCnn',
                        help='The name of baselines')  # cnnLstm, cnnVoting, cnnTransformer, cnnSvm, vdCnn, dpCnn
    parser.add_argument('-m', '--model', type=str, default="train", help='style of train')
    parser.add_argument('-few', '--few_show_learning', type=bool, default=True, help='keep few shot learning open')
    parser.add_argument('-fr', '--few_show_learning_ratio', type=float, default=0.2, help='few shot learning ratio')
    parser.add_argument('-chp', '--check_point', type=bool, default=False, help='Whether to continue training')
    parser.add_argument('-rec', '--recoding', type=bool, default=False, help='Whether to recoder result for every file')
    parser.add_argument('-unb', '--unbalance_data', type=int, default=5,
                        help="The negative sample is x times the positive sample")

    args = parser.parse_args()

    # 超参设置
    lr = args.learning_ratio
    train_path = args.train_path
    dim = args.output_dim
    test_path = args.test_path
    val_path = args.val_path
    att_path = args.attention_path
    batch_size = args.batch_size
    basename = args.basename
    epoch = args.epoch
    gpu = args.GPU
    patient = args.patient
    model = args.model
    few_shot_ratio = args.few_show_learning_ratio
    Unbalance = args.unbalance_data  # when is Unbalance == 1 mean balance
    few_show_learning = args.few_show_learning
    check_point = args.check_point
    train_path, test_path, val_path, att_path = train_path.format(patient), test_path.format(patient), val_path.format(
        patient), att_path.format(patient)

    print(args)
    bl = Baselines(patient, epoch, batch_size, dim=dim, lr=lr, gpu=gpu, train_path=train_path, test_path=test_path,
                   val_path=val_path, model=model, basename=basename, few_shot=few_show_learning,
                   few_show_ratio=few_shot_ratio, check_point=check_point, Unbalance=Unbalance)
    if model == 'train':
        bl.train()
    elif model == 'test':
        bl.test()


if __name__ == '__main__':
    run()

#
