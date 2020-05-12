from run_model import DanTrainer
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_ratio', type=float, default=0.001, help='learning ratio of model')
    parser.add_argument('-dim', '--output_dim', type=int, default=32, help='number of hidden units in encoder')
    parser.add_argument('-bs', '--batch_size', type=int, default=16, help='number of bath size')
    parser.add_argument('-gpu', '--GPU', type=int, default=0, help='GPU ID')
    parser.add_argument('-ep', '--epoch', type=int, default=10, help='number of epoch')

    parser.add_argument('-trp', '--train_path', type=str, default="../preprocess/train_{}.csv",
                        help='training data path')
    parser.add_argument('-tep', '--test_path', type=str, default="../preprocess/test_{}.csv", help='test data path')
    parser.add_argument('-vap', '--val_path', type=str, default="../preprocess/val_{}.csv", help='val data path')
    parser.add_argument('-p', '--patient', type=str, default="BDP", help='patient name')
    parser.add_argument('-m', '--model', type=str, default="train", help='style of train')
    parser.add_argument('-em', '--embedding', type=str, default="vae", help='method of embedding')


    args = parser.parse_args()

    # 超参设置
    lr = args.learning_ratio
    train_path = args.train_path
    test_path = args.test_path
    val_path = args.val_path
    batch_size = args.batch_size
    epoch = args.epoch
    gpu = args.GPU
    patient = args.patient
    model = args.model
    embedding = args.embedding
    train_path, test_path, val_path = train_path.format(patient), test_path.format(patient), val_path.format(patient)
    dan_train = DanTrainer(epoch, bath_size=batch_size, lr=lr, GPU=gpu, train_path=train_path, test_path=test_path,
                           val_path=val_path, model=model, )
    if model == 'train':
        dan_train.train()
    elif model == 'val':
        dan_train.val()
