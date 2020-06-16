from run_model import DanTrainer
import argparse


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_ratio', type=float, default=0.0001, help='learning ratio of model')
    parser.add_argument('-dim', '--output_dim', type=int, default=32, help='number of hidden units in encoder')
    parser.add_argument('-bs', '--batch_size', type=int, default=16, help='number of bath size')
    parser.add_argument('-gpu', '--GPU', type=int, default=0, help='GPU ID')
    parser.add_argument('-ep', '--epoch', type=int, default=20, help='number of epoch')

    parser.add_argument('-trp', '--train_path', type=str, default="../preprocess/train_{}.csv",
                        help='training data path')
    parser.add_argument('-tep', '--test_path', type=str, default="../preprocess/test_{}.csv",
                        help='test data path')
    parser.add_argument('-vap', '--val_path', type=str, default="../preprocess/val_{}.csv",
                        help='val data path')
    parser.add_argument('-p', '--patient', type=str, default="BDP", help='patient name')
    parser.add_argument('-m', '--model', type=str, default="train", help='style of train')
    parser.add_argument('-few', '--few_show_learning', type=bool, default=True, help='keep few shot learning open')
    parser.add_argument('-fr', '--few_show_learning_ratio', type=float, default=0.25, help='few shot learning ratio')
    parser.add_argument('-em', '--embedding', type=str, default="cnn", help='method of embedding')
    parser.add_argument('-lac', '--label_classifier_name', type=str, default='transformer',
                        help='choosing label classifier')
    parser.add_argument('-chp', '--check_point', type=bool, default=False, help='Whether to continue training')

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
    few_shot_ratio = args.few_show_learning_ratio
    few_show_learning = args.few_show_learning
    label_classifier_name = args.label_classifier_name
    check_point = args.check_point
    train_path, test_path, val_path = train_path.format(patient), test_path.format(patient), val_path.format(patient)

    print(args)
    dan_train = DanTrainer(epoch, bath_size=batch_size, lr=lr, gpu=gpu, train_path=train_path, test_path=test_path,
                           val_path=val_path, model=model, encoder_name=embedding, few_shot=few_show_learning,
                           few_show_ratio=few_shot_ratio, label_classifier_name=label_classifier_name,
                           check_point=check_point)
    if model == 'train':
        dan_train.train()
    elif model == 'test':
        dan_train.test()


if __name__ == '__main__':
    run()
