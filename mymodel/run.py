import argparse

from run_model import Dan


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_ratio', type=float, default=0.0001, help='learning ratio of model')
    parser.add_argument('-dim', '--output_dim', type=int, default=32, help='number of hidden units in encoder')
    parser.add_argument('-bs', '--batch_size', type=int, default=16, help='number of bath size')
    parser.add_argument('-gpu', '--GPU', type=int, default=3, help='GPU ID')
    parser.add_argument('-ep', '--epoch', type=int, default=30, help='number of epoch')

    parser.add_argument('-trp', '--train_path', type=str, default="../preprocess/train_{}.csv",
                        help='training data path')
    parser.add_argument('-tep', '--test_path', type=str, default="../preprocess/test_{}.csv",
                        help='test data path')
    parser.add_argument('-vap', '--val_path', type=str, default="../preprocess/val_{}.csv",
                        help='val data path')
    parser.add_argument('-atp', '--attention_path', type=str, default="../preprocess/attention_{}.csv",
                        help='attention data path')
    parser.add_argument('-p', '--patient', type=str, default="ALL", help='patient name')
    parser.add_argument('-m', '--model', type=str, default="prediction_batch", help='style of train')
    parser.add_argument('-few', '--few_show_learning', type=bool, default=True, help='keep few shot learning open')
    parser.add_argument('-fr', '--few_show_learning_ratio', type=float, default=0.2, help='few shot learning ratio')
    parser.add_argument('-em', '--embedding', type=str, default="cnn", help='method of embedding')
    parser.add_argument('-lac', '--label_classifier_name', type=str, default='transformer',
                        help='choosing label classifier')
    parser.add_argument('-chp', '--check_point', type=bool, default=False, help='Whether to continue training')
    parser.add_argument('-att', '--attention_matrix', type=bool, default=False, help='Whether to get attention matrix')
    parser.add_argument('-rec', '--recoding', type=bool, default=False, help='Whether to recode result for every file')
    parser.add_argument('-unb', '--unbalance_data', type=int, default=5,
                        help="The negative sample is x times the positive sample")
    # -------------------------------------------单文件测试模块----------------------------------------------------------
    # parser.add_argument('-fp', '--file_path', type=str,
    #                     default="/data/yh/dataset/raw_data/BDP/BDP_Pre_seizure/BDP_SZ1_pre_seizure_raw.fif",
    #                     help='Testing file path')
    # /home/yh/yh/dataset/raw_data/SYF/SYF_Pre_seizure/SYF_SZ1_pre_seizure_raw-1.fif
    parser.add_argument('-fp', '--file_path', type=str,
                        default="/home/yh/yh/dataset/raw_data/SYF/SYF_Pre_seizure/SYF_SZ1_pre_seizure_raw-1.fif",
                        help='Testing file path')
    parser.add_argument('-ts', '--test_seizure', type=str, default='pre_seizure',
                        help='Seizure status, Please input: pre_seizure or non_seizure')
    parser.add_argument('-dl', '--data_length', type=int, default=15, help='data length of segment')
    parser.add_argument('-cp', '--config_path', type=str, default='../preprocess/config/config.json',
                        help='config file path')
    parser.add_argument('-sf', '--save_file', type=str, default='../log/{}_prediction_result.csv',
                        help='save file path')
    # ---------------------------------------------------------------------------------------------------------------
    # 批量参数模块
    parser.add_argument('-fpl', '--file_path_list', type=str,
                        default="/data/yh/Python/SEEG-Timing/log/raw_data_info.csv",
                        help='Testing file path')
    # ---------------------------------------------------------------------------------------------------------------

    args = parser.parse_args()

    # 超参设置
    lr = args.learning_ratio
    train_path = args.train_path
    test_path = args.test_path
    val_path = args.val_path
    att_path = args.attention_path
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
    att = args.attention_matrix
    rec = args.recoding
    isUnbalance = args.unbalance_data  # when is Unbalance == 1 mean balance
    train_path, test_path, val_path, att_path = train_path.format(patient), test_path.format(patient), val_path.format(
        patient), att_path.format(patient)

    print(args)
    dan_train = Dan(patient, epoch, bath_size=batch_size, lr=lr, gpu=gpu, train_path=train_path,
                    test_path=test_path,
                    val_path=val_path, att_path=att_path, model=model, encoder_name=embedding,
                    few_shot=few_show_learning,
                    few_show_ratio=few_shot_ratio, label_classifier_name=label_classifier_name,
                    check_point=check_point, att=att, isUnbalance=isUnbalance)
    if model == 'train':
        dan_train.train()
    elif model == 'test':
        dan_train.test(recoding=rec)
    elif model == 'attention':  # 需要计算attention的值
        dan_train.test_attention()
    #  运行脚本：python ./run.py -m prediction -lac transformer -p BDP -gpu 1
    elif model == 'prediction':  # 单个样本的预测模型

        file_path, label, save_file, data_length, config_path = args.file_path, args.test_seizure, args.save_file, args.data_length, args.config_path
        # patient = "SYF"
        save_file = save_file.format(patient)
        dan_train.prediction_real_data(file_path, label, save_file, data_length, config_path)
    elif model == "prediction_batch":
        file_path_list, label, save_file, data_length, config_path = args.file_path_list, args.test_seizure, args.save_file, args.data_length, args.config_path
        dan_train.prediction_batch_real_data(file_path_list, label, data_length, config_path)
    else:
        print("Your choice does not exist!")

    return


if __name__ == '__main__':
    run()
