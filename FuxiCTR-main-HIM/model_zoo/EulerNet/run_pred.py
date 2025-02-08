import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys
sys.path.append('/data/workspace/FuxiCTR-main')
import logging
import pandas as pd
import fuxictr_version
from fuxictr import datasets
from datetime import datetime
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import RankDataLoader
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.preprocess import FeatureProcessor, build_dataset
import src
import gc
import argparse
import os
from pathlib import Path

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
    parser.add_argument('--expid', type=str, default='EulerNet_default', help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu index, -1 for cpu')
    args = vars(parser.parse_args())

    experiment_id = args['expid']
    params = load_config(args['config'], experiment_id)
    params['gpu'] = args['gpu']
    set_logger(params)
    logging.info("Params: " + print_to_json(params))
    seed_everything(seed=params['seed'])

    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    if params["data_format"] == "csv":
        # Build feature_map and transform data
        feature_encoder = FeatureProcessor(**params)
        params["train_data"], params["valid_data"], params["test_data"] = \
            build_dataset(feature_encoder, **params)
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params)
    logging.info("Feature specs: " + print_to_json(feature_map.features))

    model_class = getattr(src, params['model'])
    model = model_class(feature_map, **params)

    model.load_weights("checkpoints/twitter_data_train/EulerNet_default.model")
    # model.load_weights("checkpoints/tencent_train/EulerNet_default.model")
    model.count_parameters()  # print number of parameters used in model

    # 先把test_data处理成parquet的格式
    # 然后传给RankDataLoader(test_data = test_data)
    test_gen = RankDataLoader(feature_map, stage='test', **params).make_iterator()

    y_pred = model.predict(test_gen)

    # 将数组转换为 DataFrame
    df = pd.DataFrame(y_pred, columns=['pred'])

    # 保存为 CSV 文件
    df.to_csv('../../data/'+params["dataset_id"]+ '/' + params["dataset_id"] + '_output_'+params['model']+'.csv', index=False)

    print("finished!")