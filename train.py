import matlab.engine  # Must import matlab.engine first

import os
import torch
import numpy as np
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from logger import Logger
from model import BackboneNet
from dataset import SingleVideoDataset
from utils import get_dataset, eval_thumos_recog, load_config_file

import pdb

device = torch.device('cuda')


def get_diversity_loss(scores):

    assert (len(scores) > 1)

    softmax_scores = [F.softmax(i, dim=2) for i in scores]

    S1 = torch.stack(softmax_scores).permute(1, 3, 0, 2)  # (1,21,4,320)
    S2 = torch.stack(softmax_scores).permute(1, 3, 2, 0)  # (1,21,320,4)
    S1_norm = S1.norm(p=2, dim=3, keepdim=True)  # + 1e-6 carefule here  (1,21,4,1)
    S2_norm = S2.norm(p=2, dim=2, keepdim=True)  # (1,21,1,4)

    R = torch.matmul(S1, S2) / (torch.matmul(S1_norm, S2_norm) + 1e-6)  # (1,21,4,4)

    I = torch.eye(len(scores)).to(device)  # (4,4)
    I = I.repeat((R.shape[0], R.shape[1], 1, 1))  # (1,21,4,4)

    pair_num = len(scores) * (len(scores) - 1)  # 4 X 3

    loss_div = F.relu(R - I).sum(-1).sum(-1) / pair_num  # (1,21)
    loss_div = loss_div.mean()  # 0.9994

    return loss_div


def get_norm_regularization(scores):

    video_len = scores[0].shape[1]

    assert (video_len > 0)

    S_raw = torch.stack(scores).permute(1, 3, 0, 2)  # (1,21,4,10)
    S_raw_norm = S_raw.norm(p=1, dim=3) / video_len  # (1,21,4)

    deviations = S_raw_norm - S_raw_norm.mean(dim=2, keepdim=True).repeat(
        1, 1, S_raw_norm.shape[2])  # (1,21,4)

    loss_norm = torch.abs(deviations).mean()  # 0.06

    return loss_norm


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config-file', type=str)
    parser.add_argument('--train-subset-name', type=str)
    parser.add_argument('--test-subset-name', type=str)

    parser.add_argument('--test-log', dest='test_log', action='store_true')
    parser.add_argument('--no-test-log', dest='test_log', action='store_false')
    parser.set_defaults(test_log=True)

    args = parser.parse_args()

    print(args.config_file)
    print(args.train_subset_name)
    print(args.test_subset_name)
    print(args.test_log)

    all_params = load_config_file(args.config_file)  # 获取所有参数
    locals().update(all_params)

    def test(model, loader, modality):

        assert (modality in ['both', 'rgb', 'flow'])

        pred_score_dict = {}
        label_dict = {}

        correct = 0
        total_cnt = 0
        total_loss = {
            'cls': 0,
            'div': 0,
            'norm': 0,
            'sum': 0,
        }

        criterion = nn.CrossEntropyLoss(reduction='elementwise_mean')

        with torch.no_grad():

            model.eval()

            """
             data:
             {'video_name':video_validation_0000683-0,
             'rgb': (70,1024),
             'flow':(70,1024),
             'frame_rate':30,
             'frame_cnt':1137,
             'anno':[[0.3,5.5],[7.5,13.5],[15.9,18.1],[23.8,25.7]],
             'label':0,
             'weight':1.0  
             }
            
            """
            for _, data in enumerate(loader):  # No shuffle

                video_name = data['video_name'][0]
                label = data['label'].to(device)
                weight = data['weight'].to(device).float()

                if label.item() == action_class_num:
                    continue
                else:
                    total_cnt += 1

                if modality == 'both':
                    rgb = data['rgb'].to(device).squeeze(0)
                    flow = data['flow'].to(device).squeeze(0)
                    model_input = torch.cat([rgb, flow], dim=2)
                elif modality == 'rgb':
                    model_input = data['rgb'].to(device).squeeze(0)
                else:
                    model_input = data['flow'].to(device).squeeze(0)

                model_input = model_input.transpose(2, 1)   # (1,1024,70)
                _, _, out, scores, _ = model(model_input)   # out: (1,21)  scores:  [(1,60,21),(1,60,21),(1,60,21),(1,60,21)]

                out = out.mean(0, keepdim=True)

                loss_cls = criterion(out, label) * weight
                total_loss['cls'] += loss_cls.item()

                if diversity_reg:

                    loss_div = get_diversity_loss(scores) * weight
                    loss_div = loss_div * diversity_weight

                    loss_norm = get_norm_regularization(scores) * weight
                    loss_norm = loss_norm * diversity_weight

                    total_loss['div'] += loss_div.item()
                    total_loss['norm'] += loss_norm.item()

                out = out[:, :action_class_num]  # Remove bg
                pred = torch.argmax(out, dim=1)
                correct += (pred.item() == label.item())

                ###############

                video_key = ''.join(video_name.split('-')
                                    [:-1])  # remove content after the last -

                pred_score_dict[video_key] = out.cpu().numpy()

                if video_key not in label_dict.keys():
                    label_dict[video_key] = np.zeros((1, action_class_num))

                label_dict[video_key][0, label.item()] = 1
                ###############

        accuracy = correct / total_cnt
        total_loss[
            'sum'] = total_loss['cls'] + total_loss['div'] + total_loss['norm']
        avg_loss = {k: v / total_cnt for k, v in total_loss.items()}

        ##############
        pred_score_matrix = []
        label_matrix = []
        for k, v in pred_score_dict.items():
            pred_score_matrix.append(v)
            label_matrix.append(label_dict[k])

        _, mean_ap = eval_thumos_recog(
            np.concatenate(pred_score_matrix, axis=0),
            np.concatenate(label_matrix, axis=0), action_class_num)

        return accuracy, avg_loss, mean_ap

    def train(train_train_loader, train_test_loader, test_test_loader, modality,
              naming):

        assert (modality in ['both', 'rgb', 'flow'])

        log_dir = os.path.join('logs', naming, modality)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logger = Logger(log_dir)

        save_dir = os.path.join('models', naming)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 构建模型
        if modality == 'both':
            model = BackboneNet(in_features=feature_dim * 2,
                                **model_params).to(device)
        else:
            model = BackboneNet(in_features=feature_dim,
                                **model_params).to(device)

        optimizer = optim.Adam(model.parameters(),
                               lr=learning_rate,
                               weight_decay=weight_decay)

        if learning_rate_decay:
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[max_step_num // 2], gamma=0.1)

        optimizer.zero_grad()

        criterion = nn.CrossEntropyLoss(reduction='elementwise_mean')

        update_step_idx = 0
        single_video_idx = 0
        loss_recorder = {
            'cls': 0,
            'div': 0,
            'norm': 0,
            'sum': 0,
        }

        while update_step_idx < max_step_num: # 10005

            # Train loop
            for _, data in enumerate(train_train_loader):

                model.train()

                single_video_idx += 1

                label = data['label'].to(device)
                weight = data['weight'].to(device).float()

                if modality == 'both':
                    rgb = data['rgb'].to(device)
                    flow = data['flow'].to(device)
                    model_input = torch.cat([rgb, flow], dim=2)
                elif modality == 'rgb':
                    model_input = data['rgb'].to(device)
                else:
                    model_input = data['flow'].to(device)

                model_input = model_input.transpose(2, 1)
                _, _, out, scores, _ = model(model_input)

                loss_cls = criterion(out, label) * weight

                if diversity_reg:
                    loss_div = get_diversity_loss(scores) * weight
                    loss_div = loss_div * diversity_weight

                    loss_norm = get_norm_regularization(scores) * weight
                    loss_norm = loss_norm * diversity_weight

                    loss = loss_cls + loss_div + loss_norm

                    loss_recorder['div'] += loss_div.item()
                    loss_recorder['norm'] += loss_norm.item()

                else:
                    loss = loss_cls

                loss_recorder['cls'] += loss_cls.item()
                loss_recorder['sum'] += loss.item()

                loss.backward()

                # Test and Update
                if single_video_idx % batch_size == 0:

                    # Test
                    if update_step_idx % log_freq == 0:

                        train_acc, train_loss, train_map = test(
                            model, train_test_loader, modality)

                        logger.scalar_summary('Train Accuracy', train_acc,
                                              update_step_idx)

                        logger.scalar_summary('Train map', train_map,
                                              update_step_idx)

                        for k in train_loss.keys():
                            logger.scalar_summary('Train Loss {}'.format(k),
                                                  train_loss[k],
                                                  update_step_idx)

                        if args.test_log:

                            test_acc, test_loss, test_map = test(
                                model, test_test_loader, modality)

                            logger.scalar_summary('Test Accuracy', test_acc,
                                                  update_step_idx)

                            logger.scalar_summary('Test map', test_map,
                                                  update_step_idx)

                            for k in test_loss.keys():
                                logger.scalar_summary('Test Loss {}'.format(k),
                                                      test_loss[k],
                                                      update_step_idx)

                    # Batch Update
                    update_step_idx += 1

                    for k, v in loss_recorder.items():

                        print('Step {}: Loss_{}-{}'.format(
                            update_step_idx, k, v / batch_size))

                        logger.scalar_summary('Loss_{}_ps'.format(k),
                                              v / batch_size, update_step_idx)

                        loss_recorder[k] = 0

                    optimizer.step()
                    optimizer.zero_grad()

                    if learning_rate_decay:
                        scheduler.step()

                    if update_step_idx in check_points:
                        torch.save(
                            model.state_dict(),
                            os.path.join(
                                save_dir,
                                'model-{}-{}'.format(modality,
                                                     update_step_idx)))

                    if update_step_idx >= max_step_num:
                        break

    # 创建训练集数据信息字典
     """
    dataset_dict example:
        {video_validation_0000266:{
              "duration": 171.57,
              'frame_rate':30,
              'labels':[0,8],
              'annotations':{
                  0:[[72.8,76.4]],
                  8:[[9.6,12.2],[12.4,21.8],[22.0,29.2],....[137.9,148.2]]
              }
              'frame_cnt': 5143,
              'rgb_feature':(40,320,1024),
              'flow_feature':(40,320,1024)
        },
        .......
        video_validation_0000266_bg:{
              "duration": 0.63333,
              'frame_rate':30,
              'labels':[20],
              'annotations':{20:[]}
              'frame_cnt': 19,
              'rgb_feature':(40,19,1024),
              'flow_feature':(40,19,1024)
        }
        
      }
    
    """
    train_dataset_dict = get_dataset(dataset_name=dataset_name,
                                     subset=args.train_subset_name,
                                     file_paths=file_paths,
                                     sample_rate=sample_rate,
                                     base_sample_rate=base_sample_rate,
                                     action_class_num=action_class_num,
                                     modality='both',
                                     feature_type=feature_type,
                                     feature_oversample=feature_oversample,
                                     temporal_aug=True,
                                     load_background=with_bg)

    # 一个视频可能包含多个label，将data_dict中的多个label拆分开
    """
    dataset_dict example:
        {video_validation_0000266-0:{
              "duration": 171.57,
              'frame_rate':30,
              'labels':[0,8],
              'annotations':{[[72.8,76.4]]
              }
              'frame_cnt': 5143,
              'rgb_feature':(40,320,1024),
              'flow_feature':(40,320,1024)，
              'label_single': 0,
              'weight':0.5,
              'old_key':'video_validation_0000266'
        },
        video_validation_0000266-8:{
              "duration": 171.57,
              'frame_rate':30,
              'labels':[0,8],
              'annotations':{[[9.6,12.2],[12.4,21.8],[22.0,29.2],....[137.9,148.2]]
              }
              'frame_cnt': 5143,
              'rgb_feature':(40,320,1024),
              'flow_feature':(40,320,1024)，
              'label_single': 8,
              'weight':0.5,
              'old_key':'video_validation_0000266'
        },
        ........
      }
    
    """
    train_train_dataset = SingleVideoDataset(
        train_dataset_dict,
        single_label=True,
        random_select=True,
        max_len=training_max_len)  # To be checked

    train_test_dataset = SingleVideoDataset(train_dataset_dict,
                                            single_label=True,
                                            random_select=False,
                                            max_len=None)

    # 注意这里的batch_size = 1是因为他用了数据增强
    train_train_loader = torch.utils.data.DataLoader(train_train_dataset,
                                                     batch_size=1,
                                                     pin_memory=True,
                                                     shuffle=True)

    train_test_loader = torch.utils.data.DataLoader(train_test_dataset,
                                                    batch_size=1,
                                                    pin_memory=True,
                                                    shuffle=False)

    if args.test_log:
    """
    dataset_dict example:
        {video_test_0000324:{
              "duration": 171.57,
              'frame_rate':30,
              'labels':[0],
              'annotations':{0:[[49.2,53.5],[116.7,122.5]]}
              'frame_cnt': 4469,
              'rgb_feature':(40,278,1024),
              'flow_feature':(40,278,1024)，
        }, 
      }  
    """
        test_dataset_dict = get_dataset(dataset_name=dataset_name,
                                        subset=args.test_subset_name,
                                        file_paths=file_paths,
                                        sample_rate=sample_rate,
                                        base_sample_rate=base_sample_rate,
                                        action_class_num=action_class_num,
                                        modality='both',
                                        feature_type=feature_type,
                                        feature_oversample=feature_oversample,
                                        temporal_aug=True,
                                        load_background=False)

        test_test_dataset = SingleVideoDataset(test_dataset_dict,
                                               single_label=True,
                                               random_select=False,
                                               max_len=None)

        test_test_loader = torch.utils.data.DataLoader(test_test_dataset,
                                                       batch_size=1,
                                                       pin_memory=True,
                                                       shuffle=False)
    else:

        test_test_loader = None

    for run_idx in range(train_run_num):  # 3

        naming = '{}-run-{}'.format(experiment_naming, run_idx)

        train(train_train_loader, train_test_loader, test_test_loader, 'rgb',
              naming)
        train(train_train_loader, train_test_loader, test_test_loader, 'flow',
              naming)
        train(train_train_loader, train_test_loader, test_test_loader, 'both',
              naming)
