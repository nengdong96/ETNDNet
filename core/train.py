
from tools import MultiItemAverageMeter
from network.feature_map_processing import FeatureMapErasing, FeatureMapTransforming, FeatureMapNoising

def train(base, loaders, config):

    base.set_train()
    loader = loaders.loader
    meter = MultiItemAverageMeter()
    for i, data in enumerate(loader):
        imgs, pids, cids = data
        imgs, pids, cids = imgs.to(base.device), pids.to(base.device).long(), \
                           cids.to(base.device).long()
        if config.module == 'B':
            features_map = base.model(imgs)
            cls_score = base.classifier(features_map)

            ide_loss = base.pid_creiteron(cls_score, pids)

            total_loss = ide_loss

            base.model_optimizer.zero_grad()
            base.classifier_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()
            base.classifier_optimizer.step()

            meter.update({'pid_loss': ide_loss.data,
            })

        elif config.module == 'ED':

            features_map = base.model(imgs)
            erasing_features_map = FeatureMapErasing(config).__call__(features_map)

            cls_score = base.classifier(features_map)

            erasing_cls_score = base.classifier(erasing_features_map)

            ide_loss = base.pid_creiteron(cls_score, pids)

            erasing_ide_loss = base.pid_creiteron(erasing_cls_score, pids)

            total_loss = ide_loss - config.lambda1 * erasing_ide_loss

            base.classifier_optimizer.zero_grad()
            total_loss.backward()
            base.classifier_optimizer.step()

            features_map = base.model(imgs)
            erasing_features_map = FeatureMapErasing(config).__call__(features_map)

            cls_score = base.classifier(features_map)

            erasing_cls_score = base.classifier(erasing_features_map)

            ide_loss = base.pid_creiteron(cls_score, pids)

            erasing_ide_loss = base.pid_creiteron(erasing_cls_score, pids)

            total_loss = ide_loss + config.lambda1 * erasing_ide_loss

            base.model_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()

            meter.update({'pid_loss': ide_loss.data,
                          'erasing_pid_loss': erasing_ide_loss.data,
                          })

        elif config.module == 'TD':

            features_map = base.model(imgs)
            transforming_features_map = FeatureMapTransforming(config).__call__(features_map)

            cls_score = base.classifier(features_map)

            transforming_cls_score = base.classifier(transforming_features_map)

            ide_loss = base.pid_creiteron(cls_score, pids)

            transforming_ide_loss = base.pid_creiteron(transforming_cls_score, pids)

            total_loss = ide_loss - \
                         config.lambda2 * transforming_ide_loss

            base.classifier_optimizer.zero_grad()
            total_loss.backward()
            base.classifier_optimizer.step()

            features_map = base.model(imgs)
            transforming_features_map = FeatureMapTransforming(config).__call__(features_map)

            cls_score = base.classifier(features_map)

            transforming_cls_score = base.classifier(transforming_features_map)

            ide_loss = base.pid_creiteron(cls_score, pids)

            transforming_ide_loss = base.pid_creiteron(transforming_cls_score, pids)

            total_loss = ide_loss + \
                         config.lambda2 * transforming_ide_loss

            base.model_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()

            meter.update({'pid_loss': ide_loss.data,
                          'transforming_pid_loss': transforming_ide_loss.data,
                          })

        elif config.module == 'ND':

            features_map = base.model(imgs)
            noising_features_map = FeatureMapNoising(config).__call__(features_map)

            cls_score = base.classifier(features_map)

            noising_cls_score = base.classifier(noising_features_map)

            ide_loss = base.pid_creiteron(cls_score, pids)

            noising_ide_loss = base.pid_creiteron(noising_cls_score, pids)

            total_loss = ide_loss - \
                         config.lambda3 * noising_ide_loss

            base.classifier_optimizer.zero_grad()
            total_loss.backward()
            base.classifier_optimizer.step()

            features_map = base.model(imgs)
            noising_features_map = FeatureMapNoising(config).__call__(features_map)

            cls_score = base.classifier(features_map)

            noising_cls_score = base.classifier(noising_features_map)

            ide_loss = base.pid_creiteron(cls_score, pids)

            noising_ide_loss = base.pid_creiteron(noising_cls_score, pids)

            total_loss = ide_loss + \
                         config.lambda3 * noising_ide_loss

            base.model_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()

            meter.update({'pid_loss': ide_loss.data,
                          'noising_pid_loss': noising_ide_loss.data,
                          })

        elif config.module == 'ETD':

            features_map = base.model(imgs)
            erasing_features_map = FeatureMapErasing(config).__call__(features_map)
            transforming_features_map = FeatureMapTransforming(config).__call__(features_map)

            cls_score = base.classifier(features_map)

            erasing_cls_score = base.classifier(erasing_features_map)
            transforming_cls_score = base.classifier(transforming_features_map)

            ide_loss = base.pid_creiteron(cls_score, pids)

            erasing_ide_loss = base.pid_creiteron(erasing_cls_score, pids)
            transforming_ide_loss = base.pid_creiteron(transforming_cls_score, pids)

            total_loss = ide_loss - \
                         config.lambda1 * erasing_ide_loss - \
                         config.lambda2 * transforming_ide_loss

            base.classifier_optimizer.zero_grad()
            total_loss.backward()
            base.classifier_optimizer.step()

            features_map = base.model(imgs)
            erasing_features_map = FeatureMapErasing(config).__call__(features_map)
            transforming_features_map = FeatureMapTransforming(config).__call__(features_map)

            cls_score = base.classifier(features_map)

            erasing_cls_score = base.classifier(erasing_features_map)
            transforming_cls_score = base.classifier(transforming_features_map)

            ide_loss = base.pid_creiteron(cls_score, pids)

            erasing_ide_loss = base.pid_creiteron(erasing_cls_score, pids)
            transforming_ide_loss = base.pid_creiteron(transforming_cls_score, pids)

            total_loss = ide_loss + \
                         config.lambda1 * erasing_ide_loss + \
                         config.lambda2 * transforming_ide_loss

            base.model_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()

            meter.update({'pid_loss': ide_loss.data,
                          'erasing_pid_loss': erasing_ide_loss.data,
                          'transforming_pid_loss': transforming_ide_loss.data,
                          })

        elif config.module == 'END':

            features_map = base.model(imgs)
            erasing_features_map = FeatureMapErasing(config).__call__(features_map)
            noising_features_map = FeatureMapNoising(config).__call__(features_map)

            cls_score = base.classifier(features_map)

            erasing_cls_score = base.classifier(erasing_features_map)
            noising_cls_score = base.classifier(noising_features_map)

            ide_loss = base.pid_creiteron(cls_score, pids)

            erasing_ide_loss = base.pid_creiteron(erasing_cls_score, pids)
            noising_ide_loss = base.pid_creiteron(noising_cls_score, pids)

            total_loss = ide_loss - \
                         config.lambda1 * erasing_ide_loss - \
                         config.lambda3 * noising_ide_loss

            base.classifier_optimizer.zero_grad()
            total_loss.backward()
            base.classifier_optimizer.step()

            features_map = base.model(imgs)
            erasing_features_map = FeatureMapErasing(config).__call__(features_map)
            noising_features_map = FeatureMapNoising(config).__call__(features_map)

            cls_score = base.classifier(features_map)

            erasing_cls_score = base.classifier(erasing_features_map)
            noising_cls_score = base.classifier(noising_features_map)

            ide_loss = base.pid_creiteron(cls_score, pids)

            erasing_ide_loss = base.pid_creiteron(erasing_cls_score, pids)
            noising_ide_loss = base.pid_creiteron(noising_cls_score, pids)

            total_loss = ide_loss + \
                         config.lambda1 * erasing_ide_loss + \
                         config.lambda3 * noising_ide_loss

            base.model_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()

            meter.update({'pid_loss': ide_loss.data,
                          'erasing_pid_loss': erasing_ide_loss.data,
                          'noising_pid_loss': noising_ide_loss.data,
                          })

        elif config.module == 'TND':

            features_map = base.model(imgs)
            transforming_features_map = FeatureMapTransforming(config).__call__(features_map)
            noising_features_map = FeatureMapNoising(config).__call__(features_map)

            cls_score = base.classifier(features_map)

            transforming_cls_score = base.classifier(transforming_features_map)
            noising_cls_score = base.classifier(noising_features_map)

            ide_loss = base.pid_creiteron(cls_score, pids)

            transforming_ide_loss = base.pid_creiteron(transforming_cls_score, pids)
            noising_ide_loss = base.pid_creiteron(noising_cls_score, pids)

            total_loss = ide_loss - \
                         config.lambda2 * transforming_ide_loss - \
                         config.lambda3 * noising_ide_loss

            base.classifier_optimizer.zero_grad()
            total_loss.backward()
            base.classifier_optimizer.step()

            features_map = base.model(imgs)
            transforming_features_map = FeatureMapTransforming(config).__call__(features_map)
            noising_features_map = FeatureMapNoising(config).__call__(features_map)

            cls_score = base.classifier(features_map)

            transforming_cls_score = base.classifier(transforming_features_map)
            noising_cls_score = base.classifier(noising_features_map)

            ide_loss = base.pid_creiteron(cls_score, pids)

            transforming_ide_loss = base.pid_creiteron(transforming_cls_score, pids)
            noising_ide_loss = base.pid_creiteron(noising_cls_score, pids)

            total_loss = ide_loss + \
                         config.lambda2 * transforming_ide_loss + \
                         config.lambda3 * noising_ide_loss

            base.model_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()

            meter.update({'pid_loss': ide_loss.data,
                          'transforming_pid_loss': transforming_ide_loss.data,
                          'noising_pid_loss': noising_ide_loss.data,
                          })

        elif config.module == 'ETND':

            features_map = base.model(imgs)
            erasing_features_map = FeatureMapErasing(config).__call__(features_map)
            transforming_features_map = FeatureMapTransforming(config).__call__(features_map)
            noising_features_map = FeatureMapNoising(config).__call__(features_map)

            cls_score = base.classifier(features_map)

            erasing_cls_score = base.classifier(erasing_features_map)
            transforming_cls_score = base.classifier(transforming_features_map)
            noising_cls_score = base.classifier(noising_features_map)

            ide_loss = base.pid_creiteron(cls_score, pids)

            erasing_ide_loss = base.pid_creiteron(erasing_cls_score, pids)
            transforming_ide_loss = base.pid_creiteron(transforming_cls_score, pids)
            noising_ide_loss = base.pid_creiteron(noising_cls_score, pids)

            total_loss = ide_loss - \
                         config.lambda1 * erasing_ide_loss - \
                         config.lambda2 * transforming_ide_loss - \
                         config.lambda3 * noising_ide_loss

            base.classifier_optimizer.zero_grad()
            total_loss.backward()
            base.classifier_optimizer.step()

            features_map = base.model(imgs)
            erasing_features_map = FeatureMapErasing(config).__call__(features_map)
            transforming_features_map = FeatureMapTransforming(config).__call__(features_map)
            noising_features_map = FeatureMapNoising(config).__call__(features_map)

            cls_score = base.classifier(features_map)

            erasing_cls_score = base.classifier(erasing_features_map)
            transforming_cls_score = base.classifier(transforming_features_map)
            noising_cls_score = base.classifier(noising_features_map)

            ide_loss = base.pid_creiteron(cls_score, pids)

            erasing_ide_loss = base.pid_creiteron(erasing_cls_score, pids)
            transforming_ide_loss = base.pid_creiteron(transforming_cls_score, pids)
            noising_ide_loss = base.pid_creiteron(noising_cls_score, pids)

            total_loss = ide_loss + \
                         config.lambda1 * erasing_ide_loss + \
                         config.lambda2 * transforming_ide_loss + \
                         config.lambda3 * noising_ide_loss

            base.model_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()

            meter.update({'pid_loss': ide_loss.data,
                          'erasing_pid_loss': erasing_ide_loss.data,
                          'transforming_pid_loss': transforming_ide_loss.data,
                          'noising_pid_loss': noising_ide_loss.data,
                          })

    return meter.get_val(), meter.get_str()







