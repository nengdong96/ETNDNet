
import torch

from tools import CatMeter, time_now, ReIDEvaluator

def test(config, base, loader):

    base.set_eval()

    query_features_meter, query_pids_meter, query_cids_meter = CatMeter(), CatMeter(), CatMeter()
    gallery_features_meter, gallery_pids_meter, gallery_cids_meter = CatMeter(), CatMeter(), CatMeter()

    loaders = [loader.query_loader, loader.gallery_loader]

    print(time_now(), 'features start')

    with torch.no_grad():
        for loader_id, loader in enumerate(loaders):
            for data in loader:
                images, pids, cids = data
                images = images.to(base.device)
                features_map = base.model(images)
                bn_features = base.classifier(features_map)

                if loader_id == 0:
                    query_features_meter.update(bn_features.data)
                    query_pids_meter.update(pids)
                    query_cids_meter.update(cids)
                elif loader_id == 1:
                    gallery_features_meter.update(bn_features.data)
                    gallery_pids_meter.update(pids)
                    gallery_cids_meter.update(cids)

    print(time_now(), 'features done')

    query_features = query_features_meter.get_val_numpy()
    gallery_features = gallery_features_meter.get_val_numpy()


    mAP, CMC = ReIDEvaluator(dist='cosine', mode=config.test_mode).evaluate(
        query_features, query_pids_meter.get_val_numpy(), query_cids_meter.get_val_numpy(),
        gallery_features, gallery_pids_meter.get_val_numpy(), gallery_cids_meter.get_val_numpy())

    return mAP, CMC[0: 20]





