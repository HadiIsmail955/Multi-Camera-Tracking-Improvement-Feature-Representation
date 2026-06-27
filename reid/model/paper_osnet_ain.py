import torch
import torch.nn as nn
import torch.nn.functional as F


class PaperOSNetAINEmbedding(nn.Module):
    def __init__(
        self,
        model_name: str = "osnet_ain_x1_0",
        num_classes: int = 1000,
        pretrained: bool = True,
        checkpoint: str | None = None,
    ):
        super().__init__()

        try:
            import torchreid
        except ImportError as exc:
            raise ImportError(
                "torchreid is not importable. Make sure the paper repo version is installed:\n"
                "cd third_party/Glance-MCMT/deep-person-reid\n"
                "python -m pip install --no-build-isolation -e ."
            ) from exc

        print("=" * 80)
        print("BUILDING PAPER REID MODEL")
        print("=" * 80)
        print("torchreid path:", torchreid.__file__)
        print("model_name:", model_name)
        print("pretrained:", pretrained)
        print("checkpoint:", checkpoint)
        print("=" * 80)

        self.backbone = torchreid.models.build_model(
            name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
        )

        if checkpoint is not None and str(checkpoint).strip() != "":
            from collections import OrderedDict

            print("Loading paper ReID checkpoint:", checkpoint)

            ckpt = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=False,
            )

            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            elif isinstance(ckpt, dict) and "model" in ckpt:
                state_dict = ckpt["model"]
            else:
                state_dict = ckpt

            model_state = self.backbone.state_dict()
            clean_state = OrderedDict()

            loaded = 0
            skipped = 0

            for k, v in state_dict.items():
                key = k

                if key.startswith("module."):
                    key = key[len("module."):]

                if key in model_state and model_state[key].shape == v.shape:
                    clean_state[key] = v
                    loaded += 1
                else:
                    skipped += 1

            missing, unexpected = self.backbone.load_state_dict(clean_state, strict=False)

            print("Paper checkpoint loaded.")
            print("Matched keys:", loaded)
            print("Skipped keys:", skipped)
            print("Missing keys:", len(missing))
            print("Unexpected keys:", len(unexpected))

    def forward(self, images):
        features = self.backbone(images)

        if isinstance(features, dict):
            if "bn_embedding" in features:
                emb = features["bn_embedding"]
            elif "embedding" in features:
                emb = features["embedding"]
            elif "features" in features:
                emb = features["features"]
            else:
                raise KeyError(f"No embedding key found in model output: {features.keys()}")

        elif isinstance(features, (tuple, list)):
            emb = features[0]

        else:
            emb = features

        emb = emb.view(emb.size(0), -1)
        emb = F.normalize(emb.float(), p=2, dim=1)

        return {
            "embedding": emb,
            "bn_embedding": emb,
        }