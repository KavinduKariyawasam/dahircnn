import torch
# from xBD_code.zoo import model_transformer_encoding
# importlib.reload(model_transformer_encoding)
from model_transformer_encoding import BASE_Transformer_UNet, Custom_Transformer_UNet
# from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork

import torch
import torch.nn as nn
# from torchvision.models.detection.backbone_utils import BackboneWithFPN
# from torchvision.models.detection.fpn import FeaturePyramidNetwork
# from torchvision.ops import misc as misc_nn_ops
import numpy as np

from collections import OrderedDict
from torch import nn, Tensor
import warnings
from typing import Tuple, List, Dict, Optional, Union
from torchvision.models.detection.image_list import ImageList

import torch.nn.functional as F

from torchvision.ops import MultiScaleRoIAlign
import gc

# from torchvision.models.detection._utils import overwrite_eps
# from torchvision._internally_replaced_utils import load_state_dict_from_url

from torchvision.models.detection.anchor_utils import AnchorGenerator
# from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform
# from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, _validate_trainable_layers, mobilenet_backbone



class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Args:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, rpn, roi_heads, transform):
        super(GeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        # print(f"=>> GRCNN forward pass <<=")
        
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            # print(f"Lenght of images: {len(images)}, Length of targets: {len(targets)}")
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                             boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        # breakpoint()
        # images = images[0]
        # print(f"type(targets): {type(targets)}")
        # if self.training:
        #     targets = [targets[0], targets[0]]
        #     breakpoint()
            
        # print(f"Shape of images: {images[0].shape}")
        # img_post = self.transform([images[0][:3]], targets)
        # img_pre = self.transform([images[0][3:]], targets)
        images, targets = self.transform(images, targets)
        
        # print(type(images))
        # print(type(targets))

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))

        features = self.backbone(images.tensors)
        # print(f"Features: {features.keys()}")
        # print('features:', features.shape)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        
        # take only the first image and targets
        # targets = [targets[0]] if targets is not None else None
        
        # print(f"image size: {images.tensors[0].shape}")
        # images = ImageList(images.tensors.unsqueeze(0), [images.image_sizes])
        
        gc.collect()
        torch.cuda.empty_cache()
        
        proposals, proposal_losses = self.rpn(images, features, targets)
        # for i, proposal in enumerate(proposals):
        #     print(f"Number of proposals for image {i}: {proposal.size(0)}")
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        
        del features, proposals, targets
        gc.collect()
        torch.cuda.empty_cache()
        
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)


# __all__ = [
#     "FasterRCNN", "fasterrcnn_resnet50_fpn", "fasterrcnn_mobilenet_v3_large_320_fpn",
#     "fasterrcnn_mobilenet_v3_large_fpn"
# ]


class FasterRCNN(GeneralizedRCNN):

    def __init__(self, backbone, num_classes=None,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 rpn_score_thresh=0.0,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None):

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")

        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            anchor_sizes = (
                (32,),  # Matches FPN Level 3 (64x64)
                (64,),  # Matches FPN Level 2 (128x128)
                (128,), # Matches FPN Level 1 (256x256)
                (256,), # Matches FPN Level 0 (512x512)
            )

            # Define aspect ratios for each anchor
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

            # Create the AnchorGenerator
            rpn_anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
            # anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            # aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            # rpn_anchor_generator = AnchorGenerator(
            #     anchor_sizes, aspect_ratios
            # )
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
            score_thresh=rpn_score_thresh)

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=7,
                sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes)

        roi_heads = RoIHeads(
            # Box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406, 0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225, 0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        super(FasterRCNN, self).__init__(backbone, rpn, roi_heads, transform)


class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        
        # print(f"=>> TwoMLPHead forward pass <<=")
        
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        
        # print(f"=>> FastRCNNPredictor forward pass <<=")
        
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas



# from torchvision.models.detection._utils import overwrite_eps
# from torchvision._internally_replaced_utils import load_state_dict_from_url

# from torchvision.models.detection.faster_rcnn import FasterRCNN
# from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, _validate_trainable_layers

# __all__ = [
#     "MaskRCNN", "maskrcnn_resnet50_fpn",
# ]


class CUSTOM_MaskRCNN(FasterRCNN):
    
    def __init__(self, backbone, num_classes=None,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 rpn_score_thresh=0.0,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None,
                 # Mask parameters
                 mask_roi_pool=None, mask_head=None, mask_predictor=None):

        assert isinstance(mask_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if mask_predictor is not None:
                raise ValueError("num_classes should be None when mask_predictor is specified")

        out_channels = backbone.out_channels

        if mask_roi_pool is None:
            mask_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=14,
                sampling_ratio=2)

        if mask_head is None:
            mask_layers = (256, 256, 256, 256)
            mask_dilation = 1
            mask_head = MaskRCNNHeads(out_channels, mask_layers, mask_dilation)

        if mask_predictor is None:
            mask_predictor_in_channels = 256  # == mask_layers[-1]
            mask_dim_reduced = 256
            mask_predictor = MaskRCNNPredictor(mask_predictor_in_channels,
                                               mask_dim_reduced, num_classes)

        super(CUSTOM_MaskRCNN, self).__init__(
            backbone, num_classes,
            # transform parameters
            min_size, max_size,
            image_mean, image_std,
            # RPN-specific parameters
            rpn_anchor_generator, rpn_head,
            rpn_pre_nms_top_n_train, rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_train, rpn_post_nms_top_n_test,
            rpn_nms_thresh,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_score_thresh,
            # Box parameters
            box_roi_pool, box_head, box_predictor,
            box_score_thresh, box_nms_thresh, box_detections_per_img,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights)

        self.roi_heads.mask_roi_pool = mask_roi_pool
        self.roi_heads.mask_head = mask_head
        self.roi_heads.mask_predictor = mask_predictor


class MaskRCNNHeads(nn.Sequential):
    def __init__(self, in_channels, layers, dilation):
        """
        Args:
            in_channels (int): number of input channels
            layers (list): feature dimensions of each FCN layer
            dilation (int): dilation rate of kernel
        """
        d = OrderedDict()
        next_feature = in_channels
        for layer_idx, layer_features in enumerate(layers, 1):
            d["mask_fcn{}".format(layer_idx)] = nn.Conv2d(
                next_feature, layer_features, kernel_size=3,
                stride=1, padding=dilation, dilation=dilation)
            d["relu{}".format(layer_idx)] = nn.ReLU(inplace=True)
            next_feature = layer_features

        super(MaskRCNNHeads, self).__init__(d)
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
            # elif "bias" in name:
            #     nn.init.constant_(param, 0)


class MaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, dim_reduced, num_classes):
        super(MaskRCNNPredictor, self).__init__(OrderedDict([
            ("conv5_mask", nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)),
            ("relu", nn.ReLU(inplace=True)),
            ("mask_fcn_logits", nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)),
        ]))

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
            # elif "bias" in name:
            #     nn.init.constant_(param, 0)



class CustomMaskRCNNBackbone(nn.Module):
    def __init__(self, custom_backbone):
        super(CustomMaskRCNNBackbone, self).__init__()
        self.backbone = custom_backbone
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[32, 32, 32, 32],  # Channels from out_2, out_3, out_4, out_5
            out_channels=256  # Output channels for the FPN
        )
        self.out_channels = 256

    def forward(self, x):
        
        # print(f"=>> CustomMaskRCNNBackbone forward pass <<=")
        
        backbone_outputs = self.backbone(x)
        out_2, out_3, out_4, out_5 = (
            backbone_outputs["0"],
            backbone_outputs["1"],
            backbone_outputs["2"],
            backbone_outputs["3"],
        )
        fpn_inputs = {
            "0": out_2,
            "1": out_3,
            "2": out_4,
            "3": out_5,
        }
        fpn_outputs = self.fpn(fpn_inputs)
        # for level, feature_map in enumerate(fpn_outputs.values()):
        #     print(f"FPN Level {level} Feature Map Shape: {feature_map.shape}")
        return fpn_outputs



if __name__ == "__main__":
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    backbone = Custom_Transformer_UNet(input_nc=3, output_nc=5, token_len=4, resnet_stages_num=4,
                                with_pos='learned', with_decoder_pos='learned', enc_depth=1, dec_depth=8).to(device)


    # Define input channels of feature maps from the custom backbone
    fpn_in_channels = [64, 128, 256, 512]  # Replace with actual channels from your encoder
    fpn_out_channels = 32  # Number of output channels from FPN

    # Initialize the custom backbone
    custom_backbone = CustomMaskRCNNBackbone(backbone)

    img1 = np.random.rand(1024, 1024, 3)
    img2 = np.random.rand(1024, 1024, 3)

    img1_tensor = torch.tensor(img1.transpose((2, 0, 1))).float()
    img2_tensor = torch.tensor(img2.transpose((2, 0, 1))).float()

    imgs = np.concatenate([img1, img2], axis=2)

    imgs = torch.tensor(imgs.transpose((2, 0, 1))).float()

    print(imgs.shape)

    # Instantiate Mask R-CNN with the custom anchor generator
    model = CUSTOM_MaskRCNN(
        backbone=custom_backbone,
        num_classes=5,  # Adjust based on your dataset
        # rpn_anchor_generator=anchor_generator
    )

    model.transform.min_size = (1024,)
    model.eval().cuda()
    # print the model architecture
    # output = model([img1_tensor.to(device), img2_tensor.to(device)])
    print(f"image shapes: {imgs.shape}")
    output = model([imgs.to(device), imgs.to(device)])

    # print(output)
