"""Inference-only CRACKTO-SFNet V4 architecture, extracted from the provided training runtime.

Training loops and Kaggle-specific dataset setup are intentionally excluded.
"""
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k=3, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = k // 2

        self.block = nn.Sequential(
            nn.Conv2d(
                cin, cout, k,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class DirectionalFeatureBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()

        self.h = nn.Sequential(
            nn.Conv2d(
                cin, cout,
                kernel_size=(1, 7),
                padding=(0, 3),
                bias=False,
            ),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

        self.v = nn.Sequential(
            nn.Conv2d(
                cin, cout,
                kernel_size=(7, 1),
                padding=(3, 0),
                bias=False,
            ),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

        self.fuse = ConvBNAct(cout * 2, cout, 3)

    def forward(self, x):
        h = self.h(x)
        v = self.v(x)
        return self.fuse(torch.cat([h, v], dim=1))

class HighResolutionCrackBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.local = ConvBNAct(cin, cout, 3)
        self.directional = DirectionalFeatureBlock(cout, cout)
        self.refine = ConvBNAct(cout, cout, 3)

    def forward(self, x):
        return self.refine(self.directional(self.local(x)))

class HighResolutionCrackBranch(nn.Module):
    def __init__(self, in_channels=3, channels=(32, 48, 64)):
        super().__init__()

        c4, c8, c16 = channels

        self.stage4 = nn.Sequential(
            HighResolutionCrackBlock(in_channels, c4),
            HighResolutionCrackBlock(c4, c4),
        )

        self.stage8 = nn.Sequential(
            ConvBNAct(c4, c8, 3, stride=2),
            HighResolutionCrackBlock(c8, c8),
        )

        self.stage16 = nn.Sequential(
            ConvBNAct(c8, c16, 3, stride=2),
            HighResolutionCrackBlock(c16, c16),
        )

    def forward(self, x):
        f4 = self.stage4(x)
        f8 = self.stage8(f4)
        f16 = self.stage16(f8)
        return {"f4": f4, "f8": f8, "f16": f16}

class GlobalContextBranch(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Conv2d(cin, cout, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        y = self.proj(self.pool(x))
        return F.interpolate(
            y,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

class MultiScaleContextEncoder(nn.Module):
    def __init__(self, in_channels=64, out_channels=128):
        super().__init__()

        b = 32

        self.local = ConvBNAct(in_channels, b, 3)

        self.d2 = nn.Sequential(
            nn.Conv2d(
                in_channels, b, 3,
                padding=2,
                dilation=2,
                bias=False,
            ),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
        )

        self.d4 = nn.Sequential(
            nn.Conv2d(
                in_channels, b, 3,
                padding=4,
                dilation=4,
                bias=False,
            ),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
        )

        self.global_branch = GlobalContextBranch(
            in_channels, b
        )

        self.proj = nn.Sequential(
            nn.Conv2d(b * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        a = self.local(x)
        b = self.d2(x)
        c = self.d4(x)
        d = self.global_branch(x)
        return self.proj(torch.cat([a, b, c, d], dim=1))

class AdaptiveScaleFusion(nn.Module):
    def __init__(
        self,
        high_channels=32,
        mid_channels=48,
        low_channels=64,
        context_channels=128,
        fusion_channels=64,
    ):
        super().__init__()

        self.high = ConvBNAct(high_channels, fusion_channels, 1)
        self.mid = ConvBNAct(mid_channels, fusion_channels, 1)
        self.low = ConvBNAct(low_channels, fusion_channels, 1)
        self.context = ConvBNAct(context_channels, fusion_channels, 1)

        self.weights = nn.Sequential(
            nn.Conv2d(fusion_channels * 4, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 4, 1),
        )

        self.out = ConvBNAct(fusion_channels, fusion_channels, 3)

    def forward(self, f4, f8, f16, context):
        size = f4.shape[-2:]

        h = self.high(f4)
        m = self.mid(
            F.interpolate(
                f8,
                size=size,
                mode="bilinear",
                align_corners=False,
            )
        )
        l = self.low(
            F.interpolate(
                f16,
                size=size,
                mode="bilinear",
                align_corners=False,
            )
        )
        c = self.context(
            F.interpolate(
                context,
                size=size,
                mode="bilinear",
                align_corners=False,
            )
        )

        combined = torch.cat([h, m, l, c], dim=1)
        weights = torch.softmax(
            self.weights(combined),
            dim=1,
        )

        fused = (
            h * weights[:, 0:1]
            + m * weights[:, 1:2]
            + l * weights[:, 2:3]
            + c * weights[:, 3:4]
        )

        return self.out(fused), weights

class GeometryRefinementBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.direction = DirectionalFeatureBlock(
            channels, channels
        )
        self.fuse = ConvBNAct(channels * 2, channels, 3)

    def forward(self, x):
        d = self.direction(x)
        return self.fuse(torch.cat([x, d], dim=1))

class PredictionHead(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(cin, cin, 3),
            nn.Conv2d(cin, cout, 1),
        )

    def forward(self, x):
        return self.block(x)

class GeometryAwarePredictionHeads(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()

        self.refinement = GeometryRefinementBlock(
            in_channels
        )

        self.mask = PredictionHead(in_channels, 1)
        self.boundary = PredictionHead(in_channels, 1)
        self.centerline = PredictionHead(in_channels, 1)
        self.uncertainty = PredictionHead(in_channels, 1)
        self.direction = PredictionHead(in_channels, 2)

    def forward(self, x):
        f = self.refinement(x)

        mask_logits = self.mask(f)
        boundary_logits = self.boundary(f)
        centerline_logits = self.centerline(f)
        uncertainty_logits = self.uncertainty(f)
        direction = self.direction(f)

        return {
            "features": f,
            "mask_logits": mask_logits,
            "mask": torch.sigmoid(mask_logits),
            "boundary_logits": boundary_logits,
            "boundary": torch.sigmoid(boundary_logits),
            "centerline_logits": centerline_logits,
            "centerline": torch.sigmoid(centerline_logits),
            "uncertainty_logits": uncertainty_logits,
            "uncertainty": torch.sigmoid(uncertainty_logits),
            "direction": direction,
        }

class GeometryGuidanceGenerator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.boundary = nn.Conv2d(channels, 1, 1)
        self.centerline = nn.Conv2d(channels, 1, 1)
        self.uncertainty = nn.Conv2d(channels, 1, 1)

    def forward(self, x):
        return {
            "boundary": torch.sigmoid(self.boundary(x)),
            "centerline": torch.sigmoid(self.centerline(x)),
            "uncertainty": torch.sigmoid(self.uncertainty(x)),
        }

class GeometryGuidedDecoder(nn.Module):
    def __init__(
        self,
        f4_channels=32,
        f8_channels=48,
        f16_channels=64,
        context_channels=128,
        decoder_channels=64,
    ):
        super().__init__()

        self.p16 = ConvBNAct(
            f16_channels, decoder_channels, 1
        )
        self.pc = ConvBNAct(
            context_channels, decoder_channels, 1
        )
        self.p8 = ConvBNAct(
            f8_channels, decoder_channels, 1
        )
        self.p4 = ConvBNAct(
            f4_channels, decoder_channels, 1
        )

        self.refine = nn.Sequential(
            ConvBNAct(decoder_channels, decoder_channels, 3),
            DirectionalFeatureBlock(
                decoder_channels, decoder_channels
            ),
        )

        self.guidance = GeometryGuidanceGenerator(
            decoder_channels
        )

        self.modulation = nn.Sequential(
            nn.Conv2d(
                decoder_channels + 3,
                decoder_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
        )

        self.final = nn.Sequential(
            ConvBNAct(decoder_channels, decoder_channels, 3),
            DirectionalFeatureBlock(
                decoder_channels, decoder_channels
            ),
        )

    def forward(self, f4, f8, f16, context):
        x = self.p16(f16)
        x = x + self.pc(context)

        x = F.interpolate(
            x,
            size=f8.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = x + self.p8(f8)

        x = F.interpolate(
            x,
            size=f4.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = x + self.p4(f4)

        x = self.refine(x)

        guidance = self.guidance(x)

        geo = torch.cat(
            [
                guidance["boundary"],
                guidance["centerline"],
                guidance["uncertainty"],
            ],
            dim=1,
        )

        x = self.modulation(
            torch.cat([x, geo], dim=1)
        )

        x = self.final(x)

        return {
            "features": x,
            "guidance": guidance,
        }

class CrackTopoSFNetV2(nn.Module):
    def __init__(self):
        super().__init__()

        self.high_resolution = HighResolutionCrackBranch(
            3, (32, 48, 64)
        )

        self.context_encoder = MultiScaleContextEncoder(
            64, 128
        )

        self.scale_fusion = AdaptiveScaleFusion(
            32, 48, 64, 128, 64
        )

        self.prediction_heads = GeometryAwarePredictionHeads(
            64
        )

        self.decoder = GeometryGuidedDecoder(
            32, 48, 64, 128, 64
        )

        self.final_fusion = ConvBNAct(
            128, 64, 3
        )

        self.final_head = nn.Sequential(
            ConvBNAct(64, 64, 3),
            nn.Conv2d(64, 1, 1),
        )

        # FIX: context has 128 channels.
        self.coarse_head = nn.Conv2d(128, 1, 1)

    def forward(self, x):
        high = self.high_resolution(x)

        f4 = high["f4"]
        f8 = high["f8"]
        f16 = high["f16"]

        context = self.context_encoder(f16)

        initial_features, scale_weights = self.scale_fusion(
            f4, f8, f16, context
        )

        heads = self.prediction_heads(initial_features)

        decoder = self.decoder(
            f4, f8, f16, context
        )

        features = self.final_fusion(
            torch.cat(
                [
                    heads["features"],
                    decoder["features"],
                ],
                dim=1,
            )
        )

        mask_logits = self.final_head(features)

        coarse_logits = self.coarse_head(
            F.interpolate(
                context,
                size=f4.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )

        guidance = decoder["guidance"]

        return {
            "mask_logits": mask_logits,
            "mask": torch.sigmoid(mask_logits),

            "boundary_logits": heads["boundary_logits"],
            "boundary": heads["boundary"],

            "centerline_logits": heads["centerline_logits"],
            "centerline": heads["centerline"],

            "uncertainty_logits": heads["uncertainty_logits"],
            "uncertainty": heads["uncertainty"],

            "direction": heads["direction"],

            "coarse_logits": coarse_logits,
            "coarse": torch.sigmoid(coarse_logits),

            # Expose the exact V2 multi-scale feature maps for the V3 semantic/refinement branch.
            # This does NOT change the V2 parameter/state-tensor contract.
            "f4": f4,
            "f8": f8,
            "f16": f16,
            "features": features,
            "initial_features": initial_features,
            "scale_weights": scale_weights,

            "decoder_guidance_boundary":
                guidance["boundary"],
            "decoder_guidance_centerline":
                guidance["centerline"],
            "decoder_guidance_uncertainty":
                guidance["uncertainty"],
        }

class ConvGNAct(nn.Module):
    def __init__(self,cin,cout,k=3,stride=1,padding=None,groups=8):
        super().__init__(); padding=k//2 if padding is None else padding; g=min(groups,cout)
        while cout%g!=0 and g>1: g-=1
        self.block=nn.Sequential(nn.Conv2d(cin,cout,k,stride=stride,padding=padding,bias=False),nn.GroupNorm(g,cout),nn.GELU())
    def forward(self,x): return self.block(x)

class CoordAtt(nn.Module):
    def __init__(self,c,reduction=8):
        super().__init__(); h=max(8,c//reduction); g=min(8,h)
        while h%g!=0 and g>1: g-=1
        self.shared=nn.Sequential(nn.Conv2d(c,h,1,bias=False),nn.GroupNorm(g,h),nn.GELU()); self.h=nn.Conv2d(h,c,1); self.w=nn.Conv2d(h,c,1)
    def forward(self,x):
        _,_,h,w=x.shape; ph=x.mean(3,keepdim=True); pw=x.mean(2,keepdim=True).permute(0,1,3,2); y=self.shared(torch.cat([ph,pw],2)); yh,yw=torch.split(y,[h,w],2); yw=yw.permute(0,1,3,2); return x*torch.sigmoid(self.h(yh))*torch.sigmoid(self.w(yw))

class DilatedResidual(nn.Module):
    def __init__(self,c):
        super().__init__(); self.pre=ConvGNAct(c,c,3); self.d1=nn.Conv2d(c,c,3,padding=1,groups=c,bias=False); self.d2=nn.Conv2d(c,c,3,padding=2,dilation=2,groups=c,bias=False); self.d4=nn.Conv2d(c,c,3,padding=4,dilation=4,groups=c,bias=False); self.mix=ConvGNAct(c*3,c,1); self.attn=CoordAtt(c); self.post=ConvGNAct(c,c,3)
    def forward(self,x): z=self.pre(x); z=self.mix(torch.cat([self.d1(z),self.d2(z),self.d4(z)],1)); return self.post(self.attn(z))

class ResNet18Semantic(nn.Module):
    def __init__(self,weights_path=None):
        super().__init__(); from torchvision.models import resnet18; self.backbone=resnet18(weights=None); self.pretrained=False
        if weights_path is not None and Path(weights_path).exists(): self.backbone.load_state_dict(torch.load(weights_path,map_location="cpu",weights_only=False),strict=True); self.pretrained=True
    def forward(self,x):
        b=self.backbone; x=b.relu(b.bn1(b.conv1(x))); x=b.maxpool(x); f1=b.layer1(x); f2=b.layer2(f1); f3=b.layer3(f2); f4=b.layer4(f3); return {"f1":f1,"f2":f2,"f3":f3,"f4":f4}

class SemanticFuse(nn.Module):
    def __init__(self):
        super().__init__(); self.p1=ConvGNAct(64,24,1); self.p2=ConvGNAct(128,24,1); self.p3=ConvGNAct(256,24,1); self.p4=ConvGNAct(512,24,1); self.fuse=ConvGNAct(96,64,1); self.attn=CoordAtt(64)
    def forward(self,p,size):
        a=F.interpolate(self.p1(p["f1"]),size=size,mode="bilinear",align_corners=False); b=F.interpolate(self.p2(p["f2"]),size=size,mode="bilinear",align_corners=False); c=F.interpolate(self.p3(p["f3"]),size=size,mode="bilinear",align_corners=False); d=F.interpolate(self.p4(p["f4"]),size=size,mode="bilinear",align_corners=False); return self.attn(self.fuse(torch.cat([a,b,c,d],1)))

class V3(nn.Module):
    def __init__(self,weights_path=None):
        super().__init__(); self.v2=CrackTopoSFNetV2(); self.semantic=ResNet18Semantic(weights_path); self.semantic_fuse=SemanticFuse(); self.semantic_aux=nn.Sequential(ConvGNAct(64,32,3),CoordAtt(32),nn.Conv2d(32,1,1)); self.detail=ConvGNAct(32,32,3); self.fuse=nn.Sequential(ConvGNAct(160,96,3),CoordAtt(96),ConvGNAct(96,64,3)); self.refine=DilatedResidual(64); self.residual_out=nn.Conv2d(64,64,1)
        nn.init.zeros_(self.residual_out.weight); nn.init.zeros_(self.residual_out.bias)
        self.main_delta=nn.Sequential(ConvGNAct(64,64,3),CoordAtt(64),ConvGNAct(64,32,3),nn.Conv2d(32,1,1)); nn.init.zeros_(self.main_delta[-1].weight); nn.init.zeros_(self.main_delta[-1].bias)
        self.boundary_delta=nn.Sequential(ConvGNAct(64,32,3),nn.Conv2d(32,1,1)); nn.init.zeros_(self.boundary_delta[-1].weight); nn.init.zeros_(self.boundary_delta[-1].bias)
        self.centerline_delta=nn.Sequential(ConvGNAct(64,32,3),nn.Conv2d(32,1,1)); nn.init.zeros_(self.centerline_delta[-1].weight); nn.init.zeros_(self.centerline_delta[-1].bias)
        self.aux_f8=nn.Sequential(ConvGNAct(48,32,3),nn.Conv2d(32,1,1)); self.aux_f16=nn.Sequential(ConvGNAct(64,32,3),nn.Conv2d(32,1,1))
    def forward(self,x):
        base=self.v2(x); full=base["features"].shape[-2:]; half=(full[0]//2,full[1]//2); sem=self.semantic(x); semh=self.semantic_fuse(sem,half); sem_aux=F.interpolate(self.semantic_aux(semh),size=full,mode="bilinear",align_corners=False); detail=F.interpolate(self.detail(base["f4"]),size=half,mode="bilinear",align_corners=False); baseh=F.interpolate(base["features"],size=half,mode="bilinear",align_corners=False); fused=self.fuse(torch.cat([baseh,semh,detail],1)); delta=F.interpolate(self.residual_out(self.refine(fused)),size=full,mode="bilinear",align_corners=False); enhanced=base["features"]+delta; ml=base["mask_logits"]+self.main_delta(enhanced); bl=base["boundary_logits"]+self.boundary_delta(enhanced); cl=base["centerline_logits"]+self.centerline_delta(enhanced); a8=F.interpolate(self.aux_f8(base["f8"]),size=full,mode="bilinear",align_corners=False); a16=F.interpolate(self.aux_f16(base["f16"]),size=full,mode="bilinear",align_corners=False)
        return {"mask_logits":ml,"mask":torch.sigmoid(ml),"boundary_logits":bl,"boundary":torch.sigmoid(bl),"centerline_logits":cl,"centerline":torch.sigmoid(cl),"aux_f8_logits":a8,"aux_f16_logits":a16,"semantic_aux_logits":sem_aux,"base":base}

class HighResolutionDetailBranchV4(nn.Module):
    def __init__(self, in_channels=35, channels=48):
        super().__init__()
        self.stem = ConvGNAct(in_channels, channels, 3)
        self.refine1 = ConvGNAct(channels, channels, 3)
        self.refine2 = ConvGNAct(channels, channels, 3)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x, v3_f4):
        z = self.stem(torch.cat([x, v3_f4], dim=1))
        z = self.refine1(z)
        z = self.refine2(z)
        return z * (1.0 + self.gate(z))

class FrequencyDetailBranchV4(nn.Module):
    def __init__(self, out_channels=24):
        super().__init__()
        self.register_buffer(
            "laplace",
            torch.tensor(
                [[[-1.0, -1.0, -1.0],
                  [-1.0,  8.0, -1.0],
                  [-1.0, -1.0, -1.0]]],
                dtype=torch.float32,
            ).unsqueeze(1),
        )
        self.register_buffer(
            "sobel_x",
            torch.tensor(
                [[[-1.0, 0.0, 1.0],
                  [-2.0, 0.0, 2.0],
                  [-1.0, 0.0, 1.0]]],
                dtype=torch.float32,
            ).unsqueeze(1),
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor(
                [[[-1.0, -2.0, -1.0],
                  [0.0,  0.0,  0.0],
                  [1.0,  2.0,  1.0]]],
                dtype=torch.float32,
            ).unsqueeze(1),
        )
        self.stem = ConvGNAct(4, out_channels, 3)
        self.refine = nn.Sequential(
            ConvGNAct(out_channels, out_channels, 3),
            CoordAtt(out_channels),
            ConvGNAct(out_channels, out_channels, 3),
        )

    def forward(self, x):
        gray = x.float().mean(dim=1, keepdim=True)
        lap = F.conv2d(gray, self.laplace.to(dtype=gray.dtype), padding=1)
        gx = F.conv2d(gray, self.sobel_x.to(dtype=gray.dtype), padding=1)
        gy = F.conv2d(gray, self.sobel_y.to(dtype=gray.dtype), padding=1)
        grad = torch.sqrt(gx * gx + gy * gy + 1e-6)
        z = torch.cat([gray, lap.abs(), gx.abs(), grad], dim=1)
        z = self.stem(z)
        return self.refine(z)

class DirectionalAdaptiveConvV4(nn.Module):
    """Learned local directional aggregation without custom CUDA/deformable ops."""
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Conv2d(channels, 5, 1)
        self.proj = ConvGNAct(channels, channels, 3)

    @staticmethod
    def _shift(x, dy, dx):
        h, w = x.shape[-2:]
        pad_l = max(dx, 0)
        pad_r = max(-dx, 0)
        pad_t = max(dy, 0)
        pad_b = max(-dy, 0)
        y = F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode="replicate")
        y0 = max(-dy, 0)
        x0 = max(-dx, 0)
        return y[:, :, y0:y0 + h, x0:x0 + w]

    def forward(self, x):
        w = torch.softmax(self.weight(x), dim=1)
        c = x
        u = self._shift(x, -1, 0)
        d = self._shift(x, 1, 0)
        l = self._shift(x, 0, -1)
        r = self._shift(x, 0, 1)
        z = (
            w[:, 0:1] * c
            + w[:, 1:2] * u
            + w[:, 2:3] * d
            + w[:, 3:4] * l
            + w[:, 4:5] * r
        )
        return self.proj(z)

class CrossScaleGatedFusionV4(nn.Module):
    def __init__(self, in_channels=248, out_channels=96):
        super().__init__()
        self.fuse = ConvGNAct(in_channels, out_channels, 1)
        self.attn = CoordAtt(out_channels)
        self.gate = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.post = ConvGNAct(out_channels, out_channels, 3)

    def forward(self, v3_features, v3_f8, v3_f16, detail, frequency):
        size = detail.shape[-2:]
        a = v3_features
        b = F.interpolate(v3_f8, size=size, mode="bilinear", align_corners=False)
        c = F.interpolate(v3_f16, size=size, mode="bilinear", align_corners=False)
        z = torch.cat([a, b, c, detail, frequency], dim=1)
        z = self.fuse(z)
        z = self.attn(z)
        g = self.gate(z)
        return self.post(z * (1.0 + g))

class BoundaryBodyCoherenceV4(nn.Module):
    def __init__(self, channels=96):
        super().__init__()
        self.body = ConvGNAct(channels, 64, 3)
        self.boundary = ConvGNAct(channels, 32, 3)
        self.body_from_boundary = nn.Sequential(nn.Conv2d(32, 64, 1), nn.Sigmoid())
        self.boundary_from_body = nn.Sequential(nn.Conv2d(64, 32, 1), nn.Sigmoid())
        self.fuse = ConvGNAct(96, 64, 3)
        self.refine = ConvGNAct(64, 64, 3)

    def forward(self, x):
        body = self.body(x)
        boundary = self.boundary(x)
        body_gate = self.body_from_boundary(boundary)
        boundary_gate = self.boundary_from_body(body)
        body2 = body * (1.0 + body_gate)
        boundary2 = boundary * (1.0 + boundary_gate)
        z = self.fuse(torch.cat([body2, boundary2], dim=1))
        return self.refine(z), body2, boundary2

class TopologyCenterlineGateV4(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.center = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, channels, 1),
            nn.Sigmoid(),
        )
        self.post = ConvGNAct(channels, channels, 3)

    def forward(self, x, centerline_prob):
        g = self.center(centerline_prob)
        return self.post(x * (1.0 + 0.75 * g))

class CrackTopoSFNetV4(nn.Module):
    def __init__(self):
        super().__init__()
        self.v3 = V3(weights_path=None)
        self.detail = HighResolutionDetailBranchV4(35, 48)
        self.frequency = FrequencyDetailBranchV4(24)
        self.cross_scale = CrossScaleGatedFusionV4(248, 96)
        self.geometry = DirectionalAdaptiveConvV4(96)
        self.coherence = BoundaryBodyCoherenceV4(96)
        self.topology = TopologyCenterlineGateV4(64)
        self.final_refine = nn.Sequential(
            ConvGNAct(64, 64, 3),
            CoordAtt(64),
            ConvGNAct(64, 64, 3),
        )

        self.main_delta = nn.Sequential(
            ConvGNAct(64, 64, 3),
            CoordAtt(64),
            ConvGNAct(64, 32, 3),
            nn.Conv2d(32, 1, 1),
        )
        self.boundary_delta = nn.Sequential(
            ConvGNAct(64, 32, 3),
            nn.Conv2d(32, 1, 1),
        )
        self.centerline_delta = nn.Sequential(
            ConvGNAct(64, 32, 3),
            nn.Conv2d(32, 1, 1),
        )
        self.thin_aux = nn.Sequential(
            ConvGNAct(64, 32, 3),
            CoordAtt(32),
            nn.Conv2d(32, 1, 1),
        )

        # Zero-residual initialization: before training V4 reproduces V3 exactly.
        for head in [self.main_delta, self.boundary_delta, self.centerline_delta]:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, x):
        base = self.v3(x)
        v3_features = base["base"]["features"]
        v3_f4 = base["base"]["f4"]
        v3_f8 = base["base"]["f8"]
        v3_f16 = base["base"]["f16"]

        detail = self.detail(x, v3_f4)
        frequency = self.frequency(x)
        fused = self.cross_scale(v3_features, v3_f8, v3_f16, detail, frequency)
        fused = self.geometry(fused)
        coherent, body_feat, boundary_feat = self.coherence(fused)
        refined = F.interpolate(coherent, size=v3_features.shape[-2:], mode="bilinear", align_corners=False)
        refined = self.topology(refined, base["centerline"])
        refined = self.final_refine(refined)

        main_delta = self.main_delta(refined)
        boundary_delta = self.boundary_delta(refined)
        centerline_delta = self.centerline_delta(refined)
        thin_aux = self.thin_aux(refined)

        mask_logits = base["mask_logits"] + main_delta
        boundary_logits = base["boundary_logits"] + boundary_delta
        centerline_logits = base["centerline_logits"] + centerline_delta

        return {
            "mask_logits": mask_logits,
            "mask": torch.sigmoid(mask_logits),
            "boundary_logits": boundary_logits,
            "boundary": torch.sigmoid(boundary_logits),
            "centerline_logits": centerline_logits,
            "centerline": torch.sigmoid(centerline_logits),
            "thin_aux_logits": thin_aux,
            "v3": base,
            "v4_features": refined,
            "v4_body_features": body_feat,
            "v4_boundary_features": boundary_feat,
        }
