import torch
import torch.nn as nn

from core.update import BasicMultiUpdateBlock
from core.extractor import BasicEncoder, MultiBasicEncoder, ResidualBlock
from core.corr import CorrBlock1D
from core.utils.utils import upflow8
from core.raft_stereo import RAFTStereo


class BidirDPDispNet(RAFTStereo):
    '''
    Overrided RAFT stereo init with instance normalization.
    '''
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        
        context_dims = args.hidden_dims

        self.cnet = MultiBasicEncoder(output_dim=[args.hidden_dims, context_dims], norm_fn="instance", downsample=args.n_downsample)
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)

        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])

        if args.shared_backbone:
            self.conv2 = nn.Sequential(
                ResidualBlock(128, 128, 'instance', stride=1),
                nn.Conv2d(128, 256, 3, padding=1))
        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', downsample=args.n_downsample)


    '''
    Overrided RAFT stereo forward function.
    '''
    def _forward(self, image0, image1, iters=12, flow_init=None, test_mode=False):
        image0 = (2 * image0 - 1.0).contiguous()
        image1 = (2 * image1 - 1.0).contiguous()

        if self.args.shared_backbone:
            *cnet_list, x = self.cnet(torch.cat((image0, image1), dim=0), dual_inp=True, num_layers=self.args.n_gru_layers)
            fmap0, fmap1 = self.conv2(x).split(dim=0, split_size=x.shape[0]//2)
        else:
            cnet_list = self.cnet(image0, num_layers=self.args.n_gru_layers)
            fmap0, fmap1 = self.fnet([image0, image1])
        net_list = [torch.tanh(x[0]) for x in cnet_list]
        inp_list = [torch.relu(x[1]) for x in cnet_list]

        # Rather than running the GRU's conv layers on the context features multiple times, we do it once at the beginning 
        inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list, self.context_zqr_convs)]

        if self.args.corr_implementation == "reg": # Default
            corr_block = CorrBlock1D
            fmap0, fmap1 = fmap0.float(), fmap1.float()
        else: raise NotImplementedError('Other correlation layer is not tested')
        corr_fn = corr_block(fmap0, fmap1, radius=self.args.corr_radius, num_levels=self.args.corr_levels)

        coords0, coords1 = self.initialize_flow(net_list[0])

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume
            flow = coords1 - coords0
            net_list, up_mask, delta_flow = self.update_block(net_list, inp_list, corr, flow, iter32=self.args.n_gru_layers==3, iter16=self.args.n_gru_layers>=2)

            # in stereo mode, project flow onto epipolar
            # Make [:, 1, :, :] = 0 and keep [:, 0, :, :]
            delta_flow[:,1] = 0.0

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # We do not need to upsample or output intermediate results in test_mode
            if test_mode and itr < iters-1:
                continue

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_up = flow_up[:,:1]

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions


    def _forward_dp(self, imgL, imgR, iters=12, flow_init=None, test_mode=True):
        imgC = (imgL + imgR) / 2.0
        predictions = {}

        if test_mode:
            _, CtoL = self._forward(imgC, imgL, iters=iters, flow_init=flow_init, test_mode=True)
            _, CtoR = self._forward(imgC, imgR, iters=iters, flow_init=flow_init, test_mode=True)
            predictions['LCR'] = -CtoL + CtoR
        else:
            predictions['CtoL'] = self._forward(imgC, imgL, iters=iters, flow_init=flow_init, test_mode=False)
            predictions['CtoR'] = self._forward(imgC, imgR, iters=iters, flow_init=flow_init, test_mode=False)
            predictions['LCR'] = [(-CtoL + CtoR) for CtoL, CtoR in zip(predictions['CtoL'], predictions['CtoR'])]
            predictions['RCL'] = [(-CtoR + CtoL) for CtoL, CtoR in zip(predictions['CtoL'], predictions['CtoR'])]

        return predictions


    def forward(self, img1, img2, iters=12, flow_init=None, test_mode=False):
        return self._forward_dp(img1, img2, iters=iters, flow_init=flow_init, test_mode=test_mode)






