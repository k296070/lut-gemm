import torch
import torch.nn as nn
import numpy as np

#from clet.functions.rtn import Quantizer
from utils import CompressionParameter, PACKER, Quantizer
from bcq_parameter import BCQParameter

class RTNParameter(CompressionParameter):
    def compress(self, in_ch_wise=False, **kwargs):
        data_shape = self.data.shape
        group_size = -1
        if 'group_size' in kwargs:
            group_size = kwargs.pop('group_size')
        out_ch = data_shape[0]
        in_ch = data_shape[1]

        quant = Quantizer()
        quant.configure(**kwargs)
        if in_ch_wise == False:
            data = self.data
            if group_size > 0:
                data = data.reshape([-1, group_size])
            quant.find_params(data, weight=True)
            quant_data  = torch.clamp(torch.round(data / quant.scale) + quant.zero, 0, quant.maxq)
            quant_data  = quant_data.reshape([out_ch, -1]).to(torch.int)
            quant.scale = quant.scale.reshape([out_ch, -1, 1])
            quant.zero  = quant.zero.reshape([out_ch, -1, 1])
        else:
            data = self.data.T
            if group_size > 0:
                data = data.reshape([-1, group_size])
            quant.find_params(data, weight=True)
            quant_data = torch.clamp(torch.round(data / quant.scale) + quant.zero, 0, quant.maxq)
            quant_data = quant_data.reshape([in_ch, -1, group_size]).to(torch.int)
            quant.scale = quant.scale.reshape([in_ch, -1, 1])
            quant.zero  = quant.zero.reshape([in_ch, -1, 1])

        return quant.scale, quant.zero, quant_data, quant_data.shape

    def decompress(self, scale, zero, quant_data, quant_data_shape, in_ch_wise=False):
        # w.shape = [out_ch, in_ch]
        # in_ch_wise == True
        #   -> quant_data.shape = [in_ch, out_ch//group_size, group_size]
        #   -> scale.shape      = [in_ch, out_ch//group_size, 1]
        #   -> zero.shape       = [in_ch, out_ch//group_size, 1]
        # in_ch_wise == False
        #   -> quant_data.shape = [out_ch, in_ch//group_size, group_size]
        #   -> scale.shape      = [out_ch, in_ch//group_size, 1]
        #   -> zero.shape       = [out_ch, in_ch//group_size, 1]

        if in_ch_wise == True:
            out_ch = quant_data_shape[1] * quant_data_shape[2]
            decomp_w = scale * (quant_data - zero)
            decomp_w = decomp_w.reshape([-1, out_ch]).T
        else:
            out_ch = quant_data_shape[0]
            decomp_w = scale * (quant_data - zero)
            decomp_w = decomp_w.reshape([out_ch, -1])
        self.data = decomp_w

    def convert_bcq_format(self, scale, zero, quant_data, qbits, do_packing=False, in_ch_wise=False):
        global PACKER

        zero   = scale * zero #O ,#G,1
        upack  = torch.Tensor([[2**(i) for i in range(qbits)]])
        scale  = scale / 2.0
        scale  = torch.matmul(scale, upack) #O G B

        offset = scale.sum(-1).unsqueeze(-1) - zero #O G 1
        offset= offset.reshape(offset.shape[0],-1)
        binary = torch.zeros(list(quant_data.shape) + [qbits],dtype =torch.int32)
        binary_shape = binary.shape
        
        quant_data = quant_data.to(torch.int)
        for i in range(qbits):
            binary[:, :, i] = ((quant_data >> i) & 1) # O I B

        N = binary.shape[0] #output

        scale_ = scale.permute(1,2,0).contiguous() # G B O
        binary_ = binary.permute(0,2,1).reshape(-1,32).contiguous().to(torch.int64).to(torch.device('cuda'))
        offset_ = offset.permute(1,0).contiguous() # G O

        matrix_132 = torch.tensor([1<<i for i in range(32)],dtype=torch.int64,device= "cuda")

        bW_ = binary_*matrix_132

        bW__ = torch.sum(bW_, dim=1, keepdim=True)
        bW___ = bW__.reshape(N,qbits,-1).permute(2,1,0).to(torch.int32).contiguous()
        
        return scale_, bW___, binary_shape, offset_

if __name__ == '__main__':
    w_org = torch.randn(1024, 4096)

    # INT4 Quantization -> RTN
    w_rtn = RTNParameter(w_org)
    scale, zero, w_quant, w_quant_shape = w_rtn.compress(in_ch_wise=False, qbits=4, group_size=128, perchannel=True, sym=False)
    #scale, zero, w_quant, w_quant_shape = w_rtn.compress(in_ch_wise=False, qbits=4, group_size=128, perchannel=True, sym=False)

    print("quant",scale.shape,zero.shape,w_quant.shape)
    # Convert INT4 -> BCQ4
    alpha, binary, binary_shape, offset = w_rtn.convert_bcq_format(scale, zero, w_quant, qbits=4, do_packing=True, in_ch_wise=True)
    state_dict = {
        "alpha":alpha,
        "binary" :binary,
        "q_bias" :offset
    }
    print(binary.size(),alpha.size(),offset.size())
    torch.save(w_org,"../../random_weight.pt")
    torch.save(state_dict,"../../random_weight_packed.pt")
