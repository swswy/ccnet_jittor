import jittor as jt
import jittor.nn as nn

jt.flags.use_cuda = 1

def INF(B,H,W):
    # return jt.repeat(jt.diag(jt.repeat(jt.Var(float("-inf")),H),0), B*W, 1, 1)
    return jt.diag(jt.Var(float("-inf")).repeat(H),0).repeat(B*W,1,1)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(jt.zeros(1))

    def execute(self, x):
        m_batchsize, _, height, width = x.shape
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).view(m_batchsize*height,-1,width)
        energy_H = (jt.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = jt.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = nn.softmax(jt.concat([energy_H, energy_W], 3),dim=3)

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H)
        att_W = concate[:,:,:,height:height+width].view(m_batchsize*height,width,width)
        out_H = jt.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = jt.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        return self.gamma*(out_H + out_W) + x

class RCCAModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(RCCAModule, self).__init__()
        inter_channels = in_channels // 4
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels))
        self.cca = CrissCrossAttention(inter_channels)
        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def execute(self, x, recurrence=1):
        output = self.conva(x)
        for i in range(recurrence):
            output = self.cca(output)
        output = self.convb(output)

        output = self.bottleneck(jt.concat([x, output], 1))
        return output
        
if __name__ == '__main__':
    model = CrissCrossAttention(64)
    x = jt.randn(2, 64, 5, 6)
    out = model(x)
    print(out.shape)
