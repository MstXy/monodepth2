# codes from https://github.com/sniklaus/pytorch-unflow.git
import math
import numpy
import PIL
import PIL.Image
import sys
import torch
from monodepth2.networks.correlation import correlation  # the custom cost volume layer



##########################################################

arguments_strModel = 'css' # 'css', or 'css-synthia'
arguments_strOne = '../assets/one.png'
arguments_strTwo = '../assets/two.png'
arguments_strOut = '../assets/out.flo'


##########################################################


backwarp_tenGrid = {}


def backwarp(tenInput, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1).to(tenInput.device)
        tenVer = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3]).to(tenInput.device)

        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).to(tenInput.device)
    # end

    tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] * (2.0 / (tenInput.shape[3] - 1.0)), tenFlow[:, 1:2, :, :] * (2.0 / (tenInput.shape[2] - 1.0)) ], 1)

    return torch.nn.functional.grid_sample(input=tenInput,
                                           grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1),
                                           mode='bilinear', padding_mode='border', align_corners=True)
# end

##########################################################
class UnFlowNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        class Upconv(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.netSixOut = torch.nn.Conv2d(in_channels=1024, out_channels=2, kernel_size=3, stride=1, padding=1)

                self.netSixUp = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)

                self.netFivNext = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFivOut = torch.nn.Conv2d(in_channels=1026, out_channels=2, kernel_size=3, stride=1, padding=1)

                self.netFivUp = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)

                self.netFouNext = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(in_channels=1026, out_channels=256, kernel_size=4, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFouOut = torch.nn.Conv2d(in_channels=770, out_channels=2, kernel_size=3, stride=1, padding=1)

                self.netFouUp = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)

                self.netThrNext = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(in_channels=770, out_channels=128, kernel_size=4, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netThrOut = torch.nn.Conv2d(in_channels=386, out_channels=2, kernel_size=3, stride=1, padding=1)

                self.netThrUp = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)

                self.netTwoNext = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(in_channels=386, out_channels=64, kernel_size=4, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netTwoOut = torch.nn.Conv2d(in_channels=194, out_channels=2, kernel_size=3, stride=1, padding=1)

                self.netUpscale = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=3, stride=2, padding=1, bias=False),
                    torch.nn.ReplicationPad2d(padding=[ 0, 1, 0, 1 ])
                )
            # end

            def forward(self, tenOne, tenTwo, objInput):
                objOutput = {}

                tenInput = objInput['conv6']
                objOutput['flow6'] = self.netSixOut(tenInput)
                tenInput = torch.cat([ objInput['conv5'], self.netFivNext(tenInput), self.netSixUp(objOutput['flow6']) ], 1)
                objOutput['flow5'] = self.netFivOut(tenInput)
                tenInput = torch.cat([ objInput['conv4'], self.netFouNext(tenInput), self.netFivUp(objOutput['flow5']) ], 1)
                objOutput['flow4'] = self.netFouOut(tenInput)
                tenInput = torch.cat([ objInput['conv3'], self.netThrNext(tenInput), self.netFouUp(objOutput['flow4']) ], 1)
                objOutput['flow3'] = self.netThrOut(tenInput)
                tenInput = torch.cat([ objInput['conv2'], self.netTwoNext(tenInput), self.netThrUp(objOutput['flow3']) ], 1)
                objOutput['flow2'] = self.netTwoOut(tenInput)

                return self.netUpscale(self.netUpscale(objOutput['flow2'])) * 20.0
            # end
        # end

        class Complex(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.netOne = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([ 2, 4, 2, 4 ]),
                    torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netTwo = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([ 1, 3, 1, 3 ]),
                    torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netThr = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([ 1, 3, 1, 3 ]),
                    torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netRedir = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=256, out_channels=32, kernel_size=1, stride=1, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netCorrelation = correlation.ModuleCorrelation()

                self.netCombined = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=473, out_channels=256, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFou = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([ 0, 2, 0, 2 ]),
                    torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
                
                self.netFiv = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([ 0, 2, 0, 2 ]),
                    torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
                
                self.netSix = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([ 0, 2, 0, 2 ]),
                    torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netUpconv = Upconv()
            # end

            def forward(self, tenOne, tenTwo, tenFlow):
                objOutput = {}

                assert(tenFlow is None)

                objOutput['conv1'] = self.netOne(tenOne)
                objOutput['conv2'] = self.netTwo(objOutput['conv1'])
                objOutput['conv3'] = self.netThr(objOutput['conv2'])

                tenRedir = self.netRedir(objOutput['conv3'])
                tenOther = self.netThr(self.netTwo(self.netOne(tenTwo)))
                tenCorr = self.netCorrelation(objOutput['conv3'], tenOther)

                objOutput['conv3'] = self.netCombined(torch.cat([ tenRedir, tenCorr ], 1))
                objOutput['conv4'] = self.netFou(objOutput['conv3'])
                objOutput['conv5'] = self.netFiv(objOutput['conv4'])
                objOutput['conv6'] = self.netSix(objOutput['conv5'])

                return self.netUpconv(tenOne, tenTwo, objOutput)
            # end
        # end

        class Simple(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.netOne = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([ 2, 4, 2, 4 ]),
                    torch.nn.Conv2d(in_channels=14, out_channels=64, kernel_size=7, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netTwo = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([ 1, 3, 1, 3 ]),
                    torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netThr = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([ 1, 3, 1, 3 ]),
                    torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFou = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([ 0, 2, 0, 2 ]),
                    torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFiv = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([ 0, 2, 0, 2 ]),
                    torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netSix = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([ 0, 2, 0, 2 ]),
                    torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netUpconv = Upconv()
            # end

            def forward(self, tenOne, tenTwo, tenFlow):
                objOutput = {}

                tenWarp = backwarp(tenInput=tenTwo, tenFlow=tenFlow)

                objOutput['conv1'] = self.netOne(torch.cat([ tenOne, tenTwo, tenFlow, tenWarp, (tenOne - tenWarp).abs() ], 1))
                objOutput['conv2'] = self.netTwo(objOutput['conv1'])
                objOutput['conv3'] = self.netThr(objOutput['conv2'])
                objOutput['conv4'] = self.netFou(objOutput['conv3'])
                objOutput['conv5'] = self.netFiv(objOutput['conv4'])
                objOutput['conv6'] = self.netSix(objOutput['conv5'])

                return self.netUpconv(tenOne, tenTwo, objOutput)
            # end
        # end

        self.netFlownets = torch.nn.ModuleList([
            Complex(),
            Simple(),
            Simple()
        ])
        # self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-unflow/network-' + arguments_strModel + '.pytorch', file_name='unflow-' + arguments_strModel).items() })
    # end

    def forward_once(self, tenOne, tenTwo):
        tenFlow = None
        for netFlownet in self.netFlownets:
            tenFlow = netFlownet(tenOne, tenTwo, tenFlow)
        return tenFlow

    def forward(self, inputs):
        outputs = {}
        img_n1, img_0, img_1 = inputs['color_aug', -1, 0], inputs['color_aug', 0, 0], inputs['color_aug', 1, 0]
        outputs['flow_fwd'] = [self.forward_once(img_n1, img_0), self.forward_once(img_0, img_1)]
        outputs['flow_bwd'] = [self.forward_once(img_0, img_n1), self.forward_once(img_1, img_0)]
        return outputs


##########################################################

def estimate(tenOne, tenTwo):
    global netNetwork

    if netNetwork is None:
        netNetwork = UnFlowNet().cuda().eval()
    # end

    assert(tenOne.shape[1] == tenTwo.shape[1])
    assert(tenOne.shape[2] == tenTwo.shape[2])

    intWidth = tenOne.shape[2]
    intHeight = tenOne.shape[1]

    assert(intWidth == 1280) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    assert(intHeight == 384) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

    tenPreprocessedOne = tenOne.cuda().view(1, 3, intHeight, intWidth)
    tenPreprocessedTwo = tenTwo.cuda().view(1, 3, intHeight, intWidth)

    intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
    intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

    tenPreprocessedOne = torch.nn.functional.interpolate(input=tenPreprocessedOne, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
    tenPreprocessedTwo = torch.nn.functional.interpolate(input=tenPreprocessedTwo, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

    tenFlow = torch.nn.functional.interpolate(input=netNetwork(tenPreprocessedOne, tenPreprocessedTwo), size=(intHeight, intWidth), mode='bilinear', align_corners=False)
    tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
    tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

    return tenFlow[0, :, :, :].cpu()
# end

##########################################################

if __name__ == '__main__':
    tmp = PIL.Image.open(arguments_strOne)
    tmp1 = numpy.array(tmp).transpose(2, 0, 1)
    tmp2 = tmp1[::-1, :, :]

    tenOne = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(arguments_strOne))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
    tenTwo = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(arguments_strTwo))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))

    tenOutput = estimate(tenOne, tenTwo)
    import monodepth2.utils.utils as mono_utils
    mono_utils.stitching_and_show(img_list=[tenOne, tenTwo, tenOutput], ver=True, show=True)

    objOutput = open(arguments_strOut, 'wb')
    numpy.array([ 80, 73, 69, 72 ], numpy.uint8).tofile(objOutput)
    numpy.array([ tenOutput.shape[2], tenOutput.shape[1] ], numpy.int32).tofile(objOutput)
    numpy.array(tenOutput.numpy().transpose(1, 2, 0), numpy.float32).tofile(objOutput)
    objOutput.close()
# end














