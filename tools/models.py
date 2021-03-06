def init_lstm_states(num_layers, batch_size, hidden_size):
    return (torch.zeros(num_layers, batch_size, hidden_size).to(device),
            torch.zeros(num_layers, batch_size, hidden_size).to(device))

################## Wrap Model ################## 

class Model(nn.Module):
    
    """
    Recurrent Refinement Network combining ResNet, lang_lstm and RRN
    
    """

    def __init__(self, opt):
        super(Model, self).__init__()
        
        self.num_layers = opt.num_layers
        self.batch_size = opt.batch_size
        self.hidden_size = opt.hidden_size
        self.base = opt.base
        
        if opt.base == 'v2':
            self.base = DeepLabV2(n_classes=None, n_blocks=opt.n_blocks, atrous_rates=opt.atrous_rates).to(device)
            #(1, 40, 40, 2048) -> shape=(1, 40, 40, 1000)
            self.layer5_feat = nn.Conv2d(in_channels = opt.vf_dim, out_channels = 1000, kernel_size=1, stride=1).to(device)
        else:
            self.base = DeepLabV3(n_classes=None, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18], multi_grids=[1, 2, 4], output_stride=8).to(device)
            #(1, 40, 40, 2048) -> shape=(1, 40, 40, 1000)
            concat_v3 = 256 * (2 + 2)
            #concat_v3 = 256 * (3 + 2)
            self.layer5_feat = nn.Conv2d(in_channels = concat_v3, out_channels = 1000, kernel_size=1, stride=1).to(device)
        
        spatial = generate_spatial_batch(self.batch_size, opt.vf_h, opt.vf_w)
        self.spatial_grid = torch.from_numpy(spatial).permute(0,3,1,2).to(device)

        self.LSTM = LSTM_Model(opt.vocab_size, opt.embed_size, opt.hidden_size, opt.num_layers, opt.rnn_size, opt.vf_h, opt.vf_w,
                               opt.batch_size,opt.num_steps, opt.minval, opt.maxval).to(device)

        self.RRN = RRN(opt.embed_size, opt.rnn_size, opt.mlp_dim, opt.vf_dim, opt.im_h, opt.im_w).to(device)
        
    
    def forward(self, image, text):
        
        # Deeplab
        # 3: (1, 512, 40, 40), 4: (1, 1024, 40, 40), 5: (1, 2048, 40, 40)
        if opt.base == 'v2':
            layer3, layer4, layer5 = self.base(image)
        else:
            layer3, layer4, layer5, aspp = self.base(image)

        # Language lstm
        lstm_states = init_lstm_states(self.num_layers, self.batch_size, self.hidden_size)
        lstm_output = self.LSTM(text, lstm_states)        #(1, 40, 40, 1000)
        lstm_output = torch.transpose(lstm_output, 1, 3)  #(1, 1000, 40, 40)

        # concat (ResNet, lstm, spatial_grid)
        if opt.base == 'v2':
            layer5_cat = self.layer5_feat(layer5)
            fusion =  torch.cat((layer5_cat, lstm_output, self.spatial_grid),1) #(1, 2008, 40, 40)
        else:
            aspp_cat = self.layer5_feat(aspp)
            fusion =  torch.cat((aspp_cat, lstm_output, self.spatial_grid),1) #(1, 2008, 40, 40)            

        # RRN pass
        output_down, output_up = self.RRN(fusion, layer5, layer4, layer3)
        return output_down, output_up #(1, 1, 40, 40), (1, 1, 320, 320)
    


################## RRN ################## 

   
class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias, peephole = True, normalize = False):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self._peephole = peephole
        self._normalize = normalize
        self.W_ci = nn.Parameter(torch.rand((1, 500, 40, 40), requires_grad = True)).to(device)
        self.W_cf = nn.Parameter(torch.rand((1, 500, 40, 40), requires_grad = True)).to(device)
        self.W_co = nn.Parameter(torch.rand((1, 500, 40, 40), requires_grad = True)).to(device)
        self.layer_norm = nn.LayerNorm([500, 40, 40])

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        #TODO if I understand correctly, there is no bias if self._normalize == False
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        
        if self._peephole:
            cc_i = cc_i + self.W_ci * c_cur
            cc_f = cc_f + self.W_cf * c_cur

        if self._normalize:
            cc_g = self.layer_norm(cc_g)
            cc_i = self.layer_norm(cc_i)
            cc_f = self.layer_norm(cc_f)

        f = torch.sigmoid(cc_f)
        i = torch.sigmoid(cc_i)
        c_next = f * c_cur + i * torch.tanh(cc_g)

        if self._peephole:
            cc_o = cc_o + self.W_ci * c_next

        if self._normalize:
            cc_o = self.layer_norm(cc_o)
            c_next = self.layer_norm(c_next)

        o = torch.sigmoid(cc_o)
        h_next = o * torch.tanh(c_next)

        return h_next, (h_next, c_next)


    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width).to(device),
                torch.zeros(batch_size, self.hidden_dim, height, width).to(device))
    
    
class RRN(nn.Module):
    def __init__(self, embed_size, rnn_size, mlp_dim, vf_dim, im_h, im_w):
        super(RRN, self).__init__()
        self.im_h = im_h
        self.im_w = im_w
        self.conv_fusion = nn.Sequential(
            #TODO add in all:
            #W initializer for Conv2d: initializer=tf.contrib.layers.xavier_initializer_conv2d()
            #bias initiaizlier for Conv2d: initializer=tf.constant_initializer(0.)
            nn.Conv2d(in_channels = embed_size+rnn_size+8, out_channels = mlp_dim, kernel_size=1, stride=1), #(1, 40, 40, 2008) -> shape=(1, 40, 40, 500) 
            nn.ReLU()
        )
        self.ConvLSTM = ConvLSTMCell(500, 500, (1,1), True) #TODO update to parameters

        self.conv_c5 = nn.Sequential(
            nn.Conv2d(in_channels = vf_dim, out_channels = mlp_dim, kernel_size=1, stride=1), #(1, 40, 40, 2048) -> shape=(1, 40, 40, 500) 
            nn.ReLU()
        )

        self.conv_c4 = nn.Sequential(
            nn.Conv2d(in_channels = 1024, out_channels = mlp_dim, kernel_size=1, stride=1), #(1, 40, 40, 1024) -> shape=(1, 40, 40, 500) 
            nn.ReLU()
        )

        self.conv_c3 = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = mlp_dim, kernel_size=1, stride=1), #(1, 40, 40, 512) -> shape=(1, 40, 40, 500) 
            nn.ReLU()
        )
        self.conv_out = nn.Conv2d(in_channels = mlp_dim, out_channels = 1, kernel_size=1, stride=1) #(1, 40, 40, 500) -> shape=(1, 40, 40, 1) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, fusion, c5_lateral, c4_lateral, c3_lateral):
        fusion = self.conv_fusion(fusion)
        c5_lateral = self.conv_c5(c5_lateral)
        c4_lateral = self.conv_c4(c4_lateral)
        c3_lateral = self.conv_c3(c3_lateral)

        initial_state = self.ConvLSTM.init_hidden(batch_size= 1, image_size=(40,40))


        out, state = self.ConvLSTM(fusion, initial_state)
        out, state = self.ConvLSTM(c5_lateral, state)
        out, state = self.ConvLSTM(c4_lateral, state)
        out, state = self.ConvLSTM(c3_lateral, state) #out shape = (1, 500, 40, 40)
        
        out = self.conv_out(out)                      #shape = (1, 1, 40, 40)
        #should be same as Upsample which is deprecated
        out_up = nn.functional.interpolate(out, size=(self.im_w, self.im_h), mode='bilinear', align_corners=False) 
        #out_sigm = self.sigmoid(out_up)               #shape = (1, 1, 320, 320)

        return out, out_up #(1, 1, 40, 40), (1, 1, 320, 320)


################## LSTM ################## 


class LSTM_Model(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, rnn_size, vf_h, vf_w, batch_size, num_steps, minval, maxval):
        super(LSTM_Model, self).__init__()
        self.num_steps = num_steps
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.rnn_size = rnn_size
        self.vf_h = vf_h
        self.vf_w = vf_w
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx = 0) #12112->1000
        self.embed.weight.data.uniform_(minval, maxval)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True) 

    def get_first_nonpad_index(self,words):
        i = 0
        encoded_word_list = words.view(-1).tolist()
        for word in encoded_word_list:
          if word == 0:
            i+=1
          else:
            break
        return i
    
    def forward(self, words, state):
        first_nonpad_index = self.get_first_nonpad_index(words)                                       
        number_of_words = 20 - first_nonpad_index                                                     
        sentence_no_padding = torch.narrow(words, 2, first_nonpad_index, number_of_words).to(device)  
        embedded_seq = [self.embed(sentence_no_padding[:, :, n]) for n in range(number_of_words)]

        embedded_seq = torch.cat(embedded_seq) #(number_of_words, 1, 1000)
        #embedded_seq = embedded_seq.view(1, number_of_words, -1) #TODO this is old implementation
        embedded_seq = torch.transpose(embedded_seq, 0, 1) #TODO this is new implementation #(1, number_of_words, 1000)

        rnn_output, state = self.lstm(embedded_seq, state)  #rnn_output shape: (1, number_of_words, 1000)
        last_output = rnn_output[:, -1, :] #(1, 1000)
        

        lang_feat = last_output.view(self.batch_size, 1, 1, self.rnn_size)         #shape=(1, 1, 1, 1000)
        lang_feat = f.normalize(lang_feat, p=2, dim=3)                            #shape=(1, 1, 1, 1000) 
        lang_feat = lang_feat.repeat(1, self.vf_h, self.vf_w, 1)                  #shape=(1, 40, 40, 1000) 
        return lang_feat


################## DeepLab ################## 


class _ImagePool(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(320)
        self.conv = _ConvBnReLU(in_ch, out_ch, 1, 1, 0, 1)

    def forward(self, x):
        _, _, H, W = x.shape
        h = self.pool(x)
        h = self.conv(h)
        h = F.interpolate(h, size=(H, W), mode="bilinear", align_corners=False)
        return h


class _ASPP_v3(nn.Module):
    """
    Atrous spatial pyramid pooling with image-level feature
    """

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP_v3, self).__init__()
        self.stages = nn.Module()
        self.stages.add_module("c0", _ConvBnReLU(in_ch, out_ch, 1, 1, 0, 1))
        for i, rate in enumerate(rates):
            self.stages.add_module(
                "c{}".format(i + 1),
                _ConvBnReLU(in_ch, out_ch, 3, 1, padding=rate, dilation=rate),
            )
        #self.stages.add_module("imagepool", _ImagePool(in_ch, out_ch))

    def forward(self, x):
        return torch.cat([stage(x) for stage in self.stages.children()], dim=1)
    

class DeepLabV3(nn.Sequential):
    """
    DeepLab v3: Dilated ResNet with multi-grid + improved ASPP
    """

    def __init__(self, n_classes, n_blocks, atrous_rates, multi_grids, output_stride):
        super(DeepLabV3, self).__init__()

        # Stride and dilation
        if output_stride == 8:
            s = [1, 2, 1, 1]
            d = [1, 1, 2, 4]
        elif output_stride == 16:
            s = [1, 2, 2, 1]
            d = [1, 1, 1, 2]

        ch = [64 * 2 ** p for p in range(6)]
        self.layer1 = _Stem(ch[0])
        self.layer2 = _ResLayer(n_blocks[0], ch[0], ch[2], s[0], d[0])
        self.layer3 = _ResLayer(n_blocks[1], ch[2], ch[3], s[1], d[1])
        self.layer4 = _ResLayer(n_blocks[2], ch[3], ch[4], s[2], d[2])
        self.layer5 = _ResLayer(n_blocks[3], ch[4], ch[5], s[3], d[3], multi_grids)
        self.aspp_v3 = _ASPP_v3(ch[5], 256, atrous_rates)
        
        
        #self.add_module("layer1", _Stem(ch[0]))
        #self.add_module("layer2", _ResLayer(n_blocks[0], ch[0], ch[2], s[0], d[0]))
        #self.add_module("layer3", _ResLayer(n_blocks[1], ch[2], ch[3], s[1], d[1]))
        #self.add_module("layer4", _ResLayer(n_blocks[2], ch[3], ch[4], s[2], d[2]))
        #self.add_module(
        #    "layer5", _ResLayer(n_blocks[3], ch[4], ch[5], s[3], d[3], multi_grids)
        #)
        #self.add_module("aspp", _ASPP_v3(ch[5], 256, atrous_rates))
        #concat_ch = 256 * (len(atrous_rates) + 2)
        #self.add_module("fc1", _ConvBnReLU(concat_ch, 256, 1, 1, 0, 1)) removed
        #self.add_module("fc2", nn.Conv2d(256, n_classes, kernel_size=1)) removed
        
    def forward(self, image):
        image = self.layer1(image).to(device)
        image = self.layer2(image)
        output3 = self.layer3(image)
        output4 = self.layer4(output3)
        output5 = self.layer5(output4)
        output_aspp = self.aspp_v3(output5)
        return output3, output4, output5, output_aspp
        
        

class _ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling (ASPP)
    """

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        for i, rate in enumerate(rates):
            self.add_module(
                "c{}".format(i),
                nn.Conv2d(in_ch, out_ch, 3, 1, padding=rate, dilation=rate, bias=True),
            )

        for m in self.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return sum([stage(x) for stage in self.children()])

    
    
class DeepLabV2(nn.Sequential):
    """
    DeepLab v2: Dilated ResNet + ASPP
    Output stride is fixed at 8
    """

    def __init__(self, n_classes, n_blocks, atrous_rates):
        super(DeepLabV2, self).__init__()
        
        ch = [64 * 2 ** p for p in range(6)]
        self.layer1 = _Stem(ch[0])
        self.layer2 = _ResLayer(n_blocks[0], ch[0], ch[2], 1, 1)
        self.layer3 = _ResLayer(n_blocks[1], ch[2], ch[3], 2, 1)
        self.layer4 = _ResLayer(n_blocks[2], ch[3], ch[4], 1, 2)
        self.layer5 = _ResLayer(n_blocks[3], ch[4], ch[5], 1, 4)
        #self.aspp = _ASPP(ch[5], n_classes, atrous_rates) - removed aspp layer
        
        # self.add_module("layer1", _Stem(ch[0]))
        # self.add_module("layer2", _ResLayer(n_blocks[0], ch[0], ch[2], 1, 1))
        # self.add_module("layer3", _ResLayer(n_blocks[1], ch[2], ch[3], 2, 1))
        # self.add_module("layer4", _ResLayer(n_blocks[2], ch[3], ch[4], 1, 2))
        # self.add_module("layer5", _ResLayer(n_blocks[3], ch[4], ch[5], 1, 4))
        # self.add_module("aspp", _ASPP(ch[5], n_classes, atrous_rates)) - removed aspp layer

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, _ConvBnReLU.BATCH_NORM):
                m.eval()
                
    def forward(self, image):
        image = self.layer1(image).to(device)
        image = self.layer2(image)
        output3 = self.layer3(image)
        output4 = self.layer4(output3)
        output5 = self.layer5(output4)
        return output3, output4, output5
    

try:
    from encoding.nn import SyncBatchNorm

    _BATCH_NORM = SyncBatchNorm
except:
    _BATCH_NORM = nn.BatchNorm2d

_BOTTLENECK_EXPANSION = 4


class _ConvBnReLU(nn.Sequential):
    """
    Cascade of 2D convolution, batch norm, and ReLU.
    """

    BATCH_NORM = _BATCH_NORM

    def __init__(
        self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True
    ):
        super(_ConvBnReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
            ),
        )
        self.add_module("bn", _BATCH_NORM(out_ch, eps=1e-5, momentum=1 - 0.999))

        if relu:
            self.add_module("relu", nn.ReLU())


class _Bottleneck(nn.Module):
    """
    Bottleneck block of MSRA ResNet.
    """

    def __init__(self, in_ch, out_ch, stride, dilation, downsample):
        super(_Bottleneck, self).__init__()
        mid_ch = out_ch // _BOTTLENECK_EXPANSION
        self.reduce = _ConvBnReLU(in_ch, mid_ch, 1, stride, 0, 1, True)
        self.conv3x3 = _ConvBnReLU(mid_ch, mid_ch, 3, 1, dilation, dilation, True)
        self.increase = _ConvBnReLU(mid_ch, out_ch, 1, 1, 0, 1, False)
        self.shortcut = (
            _ConvBnReLU(in_ch, out_ch, 1, stride, 0, 1, False)
            if downsample
            else nn.Identity()
        )

    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        h += self.shortcut(x)
        return F.relu(h)


class _ResLayer(nn.Sequential):
    """
    Residual layer with multi grids
    """

    def __init__(self, n_layers, in_ch, out_ch, stride, dilation, multi_grids=None):
        super(_ResLayer, self).__init__()

        if multi_grids is None:
            multi_grids = [1 for _ in range(n_layers)]
        else:
            assert n_layers == len(multi_grids)

        # Downsampling is only in the first block
        for i in range(n_layers):
            self.add_module(
                "block{}".format(i + 1),
                _Bottleneck(
                    in_ch=(in_ch if i == 0 else out_ch),
                    out_ch=out_ch,
                    stride=(stride if i == 0 else 1),
                    dilation=dilation * multi_grids[i],
                    downsample=(True if i == 0 else False),
                ),
            )


class _Stem(nn.Sequential):
    """
    The 1st conv layer.
    Note that the max pooling is different from both MSRA and FAIR ResNet.
    """

    def __init__(self, out_ch):
        super(_Stem, self).__init__()
        self.add_module("conv1", _ConvBnReLU(3, out_ch, 7, 2, 3, 1))
        self.add_module("pool", nn.MaxPool2d(3, 2, 0, ceil_mode=True))


class ResNet(nn.Sequential):
    def __init__(self, n_classes, n_blocks):
        super(ResNet, self).__init__()
        ch = [64 * 2 ** p for p in range(6)]
        self.add_module("layer1", _Stem(ch[0]))
        self.add_module("layer2", _ResLayer(n_blocks[0], ch[0], ch[2], 1, 1))
        self.add_module("layer3", _ResLayer(n_blocks[1], ch[2], ch[3], 2, 1))
        self.add_module("layer4", _ResLayer(n_blocks[2], ch[3], ch[4], 2, 1))
        self.add_module("layer5", _ResLayer(n_blocks[3], ch[4], ch[5], 2, 1))
        self.add_module("pool5", nn.AdaptiveAvgPool2d(1))
        self.add_module("flatten", nn.Flatten())
        self.add_module("fc", nn.Linear(ch[5], n_classes))
        
        
