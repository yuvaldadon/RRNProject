def compute_iou(output_up, mask_pass): # expects (1, 1, 320, 320) tensors
    outputs = (output_up >= 0).int()
    labels = mask_pass.int()
    outputs = outputs.squeeze(1)
    labels = labels.squeeze(1)
    intersect = (outputs*labels).sum(2).sum(1).item()
    union = (outputs+labels).sum(2).sum(1).item()
    real_intersect = (intersect+0.001)
    real_union = (union-intersect+0.001)
    iou = real_intersect/real_union
    return iou, real_intersect, real_union

def compute_iou_np(predict, mask): # expects (1, 1, 320, 320) np float arrays
    predict = np.squeeze(predict.astype(int))
    mask = np.squeeze(mask.astype(int))
    intersect = (predict*mask).sum(0).sum(0)
    union = (predict+mask).sum(0).sum(0)
    real_intersect = (intersect+0.001)
    real_union = (union-intersect+0.001)
    iou = real_intersect/real_union
    return iou, real_intersect, real_union

def visualize(image, mask, sent, show=None, save=None): #expects image=(320,320,3), mask=(320,320) tensor
    image = image / 1.3
    image[:,:,0] += torch.squeeze(mask.int()) * 120   # apply red mask on img
    image = torch.clamp(image,0,255)   # clip to [0,255]
    image = image.to(torch.uint8)
    if show:
        plt.imshow(image) #int() to consider range 0-255
        plt.title(sent)
        plt.show()
    if save:
        plt.imsave(save, image.numpy())
      
def visualize_sep(image, sent, show=None, save=None): #expects image=(320,320,3), mask=(320,320) tensor
    image[:,:,0] += torch.squeeze(mask.int()) * 120   # apply red mask on img
    image = torch.clamp(image,0,255)   # clip to [0,255]
    image = image.to(torch.uint8)
    if show:
        plt.imshow(image) #int() to consider range 0-255
        plt.title(sent)
        plt.show()
    if save:
        plt.imsave(save, image.numpy())
        
def dcrf_calc(up_val, ubyte_im, H, W, mask_h, mask_w):
    sigm_val = torch.sigmoid(up_val).numpy()
    sigm_val = np.squeeze(sigm_val) # (1,320,320,1) => (320, 320)
    d = densecrf.DenseCRF2D(W, H, 2)
    U = np.expand_dims(-np.log(sigm_val), axis=0) # (1,320,320) float (sigm_val with -log())
    U_ = np.expand_dims(-np.log(1 - sigm_val), axis=0) # (1,320,320)
    unary = np.concatenate((U_, U), axis=0) # (2, 320, 320) (concat)
    unary = unary.reshape((2, -1)) # (2, 102400) (combine)
    d.setUnaryEnergy(unary) # (2, 102400)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=20, srgb=3, rgbim=ubyte_im, compat=10)
    Q = d.inference(5)
    pred_raw_dcrf = np.argmax(Q, axis=0).reshape((H, W)).astype(np.float32) # (320, 320) float (1s or 0s)
    predicts_dcrf = resize_and_crop(pred_raw_dcrf, mask_h, mask_w) # (320, 320) float (same)
    return predicts_dcrf



def resize_and_pad(im, input_h, input_w):
    # Resize and pad im to input_h x input_w size
    im_h, im_w = im.shape[:2]
    scale = min(input_h / im_h, input_w / im_w)
    resized_h = int(np.round(im_h * scale))
    resized_w = int(np.round(im_w * scale))
    pad_h = int(np.floor(input_h - resized_h) / 2)
    pad_w = int(np.floor(input_w - resized_w) / 2)

    resized_im = skimage.transform.resize(im, [resized_h, resized_w])
    if im.ndim > 2:
        new_im = np.zeros((input_h, input_w, im.shape[2]), dtype=resized_im.dtype)
    else:
        new_im = np.zeros((input_h, input_w), dtype=resized_im.dtype)
    new_im[pad_h:pad_h+resized_h, pad_w:pad_w+resized_w, ...] = resized_im

    return new_im


def resize_and_crop(im, input_h, input_w):
    # Resize and crop im to input_h x input_w size
    im_h, im_w = im.shape[:2]
    scale = max(input_h / im_h, input_w / im_w)
    resized_h = int(np.round(im_h * scale))
    resized_w = int(np.round(im_w * scale))
    crop_h = int(np.floor(resized_h - input_h) / 2)
    crop_w = int(np.floor(resized_w - input_w) / 2)

    resized_im = skimage.transform.resize(im, [resized_h, resized_w])
    if im.ndim > 2:
        new_im = np.zeros((input_h, input_w, im.shape[2]), dtype=resized_im.dtype)
    else:
        new_im = np.zeros((input_h, input_w), dtype=resized_im.dtype)
    new_im[...] = resized_im[crop_h:crop_h+input_h, crop_w:crop_w+input_w, ...]

    return new_im
        
        
        
def generate_spatial_batch(N, featmap_H, featmap_W):
    spatial_batch_val = np.zeros((N, featmap_H, featmap_W, 8), dtype=np.float32)
    for h in range(featmap_H):
        for w in range(featmap_W):
            xmin = w / featmap_W * 2 - 1
            xmax = (w+1) / featmap_W * 2 - 1
            xctr = (xmin+xmax) / 2
            ymin = h / featmap_H * 2 - 1
            ymax = (h+1) / featmap_H * 2 - 1
            yctr = (ymin+ymax) / 2
            spatial_batch_val[:, h, w, :] = \
                [xmin, ymin, xmax, ymax, xctr, yctr, 1/featmap_W, 1/featmap_H]
    return spatial_batch_val

#acc_all: % of total mask guessed right
#acc_pass: % of original True area guessed right
#acc_neg: % of original False area guessed right
def compute_accuracy(scores, labels): # both (1,1,40,40) torch
    is_pos = (labels != 0)
    is_neg = torch.logical_not(is_pos)
    num_pos = torch.sum(is_pos).item()
    num_neg = torch.sum(is_neg).item()
    num_all = num_pos + num_neg

    is_correct = torch.logical_xor(scores < 0, is_pos)
    accuracy_all = torch.sum(is_correct).item() / (num_all+0.0001)
    accuracy_pos = torch.sum(is_correct[is_pos]).item() / (num_pos+0.0001)
    accuracy_neg = torch.sum(is_correct[is_neg]).item() / (num_neg+0.0001)
    return accuracy_all, accuracy_pos, accuracy_neg

def compute_accuracy_test(predict, labels): # (1,1,40,40) torch
    is_pos = (labels != 0)
    is_neg = torch.logical_not(is_pos)
    num_pos = torch.sum(is_pos).item()
    num_neg = torch.sum(is_neg).item()
    num_all = num_pos + num_neg

    is_correct = torch.logical_xor(predict == 0, is_pos)
    accuracy_all = torch.sum(is_correct).item() / num_all
    accuracy_pos = torch.sum(is_correct[is_pos]).item() / (num_pos+0.00001)
    accuracy_neg = torch.sum(is_correct[is_neg]).item() / (num_neg+0.00001)
    return accuracy_all, accuracy_pos, accuracy_neg

def init_train_opt(opt):
    if opt.load_checkpoint:
        print("Loading Model:")
        for k,v in opt.items():
            print(k, '=', v)
            #if isinstance(v,list):
            #    print(k, '= [..,' + str(v[-1]) + ']') #last list value
    else:
        opt.train_loss = {}
        opt.train_acc = {}
        opt.train_acc_pos = {}
        opt.train_acc_neg = {}
        opt.train_output_dir = train_output_dir(opt.dataset)
    return opt


def save_log(opt, path):
    f = open(path, "w")
    for k, v in opt.items():
        f.write(str(k) + '=' + str(v) + '\n')
    f.close()
    
    
def train_output_dir(dataset):
    output_dir = './output/' + dataset + '_'  # output_dir='checkpoint/unc_i'
    i = 0
    while os.path.exists(output_dir + str(i)):
        i += 1
    output_dir = output_dir + str(i)
    os.makedirs(output_dir)
    return output_dir


def test_output_dir(test_dir, output_dir):
    if test_dir == None: # default to making 'test_i' dir, unless test_dir specified
        test_dir = os.path.join(opt.output_dir, 'test_')
        i = 0
        while os.path.exists(test_dir + str(i)):
            i += 1
        test_dir = test_dir + str(i)
        
    if not os.path.isdir(test_dir):
        os.makedirs(test_dir)
    return test_dir


def run_prefetch(prefetch_queue, folder_name, prefix, num_batch, shuffle):
    n_batch_prefetch = 0
    fetch_order = np.arange(num_batch)
    while True:
        # Shuffle the batch order for every epoch
        if n_batch_prefetch == 0 and shuffle:
            fetch_order = np.random.permutation(num_batch)

        # Load batch from file
        batch_id = fetch_order[n_batch_prefetch]
        save_file = os.path.join(folder_name, prefix+'_'+str(batch_id)+'.npz')
        npz_filemap = np.load(save_file)
        batch = dict(npz_filemap)
        npz_filemap.close()

        # add loaded batch to fetchqing queue
        prefetch_queue.put(batch, block=True)

        # Move to next batch
        n_batch_prefetch = (n_batch_prefetch + 1) % num_batch

class DataReader:
    def __init__(self, folder_name, prefix, shuffle=True, prefetch_num=8, max_batch=0):
        self.folder_name = folder_name
        self.prefix = prefix
        self.shuffle = shuffle
        self.prefetch_num = prefetch_num

        self.n_batch = 0
        self.n_epoch = 0

        # Search the folder to see the number of num_batch
        filelist = os.listdir(folder_name)
        num_batch = 0
        while (prefix + '_' + str(num_batch) + '.npz') in filelist:
            num_batch += 1
            if num_batch > max_batch and max_batch > 0:
                break
        if num_batch > 0:
            print('\nfound %d batches under %s with prefix "%s"' % (num_batch, folder_name, prefix))
        else:
            raise RuntimeError('\nno batches under %s with prefix "%s"' % (folder_name, prefix))
        self.num_batch = num_batch

        # Start prefetching thread
        self.prefetch_queue = queue.Queue(maxsize=prefetch_num)
        self.prefetch_thread = threading.Thread(target=run_prefetch,
            args=(self.prefetch_queue, self.folder_name, self.prefix,
                  self.num_batch, self.shuffle))
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()

    def read_batch(self, is_log = True):
        if is_log:
            print('data reader: epoch = %d, batch = %d / %d' % (self.n_epoch, self.n_batch, self.num_batch))

        # Get a batch from the prefetching queue
        if self.prefetch_queue.empty():
            #print('data reader: waiting for file input (IO is slow)...')
            pass
        batch = self.prefetch_queue.get(block=True)
        self.n_batch = (self.n_batch + 1) % self.num_batch
        self.n_epoch += (self.n_batch == 0)
        return batch