import sys
import cv2
import torch
import numpy as np
from math import ceil
from torch.autograd import Variable
from scipy.ndimage import imread
import models
import argparse
from pathlib import Path
from tqdm import tqdm
import os

"""
Contact: Deqing Sun (deqings@nvidia.com); Zhile Ren (jrenzhile@gmail.com)
"""
def writeFlowFile(filename,uv):
	"""
	According to the matlab code of Deqing Sun and c++ source code of Daniel Scharstein  
	Contact: dqsun@cs.brown.edu
	Contact: schar@middlebury.edu
	"""
	TAG_STRING = np.array(202021.25, dtype=np.float32)
	if uv.shape[2] != 2:
		sys.exit("writeFlowFile: flow must have two bands!");
	H = np.array(uv.shape[0], dtype=np.int32)
	W = np.array(uv.shape[1], dtype=np.int32)
	with open(filename, 'wb') as f:
		f.write(TAG_STRING.tobytes())
		f.write(W.tobytes())
		f.write(H.tobytes())
		f.write(uv.tobytes())

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-l", "--list", dest="list", help="list of pairs", default="/input/list.txt")
  parser.add_argument("-o", "--out", dest="out", help="out folder", default="/1000x600/out")
  args = parser.parse_args()

  out_folder = Path(args.out)

  if not os.path.exists(args.out):
    os.makedirs(args.out)

  input_folder = Path(args.list)

  f = open(args.list)
  for i in tqdm(f.readlines()):
    first_image_path = Path(i.strip().split(",")[0])
    second_image_path = Path(i.strip().split(",")[1])
    im1_fn = str(input_folder.parent / first_image_path)
    im2_fn = str(input_folder.parent / second_image_path)
    flow_fn = str(out_folder / first_image_path.stem) + '.flo'

    pwc_model_fn = './pwc_net.pth.tar';

    im_all = [cv2.merge(( cv2.imread(img, -1), cv2.imread(img, -1), cv2.imread(img, -1) )) for img in [im1_fn, im2_fn]]
    im_all = [im[:, :, :3] for im in im_all]

    # rescale the image size to be multiples of 64
    divisor = 64.
    H = im_all[0].shape[0]
    W = im_all[0].shape[1]

    H_ = int(ceil(H/divisor) * divisor)
    W_ = int(ceil(W/divisor) * divisor)
    for i in range(len(im_all)):
    	im_all[i] = cv2.resize(im_all[i], (W_, H_))

    for _i, _inputs in enumerate(im_all):
    	im_all[_i] = im_all[_i][:, :, ::-1]
    	im_all[_i] = 1.0 * im_all[_i]/65536.0
    	
    	im_all[_i] = np.transpose(im_all[_i], (2, 0, 1))
    	im_all[_i] = torch.from_numpy(im_all[_i])
    	im_all[_i] = im_all[_i].expand(1, im_all[_i].size()[0], im_all[_i].size()[1], im_all[_i].size()[2])	
    	im_all[_i] = im_all[_i].float()
        
    im_all = torch.autograd.Variable(torch.cat(im_all,1).cuda(), volatile=True)

    net = models.pwc_dc_net(pwc_model_fn)
    net = net.cuda()
    net.eval()

    flo = net(im_all)
    flo = flo[0] * 20.0
    flo = flo.cpu().data.numpy()

    # scale the flow back to the input size 
    flo = np.swapaxes(np.swapaxes(flo, 0, 1), 1, 2) # 
    u_ = cv2.resize(flo[:,:,0],(W,H))
    v_ = cv2.resize(flo[:,:,1],(W,H))
    u_ *= W/ float(W_)
    v_ *= H/ float(H_)
    flo = np.dstack((u_,v_))

    writeFlowFile(flow_fn, flo)
  f.close()
