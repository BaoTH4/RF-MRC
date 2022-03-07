import logging
import torch
import torch.nn.functional as F

def get_logger(filename,verbosity=1,name=None):
  level_dict={0:logging.DEBUG,1:logging.INFO,2:logging.WARNING}
  formatter=logging.Formatter(
      "%(asctime)s - %(filename)s - line:%(lineno)d - %(levelname)s:\t%(message)s"
  )
  logger=logging.getLogger(name)
  logger.setLevel(level_dict[verbosity])

  fh=logging.FileHandler(filename,'w')
  fh.setFormatter(formatter)
  logger.addHandler(fh)

  sh=logging.StreamHandler()
  sh.setFormatter(formatter)
  logger.addHandler(sh)

  return logger

##This cell contain function to resize tensor for Cross Entropy loss
def normalize_size(tensor):
  ##Hàm chuẩn hóa size tensor lấy lại theo code B-MRC
    if len(tensor.size()) == 3:
        tensor = tensor.contiguous().view(-1, tensor.size(2))
    elif len(tensor.size()) == 2:
        tensor = tensor.contiguous().view(-1)

    return tensor

def calculate_A_O_loss(targets,logits,ifgpu=True):
  ##Hàm này tính loss cho aspect hay opinion
  ##Theo thống kê nhãn 0 nhiều gấp 8 lần nhãn 1 và gấp 16 lần nhãn 2 nên ta sẽ đánh weight theo thứ tự
  ##[1,2,4]
  gold_targets=normalize_size(targets)
  pred=normalize_size(logits)
  if ifgpu==True:
      weight = torch.tensor([1, 2, 4]).float().cuda()
  else:
      weight = torch.tensor([1, 2, 4]).float()
  loss=F.cross_entropy(pred,gold_targets.long(),ignore_index=-1,weight=weight)
  return loss