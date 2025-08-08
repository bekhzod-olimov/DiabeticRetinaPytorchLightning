from torchvision import transforms as T

def get_fts(mean, std):  return T.Compose([ T.RandomRotation(10), T.ToTensor(), T.Normalize(mean=mean, std=std) ]), T.Compose([ T.ToTensor(), T.Normalize(mean=mean, std=std) ])