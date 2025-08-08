from torchvision import transforms as T

def get_fts(mean, std, im_size):  return T.Compose([ T.Resize((im_size, im_size)), T.RandomRotation(10), T.ToTensor(), T.Normalize(mean=mean, std=std) ]), T.Compose([ T.Resize((im_size, im_size)), T.ToTensor(), T.Normalize(mean=mean, std=std) ])