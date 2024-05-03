# Plot results

from matplotlib.colors import ListedColormap

custom_colors = [[0, 0, 0], 
                [1, 0, 0],  
                [1, 0.75, 0],   
                [0, 0.3, 0.7], 
                [0.6, 0.6, 0.6],   
                [0, 0.5, 1],   
                [0, 0.85, 0],   
                [1, 0, 1], 
                [0.7, 0.7, 0.9], 
                [0.1, 0.55, 0.3]]

color_names = ['background', 'building-flooded', 'building non-flooded', 'road flooded', 'road non-flooded', 'water', 'tree', 'vehicle', 'pool', 'grass']
custom_cmap = ListedColormap(custom_colors)

def unnormalize(tensor, mean = [-0.2417,  0.8531,  0.1789], std = [0.9023, 1.1647, 1.3271]):
    """
    Unnormalizes a tensor given mean and standard deviation.
    
    Args:
        tensor (torch.Tensor): Input tensor to be unnormalized.
        mean (float or sequence): Mean value(s) for unnormalization.
        std (float or sequence): Standard deviation value(s) for unnormalization.
        
    Returns:
        torch.Tensor: Unnormalized tensor.
    """
    if not torch.is_tensor(tensor):
        raise TypeError("Input tensor should be a torch.Tensor")
    
    mean = torch.tensor(mean, device=tensor.device, dtype=tensor.dtype)
    std = torch.tensor(std, device=tensor.device, dtype=tensor.dtype)
    
    unnormalized_tensor = tensor * std + mean
    return unnormalized_tensor

model.eval()

val_transform = T.Compose([
    T.Resize((opt.resize_height, opt.resize_width)),
    T.ToTensor(),
    T.Normalize(mean=opt.mean, std=opt.std)
])
val_target_transform = T.Compose([
    T.Resize((opt.resize_height, opt.resize_width)),
    T.PILToTensor(),
])

im1 = Image.open('./data/val/6651.jpg')
im2 = Image.open('./data/val/7488.jpg')
im3 = Image.open('./data/val/6734.jpg')

mask1 = Image.open('./data/val/6651_lab.png')
mask2 = Image.open('./data/val/7488_lab.png')
mask3 = Image.open('./data/val/6734_lab.png')

im1 = val_transform(im1)
im2 = val_transform(im2)
im3 = val_transform(im3)

mask1 = val_target_transform(mask1)
mask2 = val_target_transform(mask2)
mask3 = val_target_transform(mask3)

if torch.cuda.is_available():
    im1 = im1.cuda(non_blocking=True)
    #mask1 = mask1.cuda(non_blocking=True)
    im2 = im2.cuda(non_blocking=True)
    #mask2 = mask2.cuda(non_blocking=True)
    im3 = im3.cuda(non_blocking=True)
    #mask3 = mask3.cuda(non_blocking=True)
    
if opt.name_net == 'deeplab': 
    pred1 = model(im1[None,:,:,:])['out']
    pred2 = model(im2[None,:,:,:])['out']
    pred3 = model(im3[None,:,:,:])['out']
else:
    pred1 = model(im1[None,:,:,:])
    pred2 = model(im2[None,:,:,:])
    pred3 = model(im3[None,:,:,:])
    
pred1 = torch.squeeze(pred1)
pred1 = pred1.argmax(0).squeeze()
pred1 = pred1.cpu().detach().numpy()
pred2 = torch.squeeze(pred2)
pred2 = pred2.argmax(0).squeeze()
pred2 = pred2.cpu().detach().numpy()
pred3 = torch.squeeze(pred3)
pred3 = pred3.argmax(0).squeeze()
pred3 = pred3.cpu().detach().numpy()

fig, ax =  plt.subplots(3, 3, figsize=(18, 18))
ax[0][0].set_title('Image')
ax[0][1].set_title('Label')
ax[0][2].set_title('Prediction')
ax[1][0].set_title('Image')
ax[1][1].set_title('Label')
ax[1][2].set_title('Prediction')
ax[2][0].set_title('Image')
ax[2][1].set_title('Label')
ax[2][2].set_title('Prediction')
ax[0][0].imshow(np.squeeze(unnormalize(np.transpose(im1.squeeze().cpu(),(1,2,0)))))
ax[0][1].imshow(mask1.squeeze(), cmap = custom_cmap, vmin = 0, vmax = 9)
ax[0][2].imshow(pred1.squeeze(), cmap = custom_cmap, vmin = 0, vmax = 9)
ax[1][0].imshow(np.squeeze(unnormalize(np.transpose(im2.squeeze().cpu(),(1,2,0)))))
ax[1][1].imshow(mask2.squeeze(), cmap = custom_cmap, vmin = 0, vmax = 9)
ax[1][2].imshow(pred2.squeeze(), cmap = custom_cmap, vmin = 0, vmax = 9)
ax[2][0].imshow(np.squeeze(unnormalize(np.transpose(im3.squeeze().cpu(),(1,2,0)))))
ax[2][1].imshow(mask3.squeeze(), cmap = custom_cmap, vmin = 0, vmax = 9)
ax[2][2].imshow(pred3.squeeze(), cmap = custom_cmap, vmin = 0, vmax = 9)


