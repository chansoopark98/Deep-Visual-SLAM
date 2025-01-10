import tensorflow as tf
import vio.model.resnet_original as resnet
import torchvision.models as models
import torch
import numpy as np
import matplotlib.pyplot as plt
import PIL

model_name = '18'
inputs = tf.keras.Input((None, None,3))
if model_name ==  '18':
    resnet_torch = models.resnet18(pretrained=True)
    resnet_tf = resnet.resnet_18(inputs)
elif model_name == '34':
    resnet_torch = models.resnet34(pretrained=True)
    resnet_tf = resnet.resnet_34(inputs)

model = tf.keras.Model(inputs, resnet_tf)
tf.keras.utils.plot_model(model, to_file=f'assets/resnet{model_name}_model.png', show_layer_names=True)

# place all variables in list
tf_layer_names = [layer.name for layer in model.layers]
torch_layer_names = []
for name, module in resnet_torch.named_modules():
    torch_layer_names.append(name)

tf_layer_names = [layer for layer in tf_layer_names if layer in torch_layer_names]

# loop over all layers from Pytorch model also found in tensorflow model and port weights
for layer in tf_layer_names:
    if 'conv' in layer:
        tf_conv = model.get_layer(layer)
        weights = resnet_torch.state_dict()[layer+'.weight'].numpy()
        weights_list = [weights.transpose((2, 3, 1, 0))]
        if len(tf_conv.weights) == 2:
            bias = resnet_torch.state_dict()[layer+'.bias'].numpy()
            weights_list.append(bias)
        tf_conv.set_weights(weights_list)
    elif 'bn' in layer:
        tf_bn = model.get_layer(layer)
        gamma = resnet_torch.state_dict()[layer+'.weight'].numpy()
        beta = resnet_torch.state_dict()[layer+'.bias'].numpy()
        mean = resnet_torch.state_dict()[layer+'.running_mean'].numpy()
        var = resnet_torch.state_dict()[layer+'.running_var'].numpy()
        bn_list = [gamma, beta, mean, var]
        tf_bn.set_weights(bn_list)
    elif 'downsample.0' in layer:
        tf_downsample = model.get_layer(layer)
        weights = resnet_torch.state_dict()[layer+'.weight'].numpy()
        weights_list = [weights.transpose((2, 3, 1, 0))]
        if len(tf_downsample.weights) == 2:
            bias = resnet_torch.state_dict()[layer+'.bias'].numpy()
            weights_list.append(bias)
        tf_downsample.set_weights(weights_list)
    elif 'downsample.1' in layer:
        tf_downsample = model.get_layer(layer)
        gamma = resnet_torch.state_dict()[layer+'.weight'].numpy()
        beta = resnet_torch.state_dict()[layer+'.bias'].numpy()
        mean = resnet_torch.state_dict()[layer+'.running_mean'].numpy()
        var = resnet_torch.state_dict()[layer+'.running_var'].numpy()
        bn_list = [gamma, beta, mean, var] # [gamma, beta, mean, var]
        tf_downsample.set_weights(bn_list)
    elif 'fc' in layer:
        tf_fc = model.get_layer(layer)
        weights = resnet_torch.state_dict()[layer+'.weight'].numpy() 
        weights_list = [weights.transpose((1, 0))]
        if len(tf_fc.weights) == 2:
            bias = resnet_torch.state_dict()[layer+'.bias'].numpy()
            weights_list.append(bias)
        tf_fc.set_weights(weights_list)
    else:
        print('No parameters found for {}'.format(layer))

img = np.expand_dims(np.array(PIL.Image.open('./assets/cat.jpg', 'r')), 0).astype(np.float32)
img_torch = torch.tensor(img.transpose((0, 3, 1, 2)))

# Feed in image of cat to both models
tf_output = model.predict(img)
resnet_torch.eval()
resnet_torch.eval()
torch_output = resnet_torch(img_torch)

print(np.argmax(tf_output))
print(np.argmax(torch_output.detach().numpy()))

# Find max difference between model outputs
max_diff = np.amax(np.abs(tf_output - torch_output.detach().numpy()))
mean_diff = np.mean(np.abs(tf_output - torch_output.detach().numpy()))
print('Max difference: {}'.format(max_diff))
print('Mean difference: {}'.format(mean_diff))

# Plot the scores for each class for each model
plt.figure()
plt.plot(tf_output[0, :], 'r', label='Tensorflow')
plt.plot(torch_output.detach().numpy()[0, :], 'g', label='Pytorch')
plt.show()

# Save model weights
#model.save('ResNet/resnet') # Does not appear to be  workign in tensorflow2.0.0b1
model.save_weights('assets/weights/resnet'+model_name+'.h5')

def plot_feature_map(torch_output, tf_output, map_idx):
    if map_idx < torch_output.shape[-1]:      
        plt.figure()
        plt.subplot(121)
        plt.imshow(torch_output.detach().numpy()[0, :, :, map_idx])
        plt.subplot(122)
        plt.imshow(tf_output[0, :, :, map_idx])
        plt.show()

class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = torch.tensor(output,requires_grad=False)
    def close(self):
        self.hook.remove()



layers = [resnet_torch.conv1, resnet_torch.bn1, resnet_torch.maxpool, resnet_torch.layer1[0].conv1] 

layer_idx = 0
layer = tf_layer_names[layer_idx]
activations = SaveFeatures(layers[layer_idx])
torch_output = resnet_torch(img_torch)
torch_layer_output = activations.features.permute(0, 2, 3, 1)
intermediate_model = tf.keras.Model(inputs, model.get_layer(layer).output)
tf_layer_output = intermediate_model.predict(img) 
max_diff = np.amax(np.abs(tf_layer_output - torch_layer_output.detach().numpy()))
print(layer, ':', max_diff)

plot_feature_map(torch_layer_output, tf_layer_output, 10)