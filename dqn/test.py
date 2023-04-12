'''
structure
input: 4 x 84 x 84
conv1 = 32 filters, size = 8, stride = 4
conv2 = 64 filters, size = 3, stride = 1
conv3 = 64 filters, size = 3, stride = 1
fc = 512
out = action_dim
activation = relu

gamma = 0.99
lr = 2.5e-4
training done with 50 m steps

trick: use maxpool of the last two frames instead of last 4 frames.
'''