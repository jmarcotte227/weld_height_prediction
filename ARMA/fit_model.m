% load processed data
load('../data/mat_processed/train_data.mat')

% fit arx model
na = eye(3);
nb = ones(3,1);
% check this transport delay, I think this is zero
nk = zeros(3,1);

orders = [na,nb,nk];

% [Lambda, R] = arxRegul(data, orders);
% opt = arxOptions();
% opt.Regularizatoin.Lambda=Lambda;
% opt.Regularization.R=R;

model = arx(data, orders);

