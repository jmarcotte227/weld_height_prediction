% parameters
num_layers = 105;
num_segs = 46;
trim = 1;

% load raw dataset
load('raw/bent_tube_dataset');

part = CL_cold;
layers = fieldnames(part);

for idx=1:length(layers)
    layer_data = part.(layers{idx});
    % shift u and y to line up such that u is the next segments cmd_vel
    u = layer_data.vel_set'
    u = u(2+trim:end-trim)
    y = [layer_data.dh', layer_data.avg_temp', layer_data.vel_calc']
    y = y(1+trim:end-trim-1,:)
    
    % create iddata object
    layer_data_struct = iddata(y,u);
    if idx ==1
        data = layer_data_struct;
    else
        data = merge(data, layer_data_struct);
    end
end

save('mat_processed/train_data.mat', 'data')
