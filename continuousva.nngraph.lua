-- Joost van Amersfoort - <joost@joo.st>
require 'sys'
require 'torch'
require 'nn'
require 'nngraph'
require 'xlua'
require 'optim'
require 'gnuplot'

--Packages necessary for SGVB
require 'Reparametrize'
require 'GaussianCriterion'
require 'KLDCriterion'

--Custom Linear module to support different reset function
require 'LinearVA'

--For loading data files
require 'load'

data = loadfreyfaces('datasets/freyfaces.hdf5')

dim_input = data.train:size(2)
dim_hidden = 2
hidden_units_encoder = 200
hidden_units_decoder = 200

batchSize = 20

torch.manualSeed(1)

--The model
function make_encoder()
	local x = nn.Identity()()
	local z = nn.Linear(dim_input,hidden_units_encoder)(x)
	z = nn.SoftPlus()(z)
	local mu = nn.Linear(hidden_units_encoder,dim_hidden)(z)
	local si = nn.Linear(hidden_units_encoder,dim_hidden)(z)
	return nn.gModule({x},{mu,si})
end



--Encoding layer
encoder = make_encoder()

function make_va()
	local x = nn.Identity()()
	local z = nn.Linear(dim_input,hidden_units_encoder)(x)
	z = nn.SoftPlus()(z)
	local mu = nn.Linear(hidden_units_encoder,dim_hidden)(z)
	local si = nn.Linear(hidden_units_encoder,dim_hidden)(z)


	local va = nn.Reparametrize(dim_hidden)({mu,si})
	va = nn.Linear(dim_hidden, hidden_units_decoder)(va)
	va = nn.SoftPlus()(va)

	local one = nn.Sigmoid()(nn.Linear(hidden_units_decoder, dim_input)(va))
	local two = nn.Copy()(nn.Linear(hidden_units_decoder, dim_input)(va))

	local gaussian = nn.GaussianCriterion()({nn.Identity()({one,two}),x})
	local kld = nn.KLDCriterion()({nn.Identity()({mu,si}),x})
	return nn.gModule({x},{gaussian,kld,one,two})
end


va = make_va()

--Optimization criteria
--Gaussian = nn.GaussianCriterion()
--KLD = nn.KLDCriterion()

parameters, gradients = va:getParameters()

config = {
    learningRate = -0.01,
}

state = {}
zero = torch.zeros(batchSize,dim_input)
epoch = 0
while true do
    epoch = epoch + 1
    local lowerbound = 0
    local time = sys.clock()
    local shuffle = torch.randperm(data.train:size(1))

     --Make sure batches are always batchSize
    local N = data.train:size(1) - (data.train:size(1) % batchSize)
    local N_test = data.test:size(1) - (data.test:size(1) % batchSize)

    for i = 1, N, batchSize do
        xlua.progress(i+batchSize-1, data.train:size(1))

        local batch = torch.Tensor(batchSize,data.train:size(2))

        local k = 1
        for j = i,i+batchSize-1 do
            batch[k] = data.train[shuffle[j]]:clone()
            k = k + 1
        end

        local opfunc = function(x)
            if x ~= parameters then
                parameters:copy(x)
            end

            va:zeroGradParameters()

						local encode_img = batch[1]:reshape(28,20)
						gnuplot.figure(1)
						gnuplot.imagesc(encode_img,'encode')

            local err, KLDerr, f1, f2 = unpack(va:forward(batch))
						print(err)
						local decode1_img = f1[1]:reshape(28,20)
						local decode2_img = f2[1]:reshape(28,20)
						gnuplot.figure(2)
						gnuplot.imagesc(decode1_img,'decode1')
						gnuplot.figure(3)
						gnuplot.imagesc(decode2_img,'decode2')

            --local err = Gaussian:forward(f, batch)
            --local df_dw = Gaussian:backward(f, batch)
            va:backward(batch,{torch.ones(1),torch.ones(1),zero,zero})

            --local KLDerr = KLD:forward(va:get(1).output, batch)
            --local de_dw = KLD:backward(va:get(1).output, batch)
            --encoder:backward(batch,de_dw)

            local lowerbound = err  + KLDerr

            return lowerbound, gradients
        end

        x, batchlowerbound = optim.adagrad(opfunc, parameters, config, state)

        lowerbound = lowerbound + batchlowerbound[1]
    end

    print("\nEpoch: " .. epoch .. " Lowerbound: " .. lowerbound/N .. " time: " .. sys.clock() - time)

    if lowerboundlist then
        lowerboundlist = torch.cat(lowerboundlist,torch.Tensor(1,1):fill(lowerbound/N),1)
    else
        lowerboundlist = torch.Tensor(1,1):fill(lowerbound/N)
    end

    if epoch % 2 == 0 then
        torch.save('save/parameters.t7', parameters)
        torch.save('save/state.t7', state)
        torch.save('save/lowerbound.t7', torch.Tensor(lowerboundlist))
    end
end
