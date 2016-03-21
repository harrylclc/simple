require 'hdf5'
require 'rnn'
require 'optim'

cmd = torch.CmdLine()
cmd:option('-data', '/data_giles/cul226/simple/preprocessed/newsla_Google.hdf5', 'training data and word2vec data')
cmd:option('-gpu', 0, 'use gpu')
cmd:option('-gpuid', 1, 'gpu id')

function load_data()
    local w2v
    local dataX, dataY
    print('loading data...')
    local f = hdf5.open(opt.data, 'r')
    w2v = f:read('w2v'):all()
    dataX = f:read('data_x'):all()
    dataY = f:read('data_y'):all()
    print('done')
    return dataX, dataY, w2v
end

--[[ Forward coupling: Copy encoder cell and output to decoder LSTM ]]--
function forwardConnect(encLSTM, decLSTM)
   decLSTM.userPrevOutput = nn.rnn.recursiveCopy(decLSTM.userPrevOutput, encLSTM.outputs[opt.seqLen])
   decLSTM.userPrevCell = nn.rnn.recursiveCopy(decLSTM.userPrevCell, encLSTM.cells[opt.seqLen])
end

--[[ Backward coupling: Copy decoder gradients to encoder LSTM ]]--
function backwardConnect(encLSTM, decLSTM)
   encLSTM.userNextGradCell = nn.rnn.recursiveCopy(encLSTM.userNextGradCell, decLSTM.userGradPrevCell)
   encLSTM.gradPrevOutput = nn.rnn.recursiveCopy(encLSTM.gradPrevOutput, decLSTM.userGradPrevOutput)
end


-- options
opt = cmd:parse(arg)
opt.batchSize = 32
opt.niter = 1
opt.learningRate = 0.001
opt.momentum = 0.9

if opt.gpu == 1 then
    require 'cutorch'
    cutorch.setDevice(opt.gpuid + 1)
    require 'cunn'
    require 'cudnn'
end

local w2v
local dataX, dataY
dataX, dataY, w2v = load_data()

opt.vocabSize = w2v:size(1)
opt.vecSize = w2v:size(2)
opt.seqLen = dataX:size(2)

-- reverse input sequences
local dataXRev = torch.Tensor(dataX:size())
for i = 1, dataX:size(2) do
    dataXRev:select(2, i):copy(dataX:select(2, dataX:size(2) + 1 - i))
end

-- shift output sequences
local dataYShift = torch.cat(dataY:select(2, dataY:size(2)), dataY:sub(1, -1, 1, -2), 2)

if opt.gpu == 1 then
    dataY = dataY:float():cuda()
    dataXRev = dataXRev:float():cuda()
    dataYShitf = dataYShift:float():cuda()
end

-- pre-trained w2v
local lookup = nn.LookupTable(opt.vocabSize, opt.vecSize)
-- w2v = w2v:cuda()
lookup.weight:copy(w2v)
lookup.weight[1]:zero()

-- encoder
local enc = nn.Sequential()
enc:add(lookup)
enc:add(nn.SplitTable(1, 2))
local encLSTM = nn.LSTM(opt.vecSize, opt.vecSize)
enc:add(nn.Sequencer(encLSTM))
enc:add(nn.SelectTable(-1))

-- decoder
local dec = nn.Sequential()
dec:add(lookup)
dec:add(nn.SplitTable(1, 2))
local decLSTM = nn.LSTM(opt.vecSize, opt.vecSize)
dec:add(nn.Sequencer(decLSTM))
dec:add(nn.Sequencer(nn.Linear(opt.vecSize, opt.vocabSize)))
dec:add(nn.Sequencer(nn.LogSoftMax()))

local model = nn.Container():add(enc):add(dec)
local criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

params, gradParams = model:getParameters()

-- training
local trainSize = dataX:size(1)
local numBatches = math.floor(trainSize / opt.batchSize)

sgdState = {
    learningRate = opt.learningRate,
    momentum = opt.momentum,
    learningRateDecay = 5e-7
}

local toTable = nn.SplitTable(1, 1)
local zeroTensor = torch.Tensor(opt.batchSize, opt.vecSize):zero()

if opt.gpu == 1 then
    model:cuda()
    criterion:cuda()
    toTable:cuda()
    zeroTensor = zeroTensor:cuda()
end

for i = 1, opt.niter do
    print('Iter: ' .. i)
    local shuffle = torch.randperm(numBatches)
    for j = 1, shuffle:size(1) do
        print(j)
        local st = (shuffle[j] - 1) * opt.batchSize + 1
        local batchSize = math.min(opt.batchSize, trainSize - st + 1)
        print('start:', st, 'batchSize', batchSize)
        print('before narrow')
        local encInSeq = dataXRev:narrow(1, st, batchSize)
        local decInSeq = dataYShift:narrow(1, st, batchSize)
        local decOutSeq = dataY:narrow(1, st, batchSize)
        print('after narrow')
        --if opt.gpu == 1 then
        --    encInSeq = torch.Tensor(batchSize, encInSeq:size(2)):copy(encInSeq):float():cuda()
        --    decInSeq = decInSeq:float():cuda()
        --    decOutSeq = decOutSeq:float():cuda()
        --end
        print('before totable')
        decOutSeq = toTable:forward(decOutSeq)
        print('after totable')

        local feval = function(x)
            print('before gc')
            collectgarbage()
            print('after gc')
            if x ~= params then
                params:copy(x)
            end

            gradParams:zero()
            print('forward ...')
            -- forward
            local encOut = enc:forward(encInSeq)
            forwardConnect(encLSTM, decLSTM)
            local decOut = dec:forward(decInSeq)
            print('forward done')

            local err = criterion:forward(decOut, decOutSeq)
            print(string.format("NLL err = %f ", err))

            -- backward
            local gradOutput = criterion:backward(decOut, decOutSeq)
            dec:backward(decInSeq, gradOutput)
            backwardConnect(encLSTM, decLSTM)
            enc:backward(encInSeq, zeroTensor)

            return err, gradParams
        end

        optim.sgd(feval, params, sgdState)
    end
end

