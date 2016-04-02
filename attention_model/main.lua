require 'optim'
require 'rnn'
require '../utils/misc'

local opts = require '../utils/opts'
opt = opts.parse(arg)

if opt.gpuid >= 0 then
    require 'cutorch'
    cutorch.setDevice(opt.gpuid + 1)
    require 'cunn'
    require 'cudnn'
else
    torch.setnumthreads(opt.threads)
    print('<torch> set # threads to ' .. torch.getnumthreads())
end

local DL = require '../utils/DataLoader'
local loader = DL.create(opt.data, opt.batchSize)
local wd2id = DL.loadVocab(opt.vocab)
local id2wd = {}
for wd, id in pairs(wd2id) do
    id2wd[id] = wd
end
local vocabSize = loader.w2v:size(1)
local vecSize = loader.w2v:size(2)

local encrnn, decrnn
if opt.useBN then
    require 'GRUAttentionBN'
    require 'GRUBN'
    encrnn = nn.GRUBN
    decrnn = nn.GRUAttentionBN
else
    require 'GRUAttention'
    encrnn = nn.GRU
    decrnn = nn.GRUAttention
end

local checkpoint
if string.len(opt.init_from) > 0 then
    print('init model from', opt.init_from)
    checkpoint = torch.load(opt.init_from)
    opt.hiddenSize = checkpoint.opt.hiddenSize
end

---------------------- Model -----------------------
-- encoder
local lookup1 = nn.LookupTable(vocabSize, vecSize)
lookup1.weight:copy(loader.w2v)  -- use pretrained w2v
lookup1.weight[1]:zero()

local enc = nn.Sequential()
enc:add(lookup1)
enc:add(nn.SplitTable(1,2))
local encgru = encrnn(vecSize, opt.hiddenSize)
enc:add(nn.BiSequencer(encgru))
enc:add(nn.ConvertTable())
enc:add(nn.Transpose({1,2}))

-- decoder
local decLookup = nn.Sequential()
local lookup2 = lookup1:clone()
decLookup:add(lookup2)
decLookup:add(nn.SplitTable(1,2))
local decgru = decrnn(vecSize, opt.hiddenSize, opt.hiddenSize*2, vocabSize)

-- init decoder
local getMean = nn.Sequential()
getMean:add(nn.Mean(2))
getMean:add(nn.Narrow(2, 1, opt.hiddenSize))

-- container model
local model = nn.Container():add(enc):add(decLookup):add(decgru)

-- criterion
local criterion = nn.ClassNLLCriterion()

-- zerotensor
local zeroOutputGrad = torch.Tensor()
local zeroAttentionGrad = torch.Tensor()

if opt.gpuid >= 0 then
    model:cuda()
    criterion:cuda()
    getMean:cuda()
    zeroOutputGrad = zeroOutputGrad:cuda()
    zeroAttentionGrad = zeroAttentionGrad:cuda()
end

params, gradParams = model:getParameters()
if checkpoint ~= nil then
    params:copy(checkpoint.params)
end

local optimMethod, optimState

if opt.opt == 'sgd' then
    optimState = {
        learningRate = opt.learningRate,
        momentum = opt.momentum,
        learningRateDecay = 5e-7
    }
    optimMethod = optim.sgd
elseif opt.opt == 'rmsprop' then
    optimState = {
        learningRate = opt.learning_rate,
        alpha = 0.95
    }
    optimMethod = optim.rmsprop
elseif opt.opt == 'adadelta' then
    optimState= {
        rho = 0.95,
        eps = 1e-6
    }
    optimMethod = optim.adadelta
else
    error('unknown optimization method')
end

local iterations = opt.epochs * loader.numBatches

if opt.log then
    trainLogger = optim.Logger(string.format('%s/%s.log', opt.checkpoint_dir,
                               opt.savefile))
    trainLogger.showPlot = false
end

for i = 1, iterations do
    model:training()
    local epoch = i / loader.numBatches
    local timer = torch.Timer()

    local encInSeq, encInSeqRev, decInSeq, decOutSeq, ylen = loader:nextBatch()
    if opt.gpuid >= 0 then
        encInSeq = encInSeq:cuda()
        decInSeq = decInSeq:cuda()
        decOutSeq = decOutSeq:cuda()
    end
    local feval = function(x)
        if x ~= params then
            params:copy(x)
        end

        gradParams:zero()
        -- fwd
        local encOut = enc:forward(encInSeq)
        --- init decoder
        local encOutMean = getMean:forward(encOut)
        decgru.userPrevOutput= nn.rnn.recursiveCopy(decgru.userPrevOutput, encOutMean)
        decgru:setFeatures(encOut)


        local decInputs = decLookup:forward(decInSeq)

        decgru:forget() -- forget

        local preds = {}
        local err = 0
        for step = 1, #decInputs do
            preds[step] = decgru:forward(decInputs[step])
            for j = 1, ylen:size(1) do
                if step > ylen[j] then
                    preds[step][j] = 0
                end
            end
            err = err + criterion:forward(preds[step], decOutSeq:select(2, step))
        end

        -- print prediction
        if i % opt.print_sent_every == 0 then
            for j = 1, decOutSeq:size(1) do
                local sent, sentlen = seqtosent(decOutSeq[j], id2wd)
                print(sent)
                for step = 1, sentlen do
                    local _, wd = torch.max(preds[step][j], 1)
                    io.write(id2wd[wd[1]] .. ' ')
                end
                io.write('\n')
                print('-------')
            end
        end

        -- bwd
        local gradPreds = {}
        local decInputsGrad = {}
        zeroOutputGrad:resizeAs(decgru.output):zero()
        zeroAttentionGrad:resizeAs(decgru.attention_norm):zero()
        for step = #decInputs, 1, -1 do
            gradPreds[step] = criterion:backward(preds[step], decOutSeq:select(2, step))
            local gradOutTable = {zeroOutputGrad, zeroAttentionGrad, gradPreds[step]}
            local gradInput = decgru:backward(decInputs[step], gradOutTable)
            decInputsGrad[step] = gradInput
        end
        decLookup:backward(decInSeq, decInputsGrad)

        local encInGrad = torch.Tensor():typeAs(x):resizeAs(encOut):copy(decgru.gradFeaturesAcc)
        -- bwd mean
        local encInMeanGrad = getMean:backward(encOut, decgru.userGradPrevOutput)
        encInGrad:add(encInMeanGrad)
        enc:backward(encInSeq, encInGrad)

        gradParams:clamp(-opt.gradClip, opt.gradClip)
        return err, gradParams
    end
    local _, loss = optimMethod(feval, params, optimState)
    lookup1.weight[1]:zero()
    lookup2.weight[1]:zero()

    local time = timer:time().real
    local trainLoss = loss[1] / (math.ceil(torch.mean(ylen)))
    if trainLogger then
        trainLogger:add{['loss'] = trainLoss}
    end
    if i % opt.save_every == 0 or i % loader.numBatches == 0 or i == iterations then
        local savefile = string.format('%s/%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, trainLoss)
        print('iter ', i, 'save checkpoint', savefile)
        --model:clearState()
        local checkpoint = {}
        checkpoint.params = params -- only save params for low storage
        checkpoint.opt = opt
        checkpoint.i = i
        checkpoint.epoch = epoch
        torch.save(savefile, checkpoint)
        print('done')
        if trainLogger then
            trainLogger:style{['loss'] = '-'}
            trainLogger:plot()
        end
    end

    if i % opt.print_every == 0 then
        print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.4fs", i, iterations, epoch, trainLoss, gradParams:norm() / params:norm(), time))
    end

    if i % 10 == 0 then
        collectgarbage()
    end
end
