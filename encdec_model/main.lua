require 'optim'
require '../utils/misc'

-- options
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

-- data loader
local DL = require '../utils/DataLoader'
local loader = DL.create(opt.data, opt.batchSize)
local wd2id = DL.loadVocab(opt.vocab)
local id2wd = {}
for wd, id in pairs(wd2id) do
    id2wd[id] = wd
end

local checkpoint
if string.len(opt.init_from) > 0 then
    print('init model from', opt.init_from)
    checkpoint = torch.load(opt.init_from)
    opt.hiddenSize = checkpoint.opt.hiddenSize
end

local EncDec = require 'EncoderDecoder'
local encoderDecoder = EncDec.create(loader.w2v, opt.hiddenSize)

local model = encoderDecoder.model
local criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

local toTable = nn.SplitTable(1, 1)

if opt.gpuid >= 0 then
    model:cuda()
    criterion:cuda()
    toTable:cuda()
end

params, gradParams = model:getParameters()
print('number of parameters in the model: ' .. params:nElement())

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
trainLosses = {}
print('start training...')
for i = 1, iterations do
    model:training()
    local epoch = i / loader.numBatches
    local timer = torch.Timer()

    local encInSeq, encInSeqRev, decInSeq, decOutSeq, ylen = loader:nextBatch()
    if opt.gpuid >= 0 then
        encInSeqRev = encInSeqRev:cuda()
        decInSeq = decInSeq:cuda()
        decOutSeq = decOutSeq:cuda()
    end
    local feval = function(x)
        if x ~= params then
            params:copy(x)
        end

        gradParams:zero()
        -- forward
        local decOut = encoderDecoder:forward(encInSeqRev, decInSeq)

        -- print prediction
        if i % opt.print_sent_every == 0 then
            for j = 1, decOutSeq:size(1)  do
                local sent, sentlen = seqtosent(decOutSeq[j], id2wd)
                print(sent)
                for step = 1, sentlen do
                    local _, wd = torch.max(decOut[step][j], 1)
                    io.write(id2wd[wd[1]] .. ' ')
                end
                io.write('\n')
                print('--------')
            end
        end

        decOutSeq = toTable:forward(decOutSeq)

        -- mask decOut
        local ylenmax = torch.max(ylen)
        for j = 1, ylen:size(1) do
            for k = ylen[j] + 1, ylenmax do
                decOut[k][j]:zero()
            end
        end

        local err = criterion:forward(decOut, decOutSeq)

        -- backward
        local gradOutput = criterion:backward(decOut, decOutSeq)
        encoderDecoder:backward(encInSeqRev, decInSeq, gradOutput)

        return err, gradParams
    end

    local _, loss = optimMethod(feval, params, optimState)
    encoderDecoder.enc:get(1).weight[1]:zero()
    encoderDecoder.dec:get(1).weight[1]:zero()

    local time = timer:time().real
    local trainLoss = loss[1] / (math.ceil(torch.mean(ylen)))
    trainLosses[i] = trainLoss

    if i % opt.save_every == 0 or i % loader.numBatches == 0 or i == iterations then
        local savefile = string.format('%s/%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, trainLoss)
        print('iter ', i, 'save checkpoint', savefile)
        model:clearState()
        local checkpoint = {}
        -- checkpoint.encoderDecoder = encoderDecoder
        checkpoint.params = params -- only save params for low storage
        checkpoint.opt = opt
        -- checkpoint.trainLosses = trainLosses
        checkpoint.i = i
        checkpoint.epoch = epoch
        torch.save(savefile, checkpoint)
        print('done')
    end

    if i % opt.print_every == 0 then
        print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.4fs", i, iterations, epoch, trainLoss, gradParams:norm() / params:norm(), time))
    end

    if i % 10 == 0 then
        collectgarbage()
    end

end

