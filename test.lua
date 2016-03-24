require 'misc'

cmd = torch.CmdLine()
cmd:option('-data', '/data_giles/cul226/simple/preprocessed/newsla_Google.hdf5', 'training data and word2vec data')
cmd:option('-gpuid', 1, 'gpu id')
cmd:option('-init_from', '', 'init from checkpoint model')
cmd:option('-vocab', '', 'vocabulary file')

opt = cmd:parse(arg)

if opt.gpuid >= 0 then
    require 'cutorch'
    cutorch.setDevice(opt.gpuid + 1)
    require 'cunn'
    require 'cudnn'
end


local DL = require 'DataLoader'
local loader = DL.create(opt.data)

local wd2id = DL.loadVocab(opt.vocab)
local id2wd = {}
for wd, id in pairs(wd2id) do
    id2wd[id] = wd
end

local EncDec = require 'EncoderDecoder'

if string.len(opt.init_from) > 0 then
    print('init model from', opt.init_from)
    local checkpoint = torch.load(opt.init_from)
    encoderDecoder = checkpoint.encoderDecoder
    setmetatable(encoderDecoder, EncDec)
else
    error('need a pretrained model')
end

local model = encoderDecoder.model
model:evaluate()

local enc = encoderDecoder.enc
local dec = encoderDecoder.dec
local encLSTM = encoderDecoder.encLSTM
local decLSTM = encoderDecoder.decLSTM
local linear = dec:get(4).modules[1].modules[1]
local softmax = dec:get(5).modules[1].modules[1]

local predictor = nn.Sequential():add(decLSTM):add(linear):add(softmax)

local idx = 1000
local inSeq = loader.dataXRev[idx]

local inSent = seqtosent(loader.dataX[idx], id2wd)
print(inSent)

local outseq = loader.dataY[idx]
local outSent = seqtosent(outseq, id2wd)
print(outSent)

local encOut = enc:forward(inSeq)
EncDec.forwardConnect(encLSTM, decLSTM, inSeq:size(1))
local prevWd = 1
local maxLen = 20

for i = 1, maxLen do
    vec = loader.w2v[prevWd]
    pred = predictor:forward(vec)
    local _, prevWd = torch.max(pred, 1)
    io.write(id2wd[prevWd[1]] .. ' ')
end
io.write('\n')

