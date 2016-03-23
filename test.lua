require 'hdf5'

cmd = torch.CmdLine()
cmd:option('-data', '/data_giles/cul226/simple/preprocessed/newsla_Google.hdf5', 'training data and word2vec data')
cmd:option('-gpuid', 1, 'gpu id')
cmd:option('-init_from', '', 'init from checkpoint model')

opt = cmd:parse(arg)

if opt.gpuid >= 0 then
    require 'cutorch'
    cutorch.setDevice(opt.gpuid + 1)
    require 'cunn'
    require 'cudnn'
end

local DL = require 'DataLoader'
loader = DL.create(opt.data)

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
local enc = encoderDecoder.enc
local dec = encoderDecoder.dec
local encLSTM = encoderDecoder.encLSTM
local decLSTM = encoderDecoder.decLSTM

local inSeq = loader.dataXRev[1]
local encOut = enc:forward(inSeq)
print(encOut:size())
EncDec.forwardConnect(encLSTM, decLSTM, inSeq:size())

local zeros = torch.Tensore(encOut:size(2))


