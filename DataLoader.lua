require 'hdf5'

local DataLoader = {}
DataLoader.__index = DataLoader

function DataLoader.create(dataFile, batchSize)
    batchSize = batchSize or 1

    self = {}
    setmetatable(self, DataLoader)

    local w2v, xChunks, yChunks, ylenChunks
    xChunks, yChunks, ylenChunks, w2v = DataLoader.loadData(dataFile)

    local xrevChunks = {}
    local yshiftChunks = {}
    self.chunkBatches = {}
    self.numBatches = 0

    for k = 1, #xChunks do
        local dataX = xChunks[k]
        -- reverse input sequences
        local dataXRev = torch.Tensor(dataX:size())
        for i = 1, dataX:size(2) do
            dataXRev:select(2, i):copy(dataX:select(2, dataX:size(2) + 1 - i))
        end
        table.insert(xrevChunks, dataXRev)
        local dataY = yChunks[k]
        -- shift output sequences
        local dataYShift = torch.cat(dataY:select(2, dataY:size(2)), dataY:sub(1, -1, 1, -2), 2)
        table.insert(yshiftChunks, dataYShift)
        local nBatches = math.ceil(dataX:size(1) / batchSize)
        self.chunkBatches[k] = nBatches
        self.numBatches = self.numBatches + nBatches
    end

    self.w2v = w2v
    self.xChunks = xChunks
    self.yChunks = yChunks
    self.ylenChunks = ylenChunks
    self.xrevChunks = xrevChunks
    self.yshiftChunks = yshiftChunks

    self:prepro()

    self.chunkIdx = 1
    self.batchIdx = 0
    self.batchSize = batchSize
    collectgarbage()
    return self
end

function DataLoader:nextBatch()
    self.batchIdx = self.batchIdx + 1
    if self.batchIdx > self.chunkBatches[self.chunkIdx] then
        self.batchIdx = 1
        self.chunkIdx = self.chunkIdx + 1
        if self.chunkIdx > #self.xChunks then
            self:prepro()
            self.chunkIdx = 1
        end
    end
    local xrev = self.xrevChunks[self.chunkIdx]
    local yshift = self.yshiftChunks[self.chunkIdx]
    local y = self.yChunks[self.chunkIdx]

    local st = (self.batchIdx - 1) * self.batchSize + 1
    local batchSize = math.min(self.batchSize, self.xChunks[self.chunkIdx]:size(1) - st + 1)
    local ylen = self.ylenChunks[self.chunkIdx]:narrow(1, st, batchSize)
    local maxlen = torch.max(ylen)
    local encInSeq = xrev:narrow(1, st, batchSize)
    local decInSeq = yshift:sub(st, st + batchSize - 1, 1, maxlen)
    local decOutSeq = y:sub(st, st + batchSize - 1, 1, maxlen)
    return encInSeq, decInSeq, decOutSeq, ylen
end

function DataLoader:prepro()
    self:shuffleData()
    self:sortByLen()
end

function DataLoader:shuffleData()
    print('shuffle data...')
    for k = 1, #self.xChunks do
        local shuffle = torch.randperm(self.xChunks[k]:size(1))
        self:permuteChunk(k, shuffle)
    end
    print('shuffle data done')
end

function DataLoader:sortByLen()
    print('sorting data by the length of y...')
    for k = 1, #self.xChunks do
        local _, indices = torch.sort(self.ylenChunks[k])
        self:permuteChunk(k, indices)
    end
    print('sorting done')
end

function DataLoader:permuteChunk(k, indices)
    local dataX = torch.Tensor(self.xChunks[k]:size())
    local dataY = torch.Tensor(self.yChunks[k]:size())
    local dataXRev = torch.Tensor(self.xrevChunks[k]:size())
    local dataYShift = torch.Tensor(self.yshiftChunks[k]:size())
    local ylen = torch.Tensor(self.ylenChunks[k]:size())
    for i = 1, indices:size(1) do
        dataX:select(1, i):copy(self.xChunks[k]:select(1, indices[i]))
        dataY:select(1, i):copy(self.yChunks[k]:select(1, indices[i]))
        dataXRev:select(1, i):copy(self.xrevChunks[k]:select(1, indices[i]))
        dataYShift:select(1, i):copy(self.yshiftChunks[k]:select(1, indices[i]))
        ylen[i] = self.ylenChunks[k][indices[i]]
    end
    self.xChunks[k] = dataX
    self.yChunks[k] = dataY
    self.xrevChunks[k] = dataXRev
    self.yshiftChunks[k] = dataYShift
    self.ylenChunks[k] = ylen
    collectgarbage()
end

function DataLoader.loadData(dataFile)
    print('loading data...')
    local f = hdf5.open(dataFile, 'r')
    local w2v = f:read('w2v'):all()
    local xlens = f:read('x_lens'):all()
    local xChunks = {}
    local yChunks = {}
    local ylenChunks = {}
    local dataX, dataY, lenY
    for i = 1, xlens:size(1) do
        dataX = f:read('x_' .. xlens[i]):all()
        dataY = f:read('y_' .. xlens[i]):all()
        lenY = f:read('ylen_' .. xlens[i]):all()
        table.insert(xChunks, dataX)
        table.insert(yChunks, dataY)
        table.insert(ylenChunks, lenY)
    end
    print('done')
    return xChunks, yChunks, ylenChunks, w2v
end

function DataLoader.loadVocab(vocabFile)
    local wd2id = {}
    for line in io.lines(vocabFile) do
        local wds = {};
        for wd in string.gmatch(line, "%S+") do
            table.insert(wds, wd)
        end
        wd2id[wds[1]] = tonumber(wds[2])
    end
    return wd2id
end

return DataLoader
