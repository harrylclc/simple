local DataLoader = {}
DataLoader.__index = DataLoader

function DataLoader.create(dataFile, batchSize)
    self = {}
    setmetatable(self, DataLoader)

    local w2v, dataX, dataY
    dataX, dataY, w2v = DataLoader.loadData(dataFile)
    
    -- reverse input sequences
    local dataXRev = torch.Tensor(dataX:size())
    for i = 1, dataX:size(2) do
        dataXRev:select(2, i):copy(dataX:select(2, dataX:size(2) + 1 - i))
    end

    -- shift output sequences
    local dataYShift = torch.cat(dataY:select(2, dataY:size(2)), dataY:sub(1, -1, 1, -2), 2)
    
    self.dataX = dataX
    self.dataY = dataY
    self.dataXRev = dataXRev
    self.dataYShift = dataYShift
    self.w2v = w2v
    self.batchSize = batchSize

    self.trainSize = dataX:size(1)
    self.numBatches = math.floor(self.trainSize / batchSize)
    self.batchIdx = 0
    collectgarbage()
    return self
end

function DataLoader:nextBatch()
    self.batchIdx = self.batchIdx + 1
    if self.batchIdx > self.numBatches then
        self:shuffleData()
        self.batchIdx = 1
    end
    local st = (self.batchIdx - 1) * self.batchSize + 1
    local batchSize = math.min(self.batchSize, self.trainSize - st + 1)
    local encInSeq = self.dataXRev:narrow(1, st, self.batchSize)
    local decInSeq = self.dataYShift:narrow(1, st, self.batchSize)
    local decOutSeq = self.dataY:narrow(1, st, self.batchSize)
    return encInSeq, decInSeq, decOutSeq
end

function DataLoader:shuffleData()
    print('shuffle data...')
    local shuffle = torch.randperm(self.dataX:size(1))
    local dataX = torch.Tensor(self.dataX:size())
    local dataY = torch.Tensor(self.dataY:size())
    local dataXRev = torch.Tensor(self.dataXRev:size())
    local dataYShift = torch.Tensor(self.dataYShift:size())
    for i = 1, shuffle:size(1) do
        dataX:select(1, i):copy(dataX:select(1, shuffle[i]))
        dataY:select(1, i):copy(dataY:select(1, shuffle[i]))
        dataXRev:select(1, i):copy(dataXRev:select(1, shuffle[i]))
        dataYShift:select(1, i):copy(dataYShift:select(1, shuffle[i]))
    end
    self.dataX = dataX
    self.dataY = dataY
    self.dataXRev = dataXRev
    self.dataYShift = dataYShift
    collectgarbage()
    print('shuffle data done')
end

function DataLoader.loadData(dataFile)
    local w2v, dataX, dataY
    print('loading data...')
    local f = hdf5.open(dataFile, 'r')
    w2v = f:read('w2v'):all()
    dataX = f:read('data_x'):all()
    dataY = f:read('data_y'):all()
    print('done')
    return dataX, dataY, w2v
end

function DataLoader.loadVocab(vocabFile)
    local wd2id = {}
    for line in io.lines(vocabFile) do
        local wds = {};
        for wd in string.gmatch(line, "%S+") do
            table.insert(wds, wd)
        end
        wd2id[wds[1]] = wds[2]
    end
    return wd2id
end

return DataLoader
