function zeroDataSize(data)
    if type(data) == 'table' then
        for i = 1, #data do
            data[i] = zeroDataSize(data[i])
        end
    elseif type(data) == 'userdata' then
        data = torch.Tensor():typeAs(data)
    end
    return data
end

function cleanupModel(node)
    if node.output ~= nil then
        node.output = zeroDataSize(node.output)
    end
    if node.gradInput ~= nil then
        node.gradInput = zeroDataSize(node.gradInput)
    end
    if node.finput ~= nil then
        node.finput = zeroDataSize(node.finput)
    end
    -- Recurse on nodes with 'modules'
    if (node.modules ~= nil) then
        if (type(node.modules) == 'table') then
            for i = 1, #node.modules do
                local child = node.modules[i]
                cleanupModel(child)
            end
        end
    end
    collectgarbage()
end

function seqtosent(seq, id2wd)
    local sent = ''
    local len = 0
    for i = 1, seq:size(1) do
        local wd = seq[i]
        if wd == 1 then
            break
        end
        len = len + 1
        sent = sent .. id2wd[wd] .. ' '
    end
    return sent, len
end
