local Utils = {}

-- Helper function to serialize tables
function Utils.serializeTable(tbl, visited, path)
    visited = visited or {}
    path = path or ""
    if visited[tbl] then
        return "\"[Circular Reference at " .. path .. "]\""
    end
    visited[tbl] = true

    local result = "{"
    local first = true
    for k, v in pairs(tbl) do
        if not first then result = result .. "," end
        first = false

        -- Handle key
        if type(k) == "string" then
            result = result .. "\"" .. k .. "\":"
        else
            result = result .. "\"" .. tostring(k) .. "\":"
        end

        -- Handle value
        if type(v) == "table" then
            local new_path = path .. (path ~= "" and "." or "") .. tostring(k)
            result = result .. Utils.serializeTable(v, visited, new_path)
        elseif type(v) == "string" then
            result = result .. "\"" .. v:gsub("\"", "\\\"") .. "\""
        elseif type(v) == "function" then
            -- Use debug.getinfo to get function info
            local info = debug.getinfo(v, "uS")
            local args = {}
            if info and info.nparams then
                for i = 1, info.nparams do
                    local name = debug.getlocal(v, i)
                    if name then table.insert(args, name) end
                end
            end
            local argstr = table.concat(args, ", ")
            if info and info.what == "C" then
                result = result .. string.format("\"[C Function]\"")
            else
                result = result .. string.format("\"[Function(args: %s)]\"", argstr)
            end
        elseif type(v) == "userdata" or type(v) == "thread" then
            result = result .. "\"[" .. type(v) .. "]\""
        else
            result = result .. tostring(v)
        end
    end
    result = result .. "}"
    return result
end

-- Helper function to save game state to JSON
function Utils.saveGameStateToJson(data, filename)
    local json_path = Lovely.mod_dir .. "/balatro_agent/" .. filename
    local file = nativefs.newFile(json_path)
    local json_data = Utils.serializeTable(data)
    local success, error_msg = file:open("w")
    if success then
        file:write(json_data)
        file:close()
        print("Game state saved to: " .. json_path)
        return true
    else
        print("Failed to create JSON file: " .. (error_msg or "unknown error"))
        return false
    end
end

-- Helper function to create delayed events
function Utils.createDelayedEvent(delay, func, description)
    G.E_MANAGER:add_event(Event({
        trigger = 'after',
        delay = delay,
        blockable = false,
        blocking = false,
        func = function()
            if description then
                print("Executing delayed event: " .. description)
            end
            return func()
        end
    }))
end

return Utils 