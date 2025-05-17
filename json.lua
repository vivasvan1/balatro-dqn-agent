local json = {}

-- Helper functions
local function is_array(t)
    local i = 0
    for _ in pairs(t) do
        i = i + 1
        if t[i] == nil then
            return false
        end
    end
    return true
end

local function escape_string(s)
    return s:gsub('[\\"]', '\\%1')
end

-- Encode functions
local function encode_value(v)
    local t = type(v)
    if t == "string" then
        return '"' .. escape_string(v) .. '"'
    elseif t == "number" then
        return tostring(v)
    elseif t == "boolean" then
        return tostring(v)
    elseif t == "nil" then
        return "null"
    elseif t == "table" then
        if is_array(v) then
            local result = {}
            for _, val in ipairs(v) do
                table.insert(result, encode_value(val))
            end
            return "[" .. table.concat(result, ",") .. "]"
        else
            local result = {}
            for k, val in pairs(v) do
                table.insert(result, '"' .. escape_string(tostring(k)) .. '":' .. encode_value(val))
            end
            return "{" .. table.concat(result, ",") .. "}"
        end
    else
        error("Cannot encode value of type " .. t)
    end
end

function json.encode(t)
    return encode_value(t)
end

-- Decode functions
local function skip_whitespace(s, pos)
    return s:find("[^ \t\n\r]", pos) or #s + 1
end

local function parse_string(s, pos)
    local result = ""
    pos = pos + 1
    while pos <= #s do
        local c = s:sub(pos, pos)
        if c == '"' then
            return result, pos + 1
        elseif c == "\\" then
            pos = pos + 1
            c = s:sub(pos, pos)
            if c == '"' or c == "\\" then
                result = result .. c
            else
                error("Invalid escape sequence")
            end
        else
            result = result .. c
        end
        pos = pos + 1
    end
    error("Unterminated string")
end

local function parse_number(s, pos)
    local num_str = s:match("^-?%d+%.?%d*[eE]?[+-]?%d*", pos)
    if not num_str then
        error("Invalid number")
    end
    return tonumber(num_str), pos + #num_str
end

local function parse_value(s, pos)
    pos = skip_whitespace(s, pos)
    local c = s:sub(pos, pos)
    
    if c == '"' then
        return parse_string(s, pos)
    elseif c == "{" then
        return parse_object(s, pos)
    elseif c == "[" then
        return parse_array(s, pos)
    elseif c == "t" and s:sub(pos, pos + 3) == "true" then
        return true, pos + 4
    elseif c == "f" and s:sub(pos, pos + 4) == "false" then
        return false, pos + 5
    elseif c == "n" and s:sub(pos, pos + 3) == "null" then
        return nil, pos + 4
    elseif c == "-" or c:match("%d") then
        return parse_number(s, pos)
    else
        error("Invalid JSON")
    end
end

local function parse_object(s, pos)
    local result = {}
    pos = pos + 1
    pos = skip_whitespace(s, pos)
    
    if s:sub(pos, pos) == "}" then
        return result, pos + 1
    end
    
    while true do
        pos = skip_whitespace(s, pos)
        if s:sub(pos, pos) ~= '"' then
            error("Expected string key")
        end
        
        local key, new_pos = parse_string(s, pos)
        pos = new_pos
        pos = skip_whitespace(s, pos)
        
        if s:sub(pos, pos) ~= ":" then
            error("Expected ':'")
        end
        pos = pos + 1
        
        local value, new_pos = parse_value(s, pos)
        result[key] = value
        pos = new_pos
        pos = skip_whitespace(s, pos)
        
        if s:sub(pos, pos) == "}" then
            return result, pos + 1
        elseif s:sub(pos, pos) == "," then
            pos = pos + 1
        else
            error("Expected ',' or '}'")
        end
    end
end

local function parse_array(s, pos)
    local result = {}
    pos = pos + 1
    pos = skip_whitespace(s, pos)
    
    if s:sub(pos, pos) == "]" then
        return result, pos + 1
    end
    
    while true do
        local value, new_pos = parse_value(s, pos)
        table.insert(result, value)
        pos = new_pos
        pos = skip_whitespace(s, pos)
        
        if s:sub(pos, pos) == "]" then
            return result, pos + 1
        elseif s:sub(pos, pos) == "," then
            pos = pos + 1
        else
            error("Expected ',' or ']'")
        end
    end
end

function json.decode(s)
    local result, pos = parse_value(s, 1)
    pos = skip_whitespace(s, pos)
    if pos <= #s then
        error("Trailing garbage")
    end
    return result
end

return json 