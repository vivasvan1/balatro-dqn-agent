-- #region HTTP
-- HTTP module for Balatro agent
local Http = {}

-- Check if LuaSocket is available
local socket_ok, socket = pcall(require, "socket")
if not socket_ok then
    error("LuaSocket is required but not found. Please install it.")
end

local ltn12_ok, ltn12 = pcall(require, "ltn12")
if not ltn12_ok then
    error("LTN12 is required but not found. Please install it.")
end

function Http.post(url, options)
    options = options or {}
    local headers = options.headers or {}
    local body = options.body or ""

    -- Parse URL
    local protocol, host, port, path = url:match("^(https?)://([^:/]+):?(%d*)(.*)")
    if not protocol then
        return { status = 400, body = "Invalid URL" }
    end

    -- Default to port 80 if not specified
    port = port ~= "" and tonumber(port) or 80

    -- Create request
    local request = {
        "POST " .. (path ~= "" and path or "/") .. " HTTP/1.1",
        "Host: " .. host .. (port ~= 80 and ":" .. port or ""),
        "Content-Type: application/json",
        "Content-Length: " .. #body,
        "",
        body
    }

    -- Add custom headers
    for k, v in pairs(headers) do
        table.insert(request, 3, k .. ": " .. v)
    end

    -- Connect to server
    local tcp = socket.tcp()
    tcp:settimeout(5)
    local success, err = tcp:connect(host, port)
    if not success then
        return { status = 500, body = "Connection failed: " .. err }
    end

    -- Send request
    local request_str = table.concat(request, "\r\n")
    success, err = tcp:send(request_str .. "\r\n")
    if not success then
        tcp:close()
        return { status = 500, body = "Send failed: " .. err }
    end

    -- Receive response
    local response = {}
    local line, err = tcp:receive("*l")
    if not line then
        tcp:close()
        return { status = 500, body = "Receive failed: " .. err }
    end

    -- Parse status line
    local status = tonumber(line:match("HTTP/%d%.%d (%d+)"))
    if not status then
        tcp:close()
        return { status = 500, body = "Invalid response" }
    end

    -- Read headers and look for Content-Length
    local content_length = nil
    repeat
        line = tcp:receive("*l")
        if line then
            local cl = line:match("^[Cc]ontent%-[Ll]ength:%s*(%d+)")
            if cl then
                content_length = tonumber(cl)
            end
        end
    until line == ""

    -- Read body
    local body = ""
    if content_length then
        -- Read exact number of bytes specified by Content-Length
        body, err = tcp:receive(content_length)
        if not body then
            tcp:close()
            return { status = 500, body = "Failed to read body: " .. (err or "unknown error") }
        end
    else
        -- Fallback: try to read with timeout
        tcp:settimeout(1) -- Set a shorter timeout for body reading
        local chunk, err = tcp:receive("*a")
        if chunk then
            body = chunk
        end
    end

    tcp:close()
    return { status = status, body = body }
end

-- #endregion
