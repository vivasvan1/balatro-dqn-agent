local http = {}

-- Simple HTTP client using LuaSocket
local socket = require("socket")
local ltn12 = require("ltn12")

function http.post(url, options)
    options = options or {}
    local headers = options.headers or {}
    local body = options.body or ""
    
    -- Parse URL
    local protocol, host, path = url:match("^(https?)://([^/]+)(.*)$")
    if not protocol then
        return {status = 400, body = "Invalid URL"}
    end
    
    -- Create request
    local request = {
        "POST " .. path .. " HTTP/1.1",
        "Host: " .. host,
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
    local success, err = tcp:connect(host, 80)
    if not success then
        return {status = 500, body = "Connection failed: " .. err}
    end
    
    -- Send request
    local request_str = table.concat(request, "\r\n")
    success, err = tcp:send(request_str .. "\r\n")
    if not success then
        tcp:close()
        return {status = 500, body = "Send failed: " .. err}
    end
    
    -- Receive response
    local response = {}
    local line, err = tcp:receive("*l")
    if not line then
        tcp:close()
        return {status = 500, body = "Receive failed: " .. err}
    end
    
    -- Parse status line
    local status = tonumber(line:match("HTTP/%d%.%d (%d+)"))
    if not status then
        tcp:close()
        return {status = 500, body = "Invalid response"}
    end
    
    -- Skip headers
    repeat
        line = tcp:receive("*l")
    until line == ""
    
    -- Read body
    local body = ""
    while true do
        local chunk, err = tcp:receive("*a")
        if not chunk then break end
        body = body .. chunk
    end
    
    tcp:close()
    return {status = status, body = body}
end

return http 