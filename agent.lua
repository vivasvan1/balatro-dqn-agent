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

-- #region Json
-- JSON module for Balatro agent
local Json = {}

-- Forward declarations
local parse_object, parse_array

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

function Json.encode(t)
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

parse_object = function(s, pos)
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

parse_array = function(s, pos)
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

function Json.decode(s)
    local result, pos = parse_value(s, 1)
    pos = skip_whitespace(s, pos)
    if pos <= #s then
        error("Trailing garbage")
    end
    return result
end

-- #endregion

-- Balatro Agent - Consolidated Modular Design
-- All modules included inline for lovely.toml compatibility

Lovely = require("lovely")

-- #region UTILS MODULE
-- ============================================================================
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

-- #endregion



-- #region STATE MACHINE MODULE

local StateMachine = {}

-- Agent phases
StateMachine.PHASES = {
    IDLE = "IDLE",
    SELECTING_BLIND = "SELECTING_BLIND",
    SELECTING_HAND = "SELECTING_HAND",
    ACTION_PENDING = "ACTION_PENDING",
    WAITING_FOR_REWARD = "WAITING_FOR_REWARD",
    PROCESSING_REWARD = "PROCESSING_REWARD",
    GAME_OVER = "GAME_OVER"
}

-- Initialize state machine
function StateMachine.init()
    StateMachine.current_phase = StateMachine.PHASES.IDLE
    StateMachine.previous_phase = nil
    StateMachine.phase_start_time = love.timer.getTime()
    StateMachine.callbacks = {}
end

-- Register callback for phase transitions
function StateMachine.onPhaseChange(callback)
    table.insert(StateMachine.callbacks, callback)
end

-- Change phase with logging and callbacks
function StateMachine.setPhase(new_phase, reason)
    if StateMachine.current_phase ~= new_phase then
        local old_phase = StateMachine.current_phase
        StateMachine.previous_phase = old_phase
        StateMachine.current_phase = new_phase
        StateMachine.phase_start_time = love.timer.getTime()

        print(string.format("State Machine: %s -> %s. Reason: %s",
            old_phase, new_phase, reason or "N/A"))

        -- Call registered callbacks
        for _, callback in ipairs(StateMachine.callbacks) do
            local success, err = pcall(callback, old_phase, new_phase, reason)
            if not success then
                print("ERROR in phase change callback: " .. tostring(err))
            end
        end
    end
end

-- Get current phase
function StateMachine.getPhase()
    return StateMachine.current_phase
end

-- Get previous phase
function StateMachine.getPreviousPhase()
    return StateMachine.previous_phase
end

-- Get time spent in current phase
function StateMachine.getPhaseTime()
    return love.timer.getTime() - StateMachine.phase_start_time
end

-- Check if in specific phase
function StateMachine.isInPhase(phase)
    return StateMachine.current_phase == phase
end

-- Check if transitioning is allowed
function StateMachine.canTransitionTo(target_phase)
    local current = StateMachine.current_phase
    local valid_transitions = {
        [StateMachine.PHASES.IDLE] = {
            StateMachine.PHASES.SELECTING_BLIND,
            StateMachine.PHASES.GAME_OVER
        },
        [StateMachine.PHASES.SELECTING_BLIND] = {
            StateMachine.PHASES.SELECTING_HAND,
            StateMachine.PHASES.GAME_OVER
        },
        [StateMachine.PHASES.SELECTING_HAND] = {
            StateMachine.PHASES.ACTION_PENDING,
            StateMachine.PHASES.GAME_OVER
        },
        [StateMachine.PHASES.ACTION_PENDING] = {
            StateMachine.PHASES.WAITING_FOR_REWARD,
            StateMachine.PHASES.IDLE,
            StateMachine.PHASES.GAME_OVER
        },
        [StateMachine.PHASES.WAITING_FOR_REWARD] = {
            StateMachine.PHASES.PROCESSING_REWARD,
            StateMachine.PHASES.GAME_OVER
        },
        [StateMachine.PHASES.PROCESSING_REWARD] = {
            StateMachine.PHASES.SELECTING_HAND,
            StateMachine.PHASES.GAME_OVER,
            StateMachine.PHASES.IDLE
        },
        [StateMachine.PHASES.GAME_OVER] = {
            StateMachine.PHASES.IDLE,
            StateMachine.PHASES.SELECTING_BLIND
        }
    }

    local allowed = valid_transitions[current]
    if not allowed then
        return false
    end

    for _, allowed_phase in ipairs(allowed) do
        if allowed_phase == target_phase then
            return true
        end
    end

    return false
end

-- Safe phase transition with validation
function StateMachine.transitionTo(target_phase, reason)
    if not StateMachine.canTransitionTo(target_phase) then
        print(string.format("WARNING: Invalid transition from %s to %s. Reason: %s",
            StateMachine.current_phase, target_phase, reason or "N/A"))
        return false
    end

    StateMachine.setPhase(target_phase, reason)
    return true
end

-- Reset state machine
function StateMachine.reset()
    StateMachine.setPhase(StateMachine.PHASES.IDLE, "State machine reset")
end

-- Get state machine status for debugging
function StateMachine.getStatus()
    return {
        current_phase = StateMachine.current_phase,
        previous_phase = StateMachine.previous_phase,
        phase_time = StateMachine.getPhaseTime(),
        phase_start_time = StateMachine.phase_start_time
    }
end

-- #endregion


-- #region GAME STATE MODULE

local GameState = {}

-- Value to letter mapping for cards
local VALUE_TO_LETTER = {
    ["Ace"] = "A",
    ["King"] = "K",
    ["Queen"] = "Q",
    ["Jack"] = "J",
    ["10"] = "10",
    ["9"] = "9",
    ["8"] = "8",
    ["7"] = "7",
    ["6"] = "6",
    ["5"] = "5",
    ["4"] = "4",
    ["3"] = "3",
    ["2"] = "2"
}

-- Suit to letter mapping for cards
local SUIT_TO_LETTER = {
    ["Diamonds"] = "D", ["Hearts"] = "H", ["Clubs"] = "C", ["Spades"] = "S"
}

-- State processor for converting game state to vector
function GameState.process()
    local state = {}
    local idx = 1
    local MAX_CARDS = 8 -- Fixed maximum number of cards to always have consistent state size

    -- Add game state info
    state[idx] = G.GAME.chips
    idx = idx + 1
    state[idx] = G.GAME.current_round.hands_left
    idx = idx + 1
    state[idx] = G.GAME.current_round.discards_left
    idx = idx + 1

    -- Add whole hand to state (always 8 card slots, padded with zeros if needed)
    for i = 1, MAX_CARDS do
        if i <= #G.hand.cards then
            local card = G.hand.cards[i]
            state[idx] = i
            state[idx + 1] = VALUE_TO_LETTER[card.base.value] .. SUIT_TO_LETTER[card.base.suit]
        else
            -- Pad with zeros for empty card slots
            state[idx] = -1
            state[idx + 1] = ""
        end
        idx = idx + 2
    end

    return state
end

-- Get formatted hand state for debugging
function GameState.getHandState()
    local hand_state = { whole_hand = {}, selected_hand = {} }

    -- Add whole hand to state
    for i, card in ipairs(G.hand.cards) do
        table.insert(hand_state.whole_hand, {
            index = i, value = card.base.value, suit = card.base.suit
        })
    end

    -- Add selected hand to state
    for _, card in ipairs(G.hand.highlighted) do
        table.insert(hand_state.selected_hand, {
            value = card.base.value, suit = card.base.suit
        })
    end

    return hand_state
end

-- Check if game state is valid for processing
function GameState.isValid()
    return G and G.GAME and G.GAME.chips and G.GAME.current_round and
        G.hand and G.hand.cards and #G.hand.cards > 0
end

-- Get current game state info for logging
function GameState.getStateInfo()
    if not GameState.isValid() then
        return "Invalid game state"
    end

    return string.format(
        "Chips: %d, Hands: %d, Discards: %d, Cards: %d",
        G.GAME.chips or 0, G.GAME.current_round.hands_left or 0,
        G.GAME.current_round.discards_left or 0, #G.hand.cards
    )
end

-- #endregion


-- #region ACTION EXECUTOR MODULE

local ActionExecutor = {}

-- Helper function to select cards by indices
function ActionExecutor.selectCardsByIndices(indices)
    if not G or not G.hand or not G.hand.cards then
        print("ERROR: No valid hand found.")
        return false
    end

    G.hand:unhighlight_all()

    -- Validate indices
    for _, idx in ipairs(indices) do
        if type(idx) ~= "number" or idx < 1 or idx > #G.hand.cards then
            print("ERROR: Invalid index: " .. tostring(idx) .. " (hand size: " .. #G.hand.cards .. ")")
            return false
        end
    end

    -- Highlight cards at specified indices
    for _, idx in ipairs(indices) do
        local card = G.hand.cards[idx]
        card:highlight(true)
        table.insert(G.hand.highlighted, card)
        print(string.format("Selected card %d: %s", idx, card.base.value .. card.base.suit))
    end

    if G.STATE == G.STATES.SELECTING_HAND then
        G.hand:parse_highlighted()
    end

    print("Highlighted " .. #indices .. " cards in hand.")
    return true
end

-- Execute an action based on action data from server
function ActionExecutor.execute(action_data)
    if not action_data or not action_data.indices or not action_data.action then
        print("ERROR: Invalid action data received")
        return false
    end

    if G.STATE ~= G.STATES.SELECTING_HAND then
        print("ERROR: Cannot execute action, not in SELECTING_HAND state. Current: " .. tostring(G.STATE))
        return false
    end

    if not ActionExecutor.selectCardsByIndices(action_data.indices) then
        print("ERROR: Failed to select cards for action")
        return false
    end

    if action_data.action == "play" then
        print("Executing: Playing selected cards")
        G.FUNCS.play_cards_from_highlighted()
        return true
    elseif action_data.action == "discard" then
        print("Executing: Discarding selected cards")
        G.FUNCS.discard_cards_from_highlighted()
        return true
    else
        print("ERROR: Unknown action type: " .. tostring(action_data.action))
        return false
    end
end

-- Validate action data structure
function ActionExecutor.validateActionData(action_data)
    if not action_data then return false, "Action data is nil" end
    if not action_data.action then return false, "Missing 'action' field" end
    if not action_data.indices then return false, "Missing 'indices' field" end
    if type(action_data.indices) ~= "table" then return false, "'indices' must be a table" end
    if #action_data.indices == 0 then return false, "'indices' table is empty" end
    if action_data.action ~= "play" and action_data.action ~= "discard" then
        return false, "Invalid action type: " .. tostring(action_data.action)
    end
    return true, "Valid"
end

-- #region REWARD CALCULATOR MODULE

local RewardCalculator = {}

-- Initialize reward tracking
function RewardCalculator.init()
    RewardCalculator.current_run_reward = 0
    RewardCalculator.last_chips = 0
    RewardCalculator.saves_count = 0
end

-- Calculate reward based on game state changes
function RewardCalculator.calculate()
    local reward = 0
    local done = false

    -- Base reward for winning with a hand
    if G.GAME.chips > G.GAME.blind.chips then
        reward = reward + 150
        done = true
        print("Added +150 reward for winning hand")
    end

    -- Reward for chips gained
    local chip_difference = G.GAME.chips - RewardCalculator.last_chips
    reward = reward + chip_difference
    print("Chip difference: " .. tostring(chip_difference))
    print("Added reward from chips: " .. tostring(chip_difference))

    -- Penalty for game over
    if G.STATE == G.STATES.GAME_OVER or G.STATE == G.STATES.NEW_ROUND then
        reward = reward - 100
        done = true
        print("Added -100 penalty for game over")
    end

    return reward, done
end

-- Update reward tracking
function RewardCalculator.updateTracking(reward, is_run_end)
    RewardCalculator.current_run_reward = RewardCalculator.current_run_reward + reward

    print(string.format("Reward: %.2f, Current Run Total: %.2f",
        reward, RewardCalculator.current_run_reward))

    if is_run_end then
        print(string.format("Run ended. Run reward: %.2f",
            RewardCalculator.current_run_reward))
        RewardCalculator.current_run_reward = 0
    end
end

-- Store pre-action state for reward calculation
function RewardCalculator.storePreActionState()
    RewardCalculator.last_chips = G.GAME.chips
    print("Stored pre-action chips: " .. RewardCalculator.last_chips)
end

-- Reset rewards for new game session
function RewardCalculator.resetForNewSession()
    RewardCalculator.current_run_reward = 0
    print("Reset cumulative rewards for new game session")
end

-- Reset rewards for new run after game over
function RewardCalculator.resetForNewRun()
    RewardCalculator.current_run_reward = 0
    print("Reset cumulative rewards for new run after game over")
end

-- Get reward statistics
function RewardCalculator.getStats()
    return {
        current_run_reward = RewardCalculator.current_run_reward,
        saves_count = RewardCalculator.saves_count
    }
end

-- Increment save count
function RewardCalculator.incrementSaveCount()
    RewardCalculator.saves_count = RewardCalculator.saves_count + 1
    print("Model saved successfully! Save count: " .. RewardCalculator.saves_count)
end

-- #endregion


-- #region HTTP CLIENT MODULE

local HttpClient = {}

-- Configuration
local SERVER_URL = "http://localhost:5000"
local PREDICT_ENDPOINT = SERVER_URL .. "/predict"
local TRAIN_ENDPOINT = SERVER_URL .. "/train"
local SAVE_ENDPOINT = SERVER_URL .. "/save"

-- Get action from server
function HttpClient.getAction(state)
    local request_body = Json.encode({ state = state })

    local response = Http.post(PREDICT_ENDPOINT, {
        headers = { ["Content-Type"] = "application/json" },
        body = request_body
    })

    print("Response status: " .. tostring(response.status))
    print("Response body: '" .. response.body .. "'")

    if response.status == 200 then
        if #response.body == 0 then
            print("ERROR: Received empty response body from server")
            return nil
        end

        local success, data = pcall(Json.decode, response.body)
        if success and data and data.action ~= nil and data.indices ~= nil then
            print("Action received: " .. tostring(data.action))
            print("Indices received: " .. tostring(#data.indices) .. " indices")
            return data
        end

        -- Fallback: Extract JSON from response
        local json_start = response.body:find("{")
        local json_end = response.body:find("}")
        if json_start and json_end then
            local json_str = response.body:sub(json_start, json_end)
            local success2, data2 = pcall(Json.decode, json_str)
            if success2 and data2 and data2.action ~= nil and data2.indices ~= nil then
                print("Action received from extracted JSON: " .. tostring(data2.action))
                return data2
            end
        end

        print("ERROR: Failed to parse action from response")
        return nil
    else
        print("ERROR: Getting action failed. Status: " .. response.status)
        return nil
    end
end

-- Send training data to server
function HttpClient.sendTrainingData(state, action, reward, next_state, done)
    local request_body = Json.encode({
        state = state,
        action = action,
        reward = reward,
        next_state = next_state,
        done = done
    })

    local response = Http.post(TRAIN_ENDPOINT, {
        headers = { ["Content-Type"] = "application/json" },
        body = request_body
    })

    if response.status ~= 200 then
        print("ERROR: Sending training data failed. Status: " .. response.status)
        return false
    else
        print("Training data sent successfully")
        return true
    end
end

-- Save model on server
function HttpClient.saveModel()
    local response = Http.post(SAVE_ENDPOINT)
    if response.status == 200 then
        print("Model saved successfully on server!")
        return true
    else
        print("ERROR: Saving model failed. Status: " .. response.status)
        return false
    end
end

-- Test server connection
function HttpClient.testConnection()
    print("Testing server connection...")
    local response = Http.post(SERVER_URL .. "/ping", {
        headers = { ["Content-Type"] = "application/json" },
        body = "{}"
    })

    if response.status == 200 then
        print("Server connection successful")
        return true
    else
        print("Server connection failed. Status: " .. response.status)
        return false
    end
end

-- #endregion


-- #region INPUT HANDLER MODULE

local InputHandler = {}

-- Input state tracking
local ctrl_key_pressed = false
local trigger_cooldown = 0.5
local last_trigger_time = nil

-- Initialize input handler
function InputHandler.init()
    ctrl_key_pressed = false
    last_trigger_time = nil
end

-- Check if ctrl+key combination is pressed
function InputHandler.checkCtrlKey(key)
    local ctrl = love.keyboard.isDown("lctrl") or love.keyboard.isDown("rctrl")
    local key_pressed = love.keyboard.isDown(key)

    if ctrl and key_pressed then
        if not ctrl_key_pressed then
            ctrl_key_pressed = true
            return true
        end
    else
        ctrl_key_pressed = false
    end
    return false
end

-- Handle Ctrl+G for manual action trigger
function InputHandler.handleCtrlG(callback)
    if InputHandler.checkCtrlKey("g") then
        local current_time = love.timer.getTime()

        if last_trigger_time == nil then
            last_trigger_time = current_time
        elseif current_time - last_trigger_time >= trigger_cooldown then
            print("Manual trigger activated via Ctrl+G")
            if callback then
                local success, result = pcall(callback)
                if not success then
                    print("ERROR in Ctrl+G callback: " .. tostring(result))
                end
            end
            last_trigger_time = current_time
        else
            print("Ctrl+G pressed: Cooldown active.")
        end
    end
end

-- Handle Ctrl+P for printing hand state
function InputHandler.handleCtrlP()
    if InputHandler.checkCtrlKey("p") then
        InputHandler.printHandState()
    end
end

-- Print current hand state
function InputHandler.printHandState()
    if not G or not G.hand or not G.hand.cards then
        print("ERROR: No valid hand found for printing")
        return
    end

    print("\nSelected Hand:")
    for _, card in ipairs(G.hand.highlighted) do
        print(string.format("Selected: %s", card.base.value .. card.base.suit))
    end

    local hand_state = GameState.getHandState()
    Utils.saveGameStateToJson(hand_state, "hand_state.json")
    if G.GAME and G.GAME.blind then
        Utils.saveGameStateToJson(G.GAME.blind, "G.GAME.blind.json")
    end
    print("Hand state saved to JSON files")
end

-- Update input handler
function InputHandler.update(manual_trigger_callback)
    InputHandler.handleCtrlG(manual_trigger_callback)
    InputHandler.handleCtrlP()
end

-- Get input handler status
function InputHandler.getStatus()
    return {
        ctrl_key_pressed = ctrl_key_pressed,
        last_trigger_time = last_trigger_time,
        trigger_cooldown = trigger_cooldown
    }
end

-- #endregion


-- #region MAIN AGENT

Agent = {
    stored_data = { current_state = nil, action_taken = nil },
    prev_game_state = nil,
    game_over_time = nil,
    selection_count = 0,
    is_initialized = false
}

local prev_State = G.STATES.SPLASH

-- Initialize all modules and agent
function Agent.init()
    print("=== Initializing Consolidated Agent ===")

    StateMachine.init()
    RewardCalculator.init()
    InputHandler.init()

    if not HttpClient.testConnection() then
        print("WARNING: Server connection test failed")
    end

    StateMachine.onPhaseChange(function(old_phase, new_phase, reason)
        print(string.format("Agent Event: Phase %s -> %s (%s)", old_phase, new_phase, reason))
    end)

    Agent.is_initialized = true
    print("Agent initialization complete")
end

-- Handle game state changes
function Agent.handleGameStateChange(old_state, new_state)
    local state_names = {
        [1] = "SELECTING_HAND",
        [2] = "HAND_PLAYED",
        [3] = "DRAW_TO_HAND",
        [4] = "GAME_OVER",
        [5] = "SHOP",
        [6] = "PLAY_TAROT",
        [7] = "BLIND_SELECT",
        [8] = "ROUND_EVAL",
        [9] = "TAROT_PACK",
        [10] = "PLANET_PACK",
        [11] = "MENU",
        [12] = "TUTORIAL",
        [13] = "SPLASH",
        [14] = "SANDBOX",
        [15] = "SPECTRAL_PACK",
        [16] = "DEMO_CTA",
        [17] = "STANDARD_PACK",
        [18] = "BUFFOON_PACK",
        [19] = "NEW_ROUND"
    }


    print(string.format("Game State Change: %s (%s) -> %s (%s)",
        tostring(old_state), state_names[old_state] or "Unknown",
        tostring(new_state), state_names[new_state] or "Unknown"))

    if new_state == G.STATES.SELECTING_HAND then
        Agent.selection_count = Agent.selection_count + 1
        print("Entered SELECTING_HAND state. Count: " .. Agent.selection_count)
    end
end

-- Handle menu state
function Agent.handleMenuState()
    if StateMachine.isInPhase(StateMachine.PHASES.IDLE) and G.STATE == G.STATES.MENU then
        print("Starting new game from menu")
        RewardCalculator.resetForNewSession()
        G.FUNCS.start_run()
        StateMachine.transitionTo(StateMachine.PHASES.SELECTING_BLIND, "Started new game")
    end
end

-- Handle game over state
function Agent.handleGameOverState()
    local current_time = love.timer.getTime()

    -- Only transition to GAME_OVER phase if we're not already transitioning out of it
    if not StateMachine.isInPhase(StateMachine.PHASES.GAME_OVER) then
        print("Game reached GAME_OVER state")
        StateMachine.transitionTo(StateMachine.PHASES.GAME_OVER, "Game ended")
        Agent.game_over_time = current_time
        return
    end

    if Agent.game_over_time == nil then
        Agent.game_over_time = current_time
    end

    local time_in_game_over = current_time - Agent.game_over_time
    if time_in_game_over > 1.0 then
        print("Game over timeout reached. Starting new run.")
        RewardCalculator.resetForNewRun()
        Agent.game_over_time = nil
        G.FUNCS.start_setup_run()
        StateMachine.transitionTo(StateMachine.PHASES.SELECTING_BLIND, "New run after timeout")
    end
end

-- Handle blind selection
function Agent.handleBlindSelection()
    if not StateMachine.isInPhase(StateMachine.PHASES.SELECTING_BLIND) or G.STATE ~= G.STATES.BLIND_SELECT then
        return
    end

    if G.blind_select and G.GAME.blind_on_deck and G.blind_select_opts then
        local blind_to_select = G.GAME.blind_on_deck
        local blind_option = G.blind_select_opts[string.lower(blind_to_select)]

        if blind_option then
            local select_button = blind_option:get_UIE_by_ID('select_blind_button')
            if select_button and select_button.config.button == 'select_blind' then
                print("Auto-selecting " .. blind_to_select .. " blind")
                G.FUNCS.select_blind(select_button)
                StateMachine.transitionTo(StateMachine.PHASES.SELECTING_HAND, "Blind selected")
            end
        end
    end
end

-- Handle hand selection and action execution
function Agent.handleHandSelection()
    -- Handle ACTION_PENDING phase - transition to WAITING_FOR_REWARD
    if StateMachine.isInPhase(StateMachine.PHASES.ACTION_PENDING) then
        StateMachine.transitionTo(StateMachine.PHASES.WAITING_FOR_REWARD, "Action pending completed")
        return
    end

    if not StateMachine.isInPhase(StateMachine.PHASES.SELECTING_HAND) or G.STATE ~= G.STATES.SELECTING_HAND then
        return
    end

    if not GameState.isValid() then
        print("ERROR: Invalid game state for action")
        return
    end

    RewardCalculator.storePreActionState()
    Agent.stored_data.current_state = GameState.process()
    print("Stored current state for training")

    local action_data = HttpClient.getAction(Agent.stored_data.current_state)
    if not action_data then
        print("ERROR: Failed to get action from server")
        os.exit(1, true)
        return
    end

    local valid, error_msg = ActionExecutor.validateActionData(action_data)
    if not valid then
        print("ERROR: Invalid action data - " .. error_msg)
        os.exit(1, true)
        return
    end

    Agent.stored_data.action_taken = action_data
    print("Stored action for training: " .. action_data.action)

    if ActionExecutor.execute(action_data) then
        StateMachine.transitionTo(StateMachine.PHASES.ACTION_PENDING, "Action executed")
    else
        print("ERROR: Failed to execute action")
        StateMachine.transitionTo(StateMachine.PHASES.IDLE, "Action execution failed")
    end
end

-- Handle reward calculation and training
function Agent.handleRewardCalculation()
    if not StateMachine.isInPhase(StateMachine.PHASES.WAITING_FOR_REWARD) then
        return
    end

    local ready_to_calculate = false

    if prev_State ~= G.STATE then
        print("Game state changed, checking reward conditions...")
        print("G.STATE: " .. G.STATE)
        print("prev_State: " .. prev_State)
        if G.STATE == G.STATES.SELECTING_HAND or
            G.STATE == G.STATES.ROUND_EVAL or
            G.STATE == G.STATES.NEW_ROUND or
            G.STATE == G.STATES.GAME_OVER then
            ready_to_calculate = true
            print("Reward condition met: " .. tostring(G.STATE))
        end
    end

    if ready_to_calculate and Agent.stored_data.current_state and Agent.stored_data.action_taken then
        StateMachine.transitionTo(StateMachine.PHASES.PROCESSING_REWARD, "Ready to process reward")

        local next_state = GameState.process()
        local reward, done = RewardCalculator.calculate()
        RewardCalculator.updateTracking(reward, done)

        -- if RewardCalculator.checkNewBest() then
        --     if HttpClient.saveModel() then
        --         RewardCalculator.incrementSaveCount()
        --     end
        -- end

        print("sending training data")
        print("Agent.done", tostring(done))

        HttpClient.sendTrainingData(
            Agent.stored_data.current_state,
            Agent.stored_data.action_taken,
            reward,
            next_state,
            done
        )

        Agent.stored_data.current_state = nil
        Agent.stored_data.action_taken = nil

        if done then
            print("Episode complete (done=true)")
            StateMachine.transitionTo(StateMachine.PHASES.GAME_OVER, "Episode complete")
        else
            Utils.createDelayedEvent(0.1, function()
                StateMachine.transitionTo(StateMachine.PHASES.SELECTING_HAND, "Ready for next action")
                return true
            end, "Transition to next action")
        end
    end
end

-- Main update function
function Agent.update(dt)
    if not Agent.is_initialized then
        return
    end

    local game_state_changed = (prev_State ~= G.STATE)
    if game_state_changed then
        Agent.handleGameStateChange(prev_State, G.STATE)
    end

    if G.STATE == G.STATES.MENU then
        Agent.handleMenuState()
    elseif (G.STATE == G.STATES.GAME_OVER or G.STATE == G.STATES.ROUND_EVAL) and
        not StateMachine.isInPhase(StateMachine.PHASES.SELECTING_BLIND) and
        not StateMachine.isInPhase(StateMachine.PHASES.IDLE) then
        Agent.handleGameOverState()
    elseif G.STATE == G.STATES.BLIND_SELECT then
        Agent.handleBlindSelection()
    elseif G.STATE == G.STATES.SELECTING_HAND then
        Agent.handleHandSelection()
    end

    Agent.handleRewardCalculation()

    InputHandler.update(function()
        print("Manual trigger activated via Ctrl+G")
    end)

    if G.STATE ~= G.STATES.GAME_OVER and Agent.game_over_time then
        Agent.game_over_time = nil
    end

    if game_state_changed then
        prev_State = G.STATE
    end
end

-- Public API for debugging
function Agent.getStatus()
    return {
        initialized = Agent.is_initialized,
        phase = StateMachine.getPhase(),
        phase_time = StateMachine.getPhaseTime(),
        selection_count = Agent.selection_count,
        reward_stats = RewardCalculator.getStats(),
        game_state_info = GameState.getStateInfo(),
        input_status = InputHandler.getStatus()
    }
end

-- #endregion

-- Debug functions
function PRINT_HAND()
    InputHandler.printHandState()
end

function GET_AGENT_STATUS()
    return Agent.getStatus()
end

-- Initialize last values for compatibility
G.last_chips = 0

-- Make Agent globally accessible
_G.Agent = Agent

print("Consolidated Agent System loaded successfully")
print("Use GET_AGENT_STATUS() to check agent status")
print("Use PRINT_HAND() to print current hand state")
