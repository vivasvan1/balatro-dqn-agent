-- raise test error
-- error("test error") this is line 1137 in main.lua

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

Lovely = require("lovely")
HandObserver = {}

Agent = {
    -- Private state
    _is_file_written = false,
    _ctrl_g_pressed = false,
    phase = "IDLE", -- Possible phases: "IDLE", "ACTION_PENDING", "WAITING_FOR_REWARD"
    stored_data = {
        current_state = nil,
        action_taken = nil,
        pre_action_chips = 0,
    }
}


-- Helper function to serialize tables
local function serializeTable(tbl, visited, path)
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
            result = result .. serializeTable(v, visited, new_path)
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
local function saveGameStateToJson(data, filename)
    local json_path = Lovely.mod_dir .. "/balatro_agent/" .. filename
    local file = nativefs.newFile(json_path)
    local json_data = serializeTable(data)
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

-- Helper function to select cards by indices
function Agent._select_hand_cards_by_indices(indices)
    if not G or not G.hand or not G.hand.cards then
        print("No valid hand found.")
        return
    end

    -- First unhighlight all cards
    G.hand:unhighlight_all()

    -- Validate indices
    for _, idx in ipairs(indices) do
        if type(idx) ~= "number" or idx < 1 or idx > #G.hand.cards then
            print("Invalid index: " .. tostring(idx))
            return
        end
    end

    -- -- Print whole hand
    -- print("\nWhole Hand:")
    -- for i, card in ipairs(G.hand.cards) do
    --     print(string.format("Card %d: %s", i, card.base.value .. card.base.suit))
    -- end

    -- Highlight cards at specified indices
    for _, idx in ipairs(indices) do
        local card = G.hand.cards[idx]
        -- Call the highlight method on the card
        card:highlight(true)
        -- Add to highlighted array - makes sure internal tracking is correct
        table.insert(G.hand.highlighted, card)
    end

    -- If we're selecting a hand, make sure to parse the highlighted cards
    if G.STATE == G.STATES.SELECTING_HAND then
        G.hand:parse_highlighted()
    end


    print("\nHighlighted " .. #indices .. " cards in hand.")
end

-- Helper function to select random hand cards
function Agent._select_random_hand_cards()
    if not G or not G.hand or not G.hand.cards or #G.hand.cards < 5 then
        print("Not enough cards in hand to select 5.")
        return
    end

    -- Generate random indices
    local indices = {}
    for i = 1, #G.hand.cards do table.insert(indices, i) end
    for i = #indices, 2, -1 do
        local j = math.random(i)
        indices[i], indices[j] = indices[j], indices[i]
    end

    -- Select first 5 random cards
    local selected_indices = {}
    local max_cards = math.min(5, #G.hand.cards)
    for i = 1, max_cards do
        table.insert(selected_indices, indices[i])
    end

    -- Use the new function to select the cards
    Agent._select_hand_cards_by_indices(selected_indices)

    -- G.FUNCS.discard_cards_from_highlighted()
end

function PRINT_HAND()
    -- Print selected hand
    print("\nSelected Hand:")
    for _, card in ipairs(G.hand.highlighted) do
        print(string.format("Selected: %s", card.base.value .. card.base.suit))
    end

    -- Save hand state to JSON
    local hand_state = {
        whole_hand = {},
        selected_hand = {}
    }

    -- Add whole hand to state
    for i, card in ipairs(G.hand.cards) do
        table.insert(hand_state.whole_hand, {
            index = i,
            value = card.base.value,
            suit = card.base.suit
        })
    end

    -- Add selected hand to state
    for _, card in ipairs(G.hand.highlighted) do
        table.insert(hand_state.selected_hand, {
            value = card.base.value,
            suit = card.base.suit
        })
    end

    saveGameStateToJson(G.GAME.blind, "G.GAME.blind.json")

    -- Save to JSON file
    saveGameStateToJson(hand_state, "hand_state.json")
end

-- Configuration
local SERVER_URL = "http://localhost:5000"
local PREDICT_ENDPOINT = SERVER_URL .. "/predict"
local TRAIN_ENDPOINT = SERVER_URL .. "/train"
local SAVE_ENDPOINT = SERVER_URL .. "/save"

-- State processor for converting game state to vector
local function process_game_state()
    local state = {}
    local idx = 1


    -- Add game state info
    state[idx] = G.GAME.chips

    idx = idx + 1

    -- Add whole hand to state

    local value_to_letter = {
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

    local suit_to_letter = {
        ["Diamonds"] = "D",
        ["Hearts"] = "H",
        ["Clubs"] = "C",
        ["Spades"] = "S"
    }

    state[idx] = G.GAME.current_round.hands_left
    idx = idx + 1
    state[idx] = G.GAME.current_round.discards_left
    idx = idx + 1



    for i, card in ipairs(G.hand.cards) do
        state[idx] = i
        state[idx + 1] = value_to_letter[card.base.value] .. suit_to_letter[card.base.suit]
        idx = idx + 2
    end

    return state
end

-- Function to get action from server
local function get_action()
    local state = process_game_state()
    local request_body = Json.encode({
        state = state
    })

    local response = Http.post(PREDICT_ENDPOINT, {
        headers = {
            ["Content-Type"] = "application/json"
        },
        body = request_body
    })

    if response.status == 200 then
        -- Extract just the JSON part from the response body
        local json_start = response.body:find("{")
        local json_end = response.body:find("}")
        if json_start and json_end then
            local json_str = response.body:sub(json_start, json_end)
            local data = Json.decode(json_str)
            if data and data.action ~= nil and data.indices ~= nil then
                print("Action received: " .. tostring(data.action))
                print("Indices recieved: " .. serializeTable(data.indices))
                return data
            end
        end

        print("Failed to parse action from response: " .. response.body)
        return nil
    else
        print("Error getting action. Status: " .. response.status .. ", Body: " .. response.body)
        return nil
    end
end

-- Function to send training data to server
local function send_training_data(state, action, reward, next_state, done)
    local request_body = Json.encode({
        state = state,
        action = action,
        reward = reward,
        next_state = next_state,
        done = done
    })

    local response = Http.post(TRAIN_ENDPOINT, {
        headers = {
            ["Content-Type"] = "application/json"
        },
        body = request_body
    })

    if response.status ~= 200 then
        print("Error sending training data:", response.body)
    end
end

-- Function to save model
local function save_model()
    local response = Http.post(SAVE_ENDPOINT)

    if response.status == 200 then
        print("Model saved successfully")
    else
        print("Error saving model:", response.body)
    end
end


-- Function to execute an action
local function execute_action(action_data)
    if not action_data or not action_data.indices or not action_data.action then
        print("Invalid action data received: 'action' or 'indices' field missing or action_data is nil")
        if action_data then
            print("Received action_data.action: " .. tostring(action_data.action))
            print("Received action_data.indices: " .. serializeTable(action_data.indices))
        end
        return
    end

    -- First unhighlight all cards
    G.hand:unhighlight_all()

    -- Highlight the selected cards
    for _, idx in ipairs(action_data.indices) do
        if idx >= 1 and idx <= #G.hand.cards then -- Added check for idx >= 1
            local card = G.hand.cards[idx]
            card:highlight(true)
            table.insert(G.hand.highlighted, card)
            print(string.format("Selected card %d: %s", idx, card.base.value .. card.base.suit))
        else
            print(string.format("Invalid card index %d received. Hand size: %d", idx, #G.hand.cards))
        end
    end

    -- Make sure to parse the highlighted cards
    if G.STATE == G.STATES.SELECTING_HAND then
        G.hand:parse_highlighted()

        -- Automatically play or discard based on action_data.action
        if action_data.action == "play" then
            print("Playing selected cards")
            G.FUNCS.play_cards_from_highlighted()
        elseif action_data.action == "discard" then
            print("Discarding selected cards")
            G.FUNCS.discard_cards_from_highlighted()
        else
            print("Unknown action type in action_data.action: " .. tostring(action_data.action))
        end
    end

    print(string.format("Executed action: %s %d cards",
        action_data.action,
        #action_data.indices))
end

-- Helper to log current agent phase (optional)
local function log_agent_phase(new_phase, reason)
    if Agent.phase ~= new_phase then
        print(string.format("Agent Phase: %s -> %s. Reason: %s", Agent.phase, new_phase, reason or "N/A"))
        Agent.phase = new_phase
    end
end

-- Ensure calculate_reward is defined. This is the user's existing one.
-- It relies on G.last_chips and  being set correctly before the action for which reward is being calculated.
local function calculate_reward()
    local reward = 0
    -- Make sure G.last_hand_won is updated correctly by the game or elsewhere if used.
    -- For this example, let's assume it's managed by the game.
    -- if G.last_hand_won then reward = reward + 10 end -- This might need specific timing

    -- Reward based on chip and multiplier changes
    -- G.last_chips and  should reflect values *before* the action that led to current G.chips and G.mult
    reward = reward + (G.GAME.chips - G.last_chips) * 0.1 -- Assuming G.last_chips holds pre-action chips

    if G.STATE == G.STATES.GAME_OVER then                 -- Or G.GAME.game_over if that's the flag
        reward = reward - 20
    end

    -- Crucially, update G.last_chips *after* calculating reward,
    -- so they are set for the *next* action's reward calculation.
    -- This is typically done at the end of a turn or after reward calculation.
    -- The original calculate_reward did this.
    -- G.last_chips = G.GAME.chips
    --  = G.GAME.round_mult
    -- However, it's better to set G.last_chips when current_state is captured.
    -- Let's assume calculate_reward itself does not update G.last_chips/.
    -- We will manage G.last_chips/ explicitly.


    -- Base reward for winning with a hand
    if G.GAME.chips > G.GAME.blind.chips then
        reward = reward + 10
        print("Added +10 reward for winning hand")
    end

    -- Reward for chips gained
    local chip_difference = G.GAME.chips - G.last_chips
    reward = reward + chip_difference * 0.1
    print("Chip difference: " .. tostring(chip_difference))
    print("Added reward from chips: " .. tostring(chip_difference * 0.1))

    return reward
end

TRIGGER_COOLDOWN = 0.5

-- Main update function
function Agent.update(dt)
    local current_time = love.timer.getTime() -- For cooldowns

    -- Tracks if G.STATE has changed in this specific Agent.update() call
    local G_STATE_HAS_CHANGED_THIS_TICK = (prev_State ~= G.STATE)

    if G_STATE_HAS_CHANGED_THIS_TICK then
        local state_names_lookup = { -- Copied from user's code for logging
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
            tostring(prev_State), state_names_lookup[prev_State] or "Unknown",
            tostring(G.STATE), state_names_lookup[G.STATE] or "Unknown"))

        if G.STATE == G.STATES.SELECTING_HAND then
            NO_OF_TIMES = NO_OF_TIMES + 1 -- Your counter
            print("Entered SELECTING_HAND state. Overall Count: " .. tostring(NO_OF_TIMES))
        end
    end

    -- Agent's Turn / Action Initiation Logic
    if Agent.phase == "IDLE" and G.STATE == G.STATES.SELECTING_HAND then
        -- This condition means it's our time to act.
        -- Could be a fresh entry into SELECTING_HAND or we became IDLE while in it.
        log_agent_phase("ACTION_PENDING", "Idle in SELECTING_HAND")

        -- Store values *before* action for reward calculation
        -- These will be compared against G.GAME.chips and G.GAME.round_mult *after* the action.
        G.last_chips = G.GAME.chips


        Agent.stored_data.current_state = process_game_state()
        print("Stored current_state for training:", serializeTable(Agent.stored_data.current_state))

        local action_from_server = get_action() -- This is expected to be {indices = ..., action = ...}
        if action_from_server and action_from_server.indices and action_from_server.action then
            Agent.stored_data.action_taken = action_from_server
            print("Stored action_taken for training:", serializeTable(Agent.stored_data.action_taken))

            execute_action(action_from_server) -- This function actually plays/discards cards

            log_agent_phase("WAITING_FOR_REWARD", "Action executed")
            -- At this exact moment, G.STATE is likely still SELECTING_HAND.
            -- The game will transition in subsequent engine ticks due to execute_action.
        else
            print("Failed to get valid action from server or action was nil. Reverting to IDLE.")
            log_agent_phase("IDLE", "Failed to get action")
        end
    end

    -- Reward Calculation and Training Data Sending Logic
    if Agent.phase == "WAITING_FOR_REWARD" then
        local ready_to_calculate_reward = false
        local is_terminal_state_for_action = false

        -- Check conditions based on G.STATE changing *after* an action was taken.
        -- We are ready if the game has processed the action and settled into a new key state.
        if G_STATE_HAS_CHANGED_THIS_TICK then -- Only consider on actual game state changes
            if G.STATE == G.STATES.SELECTING_HAND then
                -- Returned to SELECTING_HAND: means previous play/discard completed, ready for next action or hand.
                ready_to_calculate_reward = true
                print("Reward condition: Re-entered SELECTING_HAND.")
            elseif G.STATE == G.STATES.ROUND_EVAL then
                -- Round evaluation state.
                ready_to_calculate_reward = true
                print("Reward condition: Entered ROUND_EVAL.")
            elseif G.STATE == G.STATES.GAME_OVER then
                -- Game over state.
                ready_to_calculate_reward = true
                is_terminal_state_for_action = true
                print("Reward condition: Entered GAME_OVER.")
            end
        end

        if ready_to_calculate_reward then
            if Agent.stored_data.current_state and Agent.stored_data.action_taken then
                log_agent_phase("PROCESSING_REWARD", "Ready to calculate reward")

                local next_state = process_game_state() -- State after action settled
                local reward = calculate_reward()       -- Uses G.last_chips set before action
                local done = is_terminal_state_for_action

                print("Calculated Reward:", reward, "Done:", done)
                print("Next State for training:", serializeTable(next_state))

                -- Python's /train endpoint expects action as a dict: {'indices': [...], 'action': "..."}
                -- Agent.stored_data.action_taken is already in this format.
                -- send_training_data(
                --     Agent.stored_data.current_state,
                --     Agent.stored_data.action_taken,
                --     reward,
                --     next_state,
                --     done
                -- )

                -- Clean up stored data for the next turn
                Agent.stored_data.current_state = nil
                Agent.stored_data.action_taken = nil
                log_agent_phase("IDLE", "Training data sent, cycle complete")
            else
                print("WARNING: Ready to calculate reward, but stored_data is missing. This shouldn't happen.")
                log_agent_phase("IDLE", "Error in reward phase, resetting")
            end
        end
    end

    -- Handle Ctrl+G keypress (Manual Trigger)
    if Agent._check_ctrl_key("g") then -- Assuming Agent._check_ctrl_key exists and works for "g"
        if current_time - last_trigger_time >= TRIGGER_COOLDOWN then
            last_trigger_time = current_time
            if Agent.phase == "IDLE" and G.STATE == G.STATES.SELECTING_HAND then
                print(
                    "Ctrl+G pressed: Agent is IDLE and in SELECTING_HAND. Will initiate action in this tick if not already.")
                -- The main logic for "IDLE and G.STATE == G.STATES.SELECTING_HAND" will cover this.
                -- No explicit action needed here other than resetting cooldown.
            elseif Agent.phase ~= "IDLE" then
                print("Ctrl+G pressed: Agent busy (Phase: " .. Agent.phase .. "). Try again later.")
            elseif G.STATE ~= G.STATES.SELECTING_HAND then
                print("Ctrl+G pressed: Not in SELECTING_HAND state (Current: " .. G.STATE .. "). Try again later.")
            end
        else
            -- print("Ctrl+G pressed: Cooldown active.") -- Optional: for spammy log
        end
    end

    -- Handle Ctrl+P for printing hand
    if Agent._check_ctrl_key("p") then -- Assuming Agent._check_ctrl_key exists
        PRINT_HAND()                   -- Assuming PRINT_HAND is defined elsewhere
    end

    -- IMPORTANT: Update prev_State *after* all logic that depends on its value from the start of the tick
    if G_STATE_HAS_CHANGED_THIS_TICK then
        prev_State = G.STATE
    end
end

-- Helper function to check Ctrl+key
function Agent._check_ctrl_key(key)
    local ctrl = love.keyboard.isDown("lctrl") or love.keyboard.isDown("rctrl")
    local key_pressed = love.keyboard.isDown(key)

    if ctrl and key_pressed and not Agent._ctrl_key_pressed then
        print("Ctrl+" .. key .. " detected!")
    end

    if ctrl and key_pressed then
        if not Agent._ctrl_key_pressed then
            Agent._ctrl_key_pressed = true
            return true
        end
    else
        Agent._ctrl_key_pressed = false
    end
    return false
end

-- Hook into game events
local function init()
    -- Register for game update events
    -- G.FUNCS.game_update = on_game_update

    -- Save model periodically (every 100 games)
    -- local games_played = 0
    -- G.FUNCS.game_end = function()
    --     games_played = games_played + 1
    --     if games_played % 100 == 0 then
    --         save_model()
    --     end
    -- end
end

-- Main initialization function
function Agent.init()
    print("Agent.init() called")

    -- Save initial game state
    -- saveGameStateToJson(G, "G.json")
    -- saveGameStateToJson(G.FUNCS, "G.FUNCS.json")

    -- Load and initialize HandObserver
    local main_logic_path = Lovely.mod_dir .. "hand_observer_main.lua"
    if nativefs.getInfo(main_logic_path) then
        assert(load(nativefs.read(main_logic_path)))()
        if HandObserver and HandObserver.init then
            HandObserver.init()
            print(
                "HandObserver initialized. Ensure lovely.toml patches an update loop to call HandObserver.observe_hand().")
        else
            print("Failed to initialize HandObserver: HandObserver.init not found after loading main logic.")
        end
    else
        print("Error: Could not find HandObserver/hand_observer_main.lua at path: " .. main_logic_path)
    end

    -- Initialize the agent
    init()
end

local prev_State = G.STATES.SPLASH;

NO_OF_TIMES = 0
IS_UPDATING = false;

-- -- Main update function
-- function Agent.update(dt)
--     if (prev_State ~= G.STATE) then
--         print("State changed from " .. tostring(prev_State) .. " to " .. tostring(G.STATE))
--         local state_names = {
--             [1] = "SELECTING_HAND",
--             [2] = "HAND_PLAYED",
--             [3] = "DRAW_TO_HAND",
--             [4] = "GAME_OVER",
--             [5] = "SHOP",
--             [6] = "PLAY_TAROT",
--             [7] = "BLIND_SELECT",
--             [8] = "ROUND_EVAL",
--             [9] = "TAROT_PACK",
--             [10] = "PLANET_PACK",
--             [11] = "MENU",
--             [12] = "TUTORIAL",
--             [13] = "SPLASH",
--             [14] = "SANDBOX",
--             [15] = "SPECTRAL_PACK",
--             [16] = "DEMO_CTA",
--             [17] = "STANDARD_PACK",
--             [18] = "BUFFOON_PACK",
--             [19] = "NEW_ROUND"
--         }

--         print("State changed to: " .. (state_names[G.STATE] or "Unknown State"))
--         prev_State = G.STATE

--         if (G.STATE == G.STATES.SELECTING_HAND) then
--             NO_OF_TIMES = NO_OF_TIMES + 1
--             print("calculate and playing hand : " .. tostring(NO_OF_TIMES))
--         end
--     end

--     -- -- Handle Ctrl+G keypress
--     -- if Agent._check_ctrl_key("g") then
--     --     -- Agent._select_random_hand_cards()
--     --     if IS_UPDATING == false then
--     --         IS_UPDATING = true
--     --         on_game_update()
--     --         IS_UPDATING = false
--     --     end
--     -- end

--     -- if Agent._check_ctrl_key("p") then
--     --     PRINT_HAND()
--     -- end




--     -- Save game stages state once
--     -- if G.STAGE == G.STAGES.RUN and not G.SETTINGS.paused and not G.OVERLAY_MENU and not Agent._is_file_written then
--     --     Agent._is_file_written = true
--     --     print(G.STAGES)
--     --     if type(G.STAGES) == "table" then
--     --         saveGameStateToJson(G.STAGES, "G.STAGES.json")
--     --     else
--     --         print("Error: G.STAGES is not a table and cannot be serialized.")
--     --     end
--     -- end
-- end

-- -- Initialize last values for reward calculation
-- G.last_chips = 0
--  = 1
-- G.last_hand_won = false

-- print("Agent module loaded, Agent.init() is ready.")
